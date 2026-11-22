"""MainWindow mix-in: markers / ground control, Optimize Cameras, Refine
Markers, point cloud filtering and tie point gradual selection."""
from __future__ import annotations

import copy
import json
import os

import numpy as np
from PySide6.QtWidgets import QMessageBox

from .filedialogs import QFileDialog

from . import colmap as cm
from . import georef as g
from .alignment_reference import gcp_crs, gcp_frame_matches_model, model_crs


GEOREF_STALE = "built before the last georeferencing (Optimize Cameras)"


def products_after_georef(ch, M, log=print, *, persist_texture=True):
    """Optimize Cameras moved the model, point cloud and mesh into a new frame by
    the similarity M (4x4). Lengths the texture measured in the old frame are
    scaled (else e.g. Build Orthomosaic defaults to a pixel size in the old
    units - issue #3), and rasters / tiles built in the old frame are marked
    outdated (they stay on disk until rebuilt). Returns the scale."""
    try:
        A = np.asarray(M, float)[:3, :3]
        s = float(abs(np.linalg.det(A)) ** (1.0 / 3.0))
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(s) or s <= 0:
        return 1.0
    t = ch.texture or {}
    if t and ch.model_ply and os.path.isfile(ch.model_ply):
        from .workflow import file_stamp
        t["mesh_stamp"] = file_stamp(ch.model_ply)
        path = os.path.join(t.get("dir") or "", "info.json")
        if persist_texture and os.path.isfile(path):
            try:
                with open(path,encoding="utf-8") as fh: info=json.load(fh)
                info["mesh_stamp"] = t["mesh_stamp"]
                with open(path+".identity.tmp","w",encoding="utf-8") as fh: json.dump(info,fh)
                os.replace(path+".identity.tmp",path)
            except (OSError,ValueError) as error:
                log(f"Could not refresh trusted transformed texture identity: {error}")
    if t and abs(s - 1.0) > 1e-9:
        for k in ("source_gsd", "texel_size"):
            if t.get(k):
                t[k] = float(t[k]) * s
        p = os.path.join(t.get("dir") or "", "info.json")
        if persist_texture and t.get("dir") and os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    info = json.load(f)
                for k in ("source_gsd", "texel_size"):
                    if info.get(k):
                        info[k] = float(info[k]) * s
                with open(p + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(info, f, indent=1)
                os.replace(p + ".tmp", p)
            except (OSError, ValueError) as e:
                log(f"Optimize Cameras: could not update {p}: {e}")
    notes = []
    for name, d in (("orthomosaic", ch.orthomosaic), ("DEM", ch.dem), ("DTM", ch.dtm), ("tiled model", ch.tiled_model)):
        if d and d.get("format") != "imported" and not d.get("stale"):
            d["stale"] = GEOREF_STALE; notes.append(name)
    if notes:
        log("Optimize Cameras: the " + ", ".join(notes) + " belonged to the previous frame - rebuild "
            + ("it" if len(notes) == 1 else "them"))
    return s


class ReferenceMixin:
    # ------------------------------------------------------------ helpers
    def _model_dir(self, ch=None):
        ch = ch or self.dataset
        wd = ch.workdir or (self.project.default_workdir(ch) if self.project.path else "")
        return os.path.join(wd, "sparse", "0") if wd else ""

    def _sparse_model(self):
        """COLMAP model of the active dataset (cached by file time)."""
        from . import colmap_io as cio
        d = self._model_dir()
        f = os.path.join(d, "images.bin")
        if not os.path.exists(f):
            return None
        key = (d, tuple((os.stat(os.path.join(d, name)).st_mtime_ns, os.stat(os.path.join(d, name)).st_size)
                        for name in ("images.bin", "points3D.bin", "cameras.bin")))
        if getattr(self, "_model_cache", (None,))[0] != key:
            self._model_cache = (key, cio.read_model(d))
        return self._model_cache[1]

    def _marker_objs(self):
        return [g.Marker(**m) for m in (self.dataset.markers or [])]

    def _invalidate_marker_error_rows(self, markers):
        """Recompute edited residuals in the current frame; never keep old error rows."""
        georef = self.dataset.georef
        if not georef:
            return
        georef.update(rows=[], control_rmse=None, check_rmse=None)
        georef['marker_image_errors'] = {}
        model = self._sparse_model()
        if model is None:
            return
        stamp = self._model_cache[0]
        if getattr(self, "_marker_residual_cache", (None,))[0] != stamp:
            self._marker_residual_cache = (stamp, {})
        try:
            from .photonames import photo_dimensions
            g.validate_projection_dimensions(model, photo_dimensions(self.dataset))
            estimates, pixels, counts, rows = g.marker_diagnostics(
                model, markers, g.CRS.parse(georef.get("crs") or self.dataset.crs),
                georef.get("offset"), georef.get("zone"), cache=self._marker_residual_cache[1], frame=georef.get("coordinate_frame"),
                compare_heights=(gcp_frame_matches_model(self.dataset)
                                 and georef.get('reference_heights_verified', True)))
        except (ValueError, np.linalg.LinAlgError) as error:
            self.log(f"GCP residuals unavailable after edit: {error}")
            return
        georef.update(rows=rows, control_rmse=g.rmse(rows, "control"), check_rmse=g.rmse(rows, "check"))
        georef['marker_image_errors'] = {label: {'pix': pixel, 'n': counts.get(label, 0)} for label, pixel in pixels.items()}
        for marker in markers:
            marker.est = estimates[marker.label].tolist() if marker.label in estimates else None

    def _marker_model_xyz(self, m, model, names):
        """Marker position in the model frame: reference (if georeferenced),
        else triangulated from its projections."""
        # Display the observed estimate, independently of the surveyed target.
        try:
            X, _n = g.triangulate_marker(model, m.projections, names, measured_only=True)
            if X is None:
                X, _n = g.triangulate_marker(model, m.projections, names)
        except (ValueError, np.linalg.LinAlgError):
            return None
        if X is not None:
            return X
        gr = self.dataset.georef or {}
        if gr.get("offset") is not None and m.ref is not None:
            try:
                if not gcp_frame_matches_model(self.dataset):
                    return None
                crs = g.CRS.parse(gr.get("crs") or self.dataset.crs)
                X = crs.to_metric(np.atleast_2d(np.asarray(m.ref, float)), gr.get("zone"), gr.get("coordinate_frame"))[0]
                return X - np.asarray(gr["offset"], float)
            except Exception:
                pass
        return None

    def _marker_positions(self):
        """Model-frame positions of the active dataset's markers (list of xyz).
        Safe to call from the view-loader thread. Results are cached per
        marker (projections/reference/model), so after an edit only that
        marker is re-triangulated instead of all of them (~2 s for 98)."""
        import json as _json
        try:
            model = self._sparse_model()
        except Exception:
            model = None
        if model is None:
            return None
        key0 = (self._model_cache[0], _json.dumps(self.dataset.georef.get("offset") if self.dataset.georef else None),
                self.dataset.crs, gcp_crs(self.dataset), _json.dumps((self.dataset.georef or {}).get("zone")),
                _json.dumps((self.dataset.georef or {}).get("coordinate_frame")),
                tuple((c.colmap_name, c.path, c.width, c.height) for c in self.dataset.cameras))
        cache = getattr(self, "_marker_xyz_cache", None)
        if cache is None or cache[0] != key0:
            from .photonames import photo_dimensions
            try:
                g.validate_projection_dimensions(model, photo_dimensions(self.dataset))
            except ValueError:
                return None
            cache = self._marker_xyz_cache = (key0, {})
        names = None
        pts = []                       # (label, xyz, enabled=control)
        for m in self._marker_objs():
            signature = (_json.dumps(m.ref), _json.dumps(m.projections, sort_keys=True))
            previous = cache[1].get(m.label)
            if previous is None or previous[0] != signature:
                names = names or g.name_index(model)
                previous = cache[1][m.label] = (signature, self._marker_model_xyz(m, model, names))
            if previous[1] is not None:
                pts.append((m.label, previous[1], bool(m.enabled)))
        return pts

    def _show_markers_3d(self, items):
        if not items:
            self.model_view.set_markers(None); return
        from .marker_style import marker_state
        self.model_view.marker_states = {m["label"]: marker_state(m) for m in self.dataset.markers}
        self.model_view.set_markers(np.array([x for _l, x, _e in items]), labels=[l for l, _x, _e in items],
                                    control=[e for _l, _x, e in items])

    def _update_view_markers(self):
        self._show_markers_3d(self._marker_positions())
        self._update_ortho_markers()

    def _marker_selection_changed(self):
        self._update_photo_markers()
        labels = self.markers_pane.selected_labels()
        self.ortho_view.set_active_marker(labels[0] if len(labels) == 1 else None)

    def _photo_gcp_context(self):
        """Source identities and calibrated photo models for display and filtering."""
        from .photonames import marker_camera_keys

        try:
            model = self._sparse_model()
        except (OSError, ValueError):
            model = None
        cameras = self.dataset.cameras
        key = (id(model), tuple((c.path, c.colmap_name, c.width, c.height) for c in cameras))
        cached = getattr(self, "_gcp_photo_context_cache", None)
        if cached is None or cached[0] != key:
            names = g.name_index(model) if model is not None else {}
            keys = marker_camera_keys(self.dataset)
            cached = self._gcp_photo_context_cache = (key, model, keys, names, {})
        return cached

    def _photo_gcp_location(self, marker, stored, index, stem, model, names, xyz):
        """(u, v, status) in original pixels, for flags, photo filtering and counts."""
        ignored = marker.disabled or stem in marker.ignored_photos
        p = stored.get(stem)
        if p is not None:
            uv = p[:2]
            state = ("pinned" if g.projection_is_pinned(p) else
                     "placed" if g.projection_is_refined(p) else "predicted")
        elif xyz is not None and stem in names:
            iid = names[stem]
            uv = g.predict_projection(model, iid, xyz)
            if uv is None:
                return None
            source = self.dataset.cameras[index]
            calibrated = model.cameras[model.images[iid].camera_id]
            if source.width and source.height:
                uv = (uv[0] * source.width / calibrated.width,
                      uv[1] * source.height / calibrated.height)
            state = "predicted"
        else:
            return None
        source = self.dataset.cameras[index]
        if not np.isfinite(uv).all() or (source.width and source.height and
                                        not (0 <= uv[0] < source.width and 0 <= uv[1] < source.height)):
            return None
        return float(uv[0]), float(uv[1]), "ignored" if ignored else state

    def _gcp_photo_locations(self, labels):
        """Compute all visible locations only for points requested by the GCP panel."""
        _, model, keys, names, cache = self._photo_gcp_context()
        positions = {label: xyz for label, xyz, _ in self._marker_positions() or ()}
        selected = set(labels)
        for marker in self._marker_objs():
            if marker.label not in selected:
                continue
            xyz = positions.get(marker.label)
            signature = (marker.projections, marker.disabled, marker.ignored_photos,
                         None if xyz is None else tuple(xyz))
            previous = cache.get(marker.label)
            if previous is None or previous[0] != signature:
                stored = {g._stem(name): value for name, value in marker.projections.items()}
                locations = {}
                for index, stem in keys.items():
                    location = self._photo_gcp_location(marker, stored, index, stem, model, names, xyz)
                    if location is not None:
                        locations[index] = location
                # Cache independent values since the current project projections remain editable.
                previous = cache[marker.label] = (copy.deepcopy(signature), locations)
            yield previous[1]

    def marker_prediction_count(self, label):
        return sum(state == "predicted" for locations in self._gcp_photo_locations([label])
                   for _, _, state in locations.values())

    def _update_photo_markers(self):
        k = getattr(self, "current_photo", -1)
        if self.photo_view.path is None or not (0 <= k < len(self.dataset.cameras)):
            return
        _, model, keys, names, _cache = self._photo_gcp_context()
        stem = keys.get(k)
        points = self._marker_objs()
        items, ignored_here, ignored_all = [], set(), set()
        if stem is not None:
            positions = {label: xyz for label, xyz, _ in self._marker_positions() or ()}
            for marker in points:
                stored = {g._stem(name): value for name, value in marker.projections.items()}
                location = self._photo_gcp_location(marker, stored, k, stem, model, names,
                                                    positions.get(marker.label))
                if location is not None:
                    items.append((marker.label, *location))
                if marker.disabled:
                    ignored_all.add(marker.label)
                if stem in marker.ignored_photos:
                    ignored_here.add(marker.label)
        for key in ("markers", "marker_labels"):
            self.photo_view.set_visible(key, self.layer_action(key).isChecked())
        self.photo_view.set_markers(items, self.markers_pane.current_label(),
                                    [m.label for m in points] if stem is not None else (),
                                    ignored_here=ignored_here, ignored_all=ignored_all)

    def toggle_marker(self, label, enabled):
        ms = self._marker_objs()
        self._markers_snapshot()
        for m in ms:
            if m.label == label:
                m.enabled = enabled
        self._commit_markers(ms)

    def add_marker(self):
        ms = self._marker_objs()
        n = len(ms) + 1
        labels = {m.label for m in ms}
        while f"point {n}" in labels:
            n += 1
        self._markers_snapshot()
        ms.append(g.Marker(label=f"point {n}"))
        self._commit_markers(ms)

    # ------------------------------------------------------------ import / export
    def change_gcp_coordinates(self):
        from .dialogs import GcpCoordinatesDialog

        ch = self.dataset
        current = gcp_crs(ch)
        dialog = GcpCoordinatesDialog(current, self)
        if not dialog.exec():
            return
        points = self._marker_objs()
        try:
            target = dialog.selected_crs()
            crs = g.CRS.parse(target)
            if crs == g.CRS.parse(current):
                self.statusBar().showMessage(f"GCP coordinate system is already {crs.name}", 4000)
                return
            for point in points:
                if point.ref is not None:
                    g.validate_control_reference(point, crs)
        except ValueError as error:
            QMessageBox.warning(self, dialog.windowTitle(), f"Cannot assign the GCP coordinate system. {error}")
            return
        self._markers_snapshot()
        ch.reference_settings["gcp_crs"] = target
        self._commit_markers(points, f"GCP coordinate system assigned: {crs.name}")
        self.statusBar().showMessage(f"GCP coordinate system assigned: {crs.name}", 4000)

    def import_markers(self, path=None):
        path = path or QFileDialog.getOpenFileName(
            self, "Import GCPs / Reference", "",
            "GCPs and reference (*.xml *.txt *.csv);;All files (*)")[0]
        if not path:
            return
        ch = self.dataset
        try:
            new, crs = g.import_markers(path, default_crs=ch.reference_settings.get("gcp_crs") or model_crs(ch))
        except Exception as e:
            QMessageBox.warning(self, "Import GCPs", f"Cannot read {path}:\n{e}"); return
        existing = self._marker_objs()
        supplied_refs = {point.label for point in new if point.ref is not None}
        if crs is not None and supplied_refs:
            if (crs != g.CRS.parse(gcp_crs(ch))
                    and any(point.ref is not None and point.label not in supplied_refs for point in existing)):
                QMessageBox.warning(self, "Import GCPs",
                    "The imported coordinates use a different system from the existing points. "
                    "Convert the imported XYZ to the GCP system or replace all existing reference coordinates.")
                return
        self._markers_snapshot()
        if crs is not None and supplied_refs:
            ch.reference_settings["gcp_crs"] = (f'LOCAL_CS["{crs.name}"]' if crs.kind == "local" else crs.name)
        self._commit_markers(g.merge_markers(existing, new))
        nproj = sum(len(m.projections) for m in new)
        self.log(f"Imported {len(new)} GCPs ({nproj} photo projections) from {path}; "
                 f"coordinate system: {crs.name if crs else 'unknown'}")
        self._show_dock("Markers", True)

    def export_markers(self):
        rows = (self.dataset.georef or {}).get("rows")
        if not self.dataset.markers:
            QMessageBox.information(self, "Export GCPs", "No GCPs."); return
        dst = QFileDialog.getSaveFileName(self, "Export GCPs", "gcps.txt",
                                          "Reference report (*.txt);;GCP XML (*.xml)")[0]
        if not dst:
            return
        if dst.lower().endswith(".xml"):
            g.write_markers_xml(dst, self._marker_objs(), gcp_crs(self.dataset))
        else:
            crs = g.CRS.parse(gcp_crs(self.dataset))
            if rows:
                g.write_report(dst, rows, crs)
            else:
                with open(dst, "w", encoding="utf-8") as f:
                    f.write(f"# CoordinateSystem: {crs.name}\n#Label\tX\tY\tZ\tEnabled\tProjections\n")
                    for m in self._marker_objs():
                        r = m.ref or ["", "", ""]
                        f.write(f"{m.label}\t{r[0]}\t{r[1]}\t{r[2]}\t{int(m.enabled)}\t{len(m.projections)}\n")
        self.log(f"Exported GCPs to {dst}")

    # ------------------------------------------------------------ optimize
    def optimize_cameras(self):
        if not self._require_saved("Optimize Cameras"):
            return None
        from copy import deepcopy
        from pathlib import Path
        import tempfile
        from .dialogs import OptimizeCamerasDialog
        from .alignment_reference import GEOREF_FIELDS, gps_output_crs
        from .workflow import publish_outputs
        ch = self.dataset
        if not ch.aligned_count() or not os.path.isdir(self._model_dir()):
            QMessageBox.information(self, "Optimize Cameras", "Align photos first."); return
        self._photo_map(ch)  # recover legacy aliases; ambiguous inputs remain excluded
        from .photonames import photo_dimensions
        try:
            model = self._sparse_model()
            if model is None or not model.images:
                raise ValueError("The saved alignment is missing or has no cameras. Align photos first.")
            sizes = photo_dimensions(ch)
            g.validate_projection_dimensions(model, sizes)
        except (OSError, ValueError, EOFError, KeyError) as error:
            QMessageBox.information(self, "Optimize Cameras", str(error)); return
        defaults = dict(ch.reference_settings)
        from .photo_gps import camera_gps, use_photo_gps
        defaults.setdefault("use_cameras", bool(camera_gps(ch)))
        try:
            dlg = OptimizeCamerasDialog(self, defaults, model=model)
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "Optimize Cameras", str(error))
            return
        if not dlg.exec():
            return
        s = dlg.settings()
        local_control_grid = s["local_control_grid"]
        original_crs = model_crs(ch)
        target_crs = original_crs
        control_target = s["use_markers"] and any(point.get("ref") is not None for point in ch.markers)
        if s.get("use_local_override", local_control_grid):
            target_crs = g.CRS.parse("LOCAL").name
        elif control_target:
            target_crs = gcp_crs(ch)
        elif s["use_cameras"]:
            target_crs = gps_output_crs(ch)
        crs = g.CRS.parse(target_crs)
        frame_changed = g.CRS.parse(original_crs) != crs
        gps_on = use_photo_gps(ch)
        requested_cameras = s["use_cameras"] if gps_on else ch.reference_settings.get("use_cameras", True)
        s["use_cameras"] = gps_on and s["use_cameras"]
        reference_settings = dict(s["accuracies"], use_cameras=requested_cameras,
                                  use_photo_gps=gps_on, use_markers=s["use_markers"],
                                  camera_height_reference=s.get("camera_height_reference", "unknown"),
                                  marker_height_reference=s.get("marker_height_reference", "unknown"),
                                  reported_gps_accuracy=s.get("reported_gps_accuracy", False),
                                  intrinsics=s.get("intrinsics", "f,cx,cy,k1-k3,p1,p2"))
        from .photo_gps import camera_accuracies
        wd = self.project.default_workdir(ch)
        cfg = {"model": self._model_dir(), "photo_sizes": sizes,
               "markers": ch.markers or [], "crs": target_crs,
               "use_markers": s["use_markers"], "use_cameras": s["use_cameras"],
               "camera_gps": camera_gps(ch) if s["use_cameras"] else {},
               "camera_accuracies": camera_accuracies(ch, reported=s.get("reported_gps_accuracy", False)) if s["use_cameras"] else {},
               "strict_heights": True,
               "accuracies": s["accuracies"], "bundle": s["bundle"], "tie_points": s["tie_points"],
               "fit_intrinsics": s["fit_intrinsics"], "offset": (ch.georef or {}).get("offset"),
               "intrinsics": s.get('intrinsics', 'f,cx,cy,k1-k3,p1,p2'),
               "adjustment_frame": s.get('adjustment_frame', 'projected'),
               "coordinate_frame": (ch.georef or {}).get('coordinate_frame'),
               "camera_height_reference": s.get('camera_height_reference', 'unknown'),
               "marker_height_reference": s.get('marker_height_reference', 'unknown')}
        if local_control_grid or frame_changed or bool(cfg['coordinate_frame']) != (cfg['adjustment_frame'] == 'enu'):
            cfg['offset'] = None
            cfg['coordinate_frame'] = None
        cfg["compare_marker_xyz"] = s["use_markers"] or g.CRS.parse(gcp_crs(ch)) == crs
        try:
            status = g.reference_status(model, self._marker_objs() if s["use_markers"] else [],
                                        crs, cfg["camera_gps"], detailed=True)
            if status["source"]:
                g.validate_adjustment_references(cfg, crs, status["source"], status["usable"], status["gps"])
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            QMessageBox.warning(self, "Optimize Cameras", str(error))
            return
        status["imported"] = len(ch.markers or [])
        if control_target and frame_changed and status["source"] != "GCPs" and not local_control_grid:
            QMessageBox.information(self, "Optimize Cameras",
                f"Only {status['usable']} usable GCPs. At least three non-collinear GCPs are needed to change the model CRS.")
            return
        ready = status["source"] is not None and (not local_control_grid or status["source"] == "GCPs")
        if not ready:
            if local_control_grid:
                QMessageBox.information(self, "Local survey grid",
                    f"{status['usable']} usable control points. To fit the local grid, pin at least "
                    "three non-collinear controls with XYZ on two or more aligned photos each.")
                return
            msg = (f"{status['imported']} GCPs imported; {status['references']} enabled with complete XYZ; "
                   f"{status['usable']} usable controls on at least two aligned photos; "
                   f"{status['gps']} camera GPS positions usable in the selected CRS.\n\n"
                   "Georeferencing needs at least 3 non-collinear controls or camera GPS positions. "
                   "Imported XYZ alone is insufficient: place each control on at least two aligned photos.\n\n"
                   "Optimize cameras in the existing frame instead? This uses tie points" +
                   (" and enabled GCP image observations" if s["use_markers"] else "") +
                   ", ignores reference coordinates and GPS, and does not georeference the model.")
            if not s["bundle"]:
                QMessageBox.information(self, "Optimize Cameras", msg +
                                        "\nEnable Bundle adjustment for optimization without georeferencing.")
                return
            if QMessageBox.question(self, "Optimize without georeferencing?", msg,
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
            cfg["local_only"] = True
            cfg["crs"] = original_crs
            cfg["offset"] = (ch.georef or {}).get("offset")

        staging = tempfile.TemporaryDirectory(prefix=".optimize-", dir=wd)
        stage = Path(staging.name)
        (stage / "sparse_txt").mkdir()
        cfg.update(stage=str(stage), report=str(stage / "georef_report.txt"), result=str(stage / "georef.json"))
        replacements, copies = {}, {}
        geometry_target = Path(wd) / ("adjusted-geometry-" + stage.name.removeprefix(".optimize-"))
        if not cfg.get("local_only"):
            for key, name in (("point_cloud_ply", "cloud.ply"), ("model_ply", "mesh.ply")):
                value = getattr(ch, key)
                if value and os.path.isfile(value):
                    source = os.path.normcase(os.path.realpath(value))
                    staged = copies.setdefault(source, str(stage / "geometry" / name))
                    replacements[key] = str(geometry_target / Path(staged).name)
        if copies:
            (stage / "geometry").mkdir()
        cfg["transform_copies"] = copies
        fused = Path(wd) / "dense" / "fused.ply"
        staged_fused = stage / "managed-fused.ply"
        if (not cfg.get("local_only") and fused.is_file() and not fused.is_symlink()
                and fused.resolve().is_relative_to(Path(wd).resolve())
                and os.path.normcase(os.path.realpath(fused)) not in copies
                and Path(str(fused) + ".vis").is_file()):
            copies[os.path.normcase(os.path.realpath(fused))] = str(staged_fused)
        path = os.path.join(wd, "georef_task.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, default=float)
        title = "Optimizing cameras (local bundle adjustment)" if cfg.get("local_only") else "Optimizing cameras (georeferencing)"
        steps = [(title, [cm.PYTHON, "georef", path], 85)] + \
            self._export_sparse_steps(str(stage))
        result = {}

        def activate(res):
            old_frame = ch.georef or {}
            for key, new_path in replacements.items():
                setattr(ch, key, new_path)
            if not s["use_cameras"] or not s["bundle"]:
                reference_settings.pop("reported_gps_accuracy", None)
            ch.reference_settings.update(reference_settings)
            if res.get("similarity") is not None:
                products_after_georef(ch, res["similarity"], self.log, persist_texture=False)
            if not res.get("local_only"):
                if s["use_markers"] and (status["source"] == "GCPs" or
                                         (s["bundle"] and status["usable"])):
                    ch.reference_settings["gcp_crs"] = target_crs
                elif ch.markers:
                    ch.reference_settings.setdefault("gcp_crs", original_crs or "LOCAL")
                ch.georef = {k: res.get(k) for k in GEOREF_FIELDS}
                ch.crs = res.get("crs") or ch.crs
                for r in ch.georef.get("rows") or []:
                    r["pix"] = ((res.get("markers") or {}).get(r["label"]) or {}).get("pix")
                    r["n"] = ((res.get("markers") or {}).get(r["label"]) or {}).get("n")
                if ch.shapes and (frame_changed or local_control_grid or not old_frame
                                  or old_frame.get("coordinate_frame") != res.get("coordinate_frame")
                                  or old_frame.get("zone") != res.get("zone")):
                    ch.shapes["stale_frame"] = "These shapes use the previous project frame. Reimport them in the current coordinates."
                    self.log(ch.shapes["stale_frame"])
            else:
                # Existing CRS/offset still describe the same frame, but old errors no longer apply.
                if ch.georef:
                    ch.georef.update(rows=[], control_rmse=None, check_rmse=None, marker_image_errors={})
                for d in (ch.orthomosaic, ch.dem, ch.dtm, ch.tiled_model):
                    if d and d.get("format") != "imported":
                        d["stale"] = "built before camera optimization"
                self.log("Cameras optimized without georeferencing. Rebuild dense products to use the new cameras.")
            ms = self._marker_objs()
            for m in ms:
                e = (res.get("markers") or {}).get(m.label, {})
                m.est = e.get("est")
            ch.markers = [m.to_dict() for m in ms]
            self._invalidate_camera_products(ch)
            self._reload_alignment(ch, wd)
            if res.get("local_only") and ch.georef:
                self._commit_markers(self._marker_objs())
            c, k = res.get("control_rmse"), res.get("check_rmse")
            msg = ("Cameras optimized in the existing frame. " if res.get("local_only") else "Georeferenced. ") + (f"Control RMSE {c['total']:.3f} m (n={c['n']}). " if c else "") + \
                (f"Check RMSE {k['total']:.3f} m (n={k['n']})." if k else "")
            self.log(msg + f" Report: {os.path.join(wd, 'georef_report.txt')}")
            self.statusBar().showMessage(msg, 15000)

        def loaded():
            try:
                with open(cfg["result"], encoding="utf-8") as f:
                    res = json.load(f)
                camera_list = stage / "sparse_txt" / "images.txt"
                if not camera_list.is_file() or not cm.read_images_txt(str(camera_list)):
                    raise ValueError("The newly exported camera list is empty or missing.")
                if not (stage / "sparse_txt" / "points3D.txt").is_file():
                    raise ValueError("The newly exported tie-point list is missing.")
                from .plyio import ply_counts
                ply_counts(str(stage / "tie_points.ply"))
                pairs = [
                    (stage / "sparse" / "0", cfg["model"]),
                    (stage / "sparse_txt", Path(wd) / "sparse_txt"),
                    (stage / "tie_points.ply", Path(wd) / "tie_points.ply"),
                    (Path(cfg["report"]), Path(wd) / "georef_report.txt"),
                    (Path(cfg["result"]), Path(wd) / "georef.json"),
                ]
                if replacements:
                    pairs.append((stage / "geometry", geometry_target))
                if staged_fused.is_file():
                    pairs.append((staged_fused, fused))
                snapshot = deepcopy(ch.__dict__)
                was_modified = self.project.modified
                try:
                    publish_outputs(pairs, activate=lambda: activate(res))
                except Exception:
                    ch.__dict__.update(snapshot)
                    self.project.modified = was_modified
                    self.reload_view()
                    raise
                result.update(res)
            finally:
                staging.cleanup()

        def meta(dt, stages):
            from . import procmeta
            res = result
            procmeta.record(ch, "camera_optimization" if cfg.get("local_only") else "georeference", params={k: cfg[k] for k in ("crs", "use_markers", "use_cameras",
                                                                             "accuracies", "bundle", "tie_points",
                                                                             "fit_intrinsics")},
                            seconds=dt, stages=stages, control_rmse=res.get("control_rmse"),
                            check_rmse=res.get("check_rmse"), bundle=res.get("bundle"),
                            transformed=list(replacements.values()))
            ch.meta.pop("sparse_stats", None)
        self._run("Optimize Cameras", steps, loaded, meta, on_fail=staging.cleanup, params=s)

    def _invalidate_camera_products(self, ch):
        """Camera refinement invalidates camera-derived products, not imported rasters."""
        reason = "Cameras changed - rebuild this product"
        for key in ("point_cloud", "model", "texture", "depth_maps"):
            if key in ch.meta:
                ch.meta[key]["stale"] = reason
        if ch.model_ply:
            ch.meta.setdefault("model", {})["stale"] = reason
        for d in (ch.texture, ch.tiled_model, ch.orthomosaic, ch.dem, ch.dtm):
            if d and d.get("format") != "imported":
                d["stale"] = reason
        if hasattr(self, "_update_stale_banners"):
            self._update_stale_banners()

    def _export_sparse_steps(self, wd):
        sp = os.path.join(wd, "sparse", "0")
        return [("Exporting tie points", ["model_converter", "--input_path", sp, "--output_path",
                                          os.path.join(wd, "tie_points.ply"), "--output_type", "PLY"], 8),
                ("Exporting cameras", ["model_converter", "--input_path", sp, "--output_path",
                                       os.path.join(wd, "sparse_txt"), "--output_type", "TXT"], 7)]

    def _reload_alignment(self, ch, wd):
        poses = cm.read_images_txt(os.path.join(wd, "sparse_txt", "images.txt"))
        from .photonames import camera_for_names, recover_names
        if any(c.aligned and not c.colmap_name for c in ch.cameras):
            recover_names(ch, wd)
        cams, ambiguous = camera_for_names(ch, list(poses))      # exact names, not file names
        if ambiguous:
            self._warn_ambiguous(ambiguous)
        by_cam = {id(c): poses[n] for n, c in cams.items()}
        for c in ch.cameras:
            p = by_cam.get(id(c))
            c.aligned = p is not None
            c.est = p[0].tolist() if p else None
            c.rot = p[1].tolist() if p else None
        ch.tie_points_ply = os.path.join(wd, "tie_points.ply")
        ch.tie_point_count = cm.count_points3d(os.path.join(wd, "sparse_txt", "points3D.txt"))
        self.model_view.set_z_up(bool(ch.georef))
        self.set_modified(); self.reload_view()
        self._update_view_markers()

    # ------------------------------------------------------------ refine
    def refine_markers(self):
        if not self._require_saved("Refine GCPs"):
            return None
        ch = self.dataset
        if not ch.markers or not os.path.isdir(self._model_dir()):
            QMessageBox.information(self, "Refine GCPs", "Align photos and import GCPs first."); return
        from .photonames import photo_dimensions
        try:
            g.validate_projection_dimensions(self._sparse_model(), photo_dimensions(ch))
        except ValueError as error:
            QMessageBox.information(self, "Refine GCPs", str(error)); return
        wd = self.project.default_workdir(ch)
        cfg = {"model": self._model_dir(), "image_root": cm.image_dir(wd), "markers": [m for m in ch.markers if not m.get("disabled", False) and not m.get("locked", False)],
               "offset": (ch.georef or {}).get("offset"), "use_reference": bool(ch.georef),
               "search": 25, "output": os.path.join(wd, "refined_markers.json"),
               # model-frame positions computed here (CRS-aware: geographic references are
               # converted to metres before the offset is removed, which the task alone did not do)
               "positions": {l: np.asarray(x, float).tolist() for l, x, _e in (self._marker_positions() or [])}}
        path = os.path.join(wd, "refine_task.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, default=float)

        def loaded():
            with open(cfg["output"], encoding="utf-8") as result:
                res = json.load(result)
            ms = self._marker_objs(); n = 0
            for m in ms:
                if m.disabled or m.locked:
                    continue
                cur = {g._stem(a): a for a in m.projections}
                for stem, (u, v, _pin, _sc, how) in (res.get(m.label) or {}).items():
                    if stem in m.ignored_photos:
                        continue
                    if stem in cur and g.projection_is_pinned(m.projections[cur[stem]]):
                        continue                     # never move a user-pinned placement
                    if how in ("saddle", "template"):
                        m.projections.pop(cur.get(stem, stem), None)
                        m.projections[stem] = [u, v, False, "refined"]; n += 1
            if n:
                self._markers_snapshot()
                self._commit_markers(ms)
            self.log(f"Refine GCPs: {n} projections placed/refined (unpinned; pinned ones untouched)")
        self._run("Refine GCPs", [("Placing GCPs", [cm.PYTHON, "refine_markers", path], 100)], loaded)

    # ------------------------------------------------------------ point cloud filtering
    def filter_point_cloud(self, confidence_only=False):
        if not self._require_saved("Filter Point Cloud"):
            return None
        from .dialogs import FilterPointCloudDialog
        ch = self.dataset
        if not ch.point_cloud_ply or not os.path.exists(ch.point_cloud_ply):
            QMessageBox.information(self, "Filter Point Cloud", "Build or import a point cloud first."); return
        dlg = FilterPointCloudDialog(self, confidence_only)
        if not dlg.exec():
            return
        wd = self.project.default_workdir(ch)
        out = os.path.join(wd, "dense", "point_cloud_filtered.ply")
        if os.path.abspath(out) == os.path.abspath(ch.point_cloud_ply):
            out = os.path.join(wd, "dense", "point_cloud_filtered2.ply")
        cfg = dict(dlg.settings(), input=ch.point_cloud_ply, output=out,
                   model=self._model_dir() if os.path.isdir(self._model_dir()) else "")
        if ch.georef:
            cfg["up"] = [0.0, 0.0, 1.0]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        path = os.path.join(wd, "dense", "filter_ui.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        prev = dict((ch.meta or {}).get("point_cloud") or {})

        def meta(dt, stages):
            from . import procmeta
            e = procmeta.record(ch, "point_cloud", params=dict(prev.get("params") or {}, filter=cfg),
                                seconds=prev.get("seconds"), stages=prev.get("stages"), points=ch.point_count,
                                fusion_seconds=prev.get("fusion_seconds"), filtering_seconds=dt,
                                points_fused=prev.get("points"), filtering=[], **procmeta.file_info(out))
            e["filtered_from"] = cfg["input"]
        self._run("Filter Point Cloud", [("Filtering point cloud", [cm.PYTHON, "filter_cloud", path], 100)],
                  lambda: self.load_result("cloud", out), meta)

    def gradual_selection(self):
        if not self._require_saved("Gradual Selection"):
            return None
        from .dialogs import GradualSelectionDialog
        from . import colmap_io as cio
        from . import pcfilter as pf
        ch = self.dataset
        model = self._sparse_model() if os.path.isdir(self._model_dir()) else None
        if model is None:
            QMessageBox.information(self, "Gradual Selection", "Align photos first."); return
        wd = self.project.default_workdir(ch)
        scales = None
        db = os.path.join(wd, "database.db")
        if os.path.exists(db):
            try:
                scales = pf.keypoint_scales(db, model)
            except Exception as e:
                self.log(f"Gradual Selection: keypoint scales unavailable ({e})")
        metrics = pf.tie_point_metrics(model, scales)
        dlg = GradualSelectionDialog(metrics, self)
        if not dlg.exec():
            return
        sel = dlg.selection()
        n = cio.remove_points(model, sel)
        cio.write_model(self._model_dir(), model)
        from .plyio import save_ply
        save_ply(os.path.join(wd, "tie_points.ply"), model.points.xyz, model.points.rgb / 255.0)
        ch.tie_points_ply = os.path.join(wd, "tie_points.ply"); ch.tie_point_count = len(model.points)
        self._model_cache = (None,)
        self.log(f"Gradual Selection ({dlg.crit.currentText()} > {dlg.level.value():g}): "
                 f"removed {n:,} tie points, {len(model.points):,} remain. Run Optimize Cameras to re-adjust.")
        self.set_modified(); self.reload_view()

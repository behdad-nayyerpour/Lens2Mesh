import json
from pathlib import Path

import numpy as np
import pytest

from photomesh import colmap_io as cio, georef as g
from photomesh.alignment_reference import alignment_reference_config, control_availability
from photomesh.dialogs import AlignPhotosDialog
from photomesh.project import Camera
from test_dense_mesh_gcp import synthetic_model


def _data(window, tmp_path):
    model = synthetic_model(n_img=5, n_pts=100, two_cameras=False)
    ch = window.dataset
    ch.crs = "EPSG:32638"
    base = np.array([450000., 4420000., 100.])
    from PIL import Image
    for i, image in enumerate(model.images.values()):
        path = tmp_path / image.name
        Image.new("RGB", (600, 450)).save(path)
        # Make the camera flight non-collinear and preserve its 3-D tracks.
        center = image.center() + [0, (i % 2) * 5, 0]
        image.tvec = -image.R() @ center
        xyz = model.points.xyz[np.maximum(image.point3D_ids, 0)]
        uv, _ = cio.project(model.cameras[image.camera_id], image.R(), image.tvec, xyz)
        image.xys[image.point3D_ids >= 0] = uv[image.point3D_ids >= 0]
        e, n, h = center + base
        lat, lon = g.utm_inverse(e, n, 38, False)
        ch.cameras.append(Camera(str(path), lon=float(lon), lat=float(lat), alt=float(h), width=600, height=450))
    for i, point in enumerate(np.array([[3, 2, 0], [9, 2, 0], [5, 8, 0], [10, 9, 1.]])):
        observations = {}
        for image_id, image in model.images.items():
            uv = g.predict_projection(model, image_id, point)
            if uv is not None:
                observations[image.name] = [*uv, True] if i != 2 else [*uv, False, "refined"]
        ch.markers.append(g.Marker(f"GCP {i}", (point + base).tolist(), enabled=i < 3,
                                   projections=observations).to_dict())
    return ch, model


def test_dialog_reference_choices_are_independent_and_use_measurements(window, tmp_path):
    ch, _ = _data(window, tmp_path)
    dialog = AlignPhotosDialog(window)
    assert dialog.gps_frame.isEnabled() and dialog.gps_frame.isChecked()
    assert dialog.use_gcps.isEnabled() and not dialog.use_gcps.isChecked()
    dialog.use_gcps.setChecked(True)
    assert not dialog.common_heights.isHidden()
    with pytest.raises(ValueError, match="common height reference"):
        dialog.validate_settings()
    dialog.common_heights.setCurrentIndex(dialog.common_heights.findData("ellipsoidal"))
    settings = dialog.settings()
    assert settings["use_gcps"] and settings["gps_frame"]
    assert settings["common_height_reference"] == "ellipsoidal"
    dialog.validate_settings()
    dialog.gps_frame.setChecked(False)
    assert dialog.settings()["use_gcps"] and not dialog.settings()["gps_frame"]
    assert dialog.common_heights.isHidden()
    for observation in ch.markers[2]["projections"].values():
        observation[3] = "predicted"
    assert not control_availability(ch)[0]


@pytest.mark.parametrize("visual", [False, True])
def test_fast_alignment_prepares_only_missing_visual_descriptors(window, tmp_path, monkeypatch, visual):
    from photomesh import fastpipe, mainwindow

    paths = [tmp_path / name for name in ("a.jpg", "b.jpg")]
    for path in paths:
        path.write_bytes(b"photo")
    cached = tmp_path / "cached.npy"
    cached.touch()
    window.dataset.cameras = [
        Camera(str(paths[0]), desc=str(cached), width=123, height=100),
        Camera(str(paths[1]), width=123, height=100),
    ]
    window.project.path = str(tmp_path / "survey.pmsx")
    monkeypatch.setattr(window, "_require_saved", lambda _: True)
    monkeypatch.setattr(window, "_need_colmap", lambda: "engine")
    monkeypatch.setattr(AlignPhotosDialog, "exec", lambda _: True)
    monkeypatch.setattr(AlignPhotosDialog, "settings", lambda _:
                        {"mode": "Fast", "accuracy": "High", "generic": visual})
    monkeypatch.setattr(window, "cancel_thumbnails", lambda: None)
    analyzed = []
    runs = []

    def analyze(sources, cache, **kwargs):
        analyzed.extend(sources)
        return {path: {"desc": str(cached), "width": 300, "height": 200} for path in sources}

    monkeypatch.setattr(fastpipe, "analyze_images", analyze)
    monkeypatch.setattr(mainwindow, "run_modal", lambda parent, title, label, fn, total:
                        (fn(lambda *_: None, lambda: False), False, None))
    monkeypatch.setattr(window, "_run", lambda title, *args, **kwargs: runs.append(title))
    window.align_photos()
    assert runs == ["Align Photos"]
    assert analyzed == ([str(paths[1])] if visual else [])
    assert window.dataset.cameras[0].width == 123
    assert window.dataset.cameras[1].width == (300 if visual else 123)


@pytest.mark.parametrize("gps,gcps,local", [(False, False, False), (True, False, False), (False, True, False),
                                          (True, True, False), (True, False, True)])
def test_alignment_applies_only_selected_references_and_publishes_diagnostics(window, tmp_path, monkeypatch,
                                                                            gps, gcps, local):
    from photomesh import colmap, engine

    ch, model = _data(window, tmp_path)
    if local:
        ch.crs = "LOCAL"
        ch.georef = {"crs": "LOCAL", "offset": [26000, 22000, 1400]}
    previous_georef = dict(ch.georef)
    window.project.path = str(tmp_path / "survey.pmsx")
    old_workdir = ch.workdir
    monkeypatch.setattr(window, "_require_saved", lambda _: True)
    monkeypatch.setattr(window, "_need_colmap", lambda: "colmap")
    monkeypatch.setattr(AlignPhotosDialog, "exec", lambda _: True)
    settings = {"mode": "Exhaustive", "accuracy": "High", "gps_frame": gps, "use_gcps": gcps,
                "common_height_reference": "ellipsoidal"}
    monkeypatch.setattr(AlignPhotosDialog, "settings", lambda _: dict(settings))
    monkeypatch.setattr(engine, "execution_settings", lambda *_: {})
    monkeypatch.setattr(window, "reload_view", lambda: None)
    monkeypatch.setattr(window, "show_view", lambda _: None)
    monkeypatch.setattr(window, "_photo_map", lambda _: {})

    def plan(*args, **kwargs):
        workdir = Path(args[2])
        cio.write_model(str(workdir / "sparse/0"), model)
        return [("Exporting tie points", ["model_converter"], 1)]

    def run(_title, steps, loaded, _meta, **kwargs):
        planned = steps[0][1]()
        workdir, = Path(window.project.data_dir).glob("alignment-*")
        assert ch.workdir == old_workdir and ch.georef == previous_georef
        if gps or gcps:
            labels = [label for label, _, _ in planned]
            assert labels.index("Applying alignment references") < labels.index("Exporting tie points")
            cfg = json.loads((workdir / "alignment_reference.json").read_text(encoding="utf-8"))
            assert cfg["use_markers"] == gcps and cfg["use_cameras"] == gps
            if local:
                assert cfg["crs"] == "EPSG:4326" and cfg["offset"] is None
            if gps and gcps:
                assert cfg["bundle"] and cfg["camera_height_reference"] == "ellipsoidal"
            g.run_task(cfg)
        sparse = workdir / "sparse_txt"
        sparse.mkdir()
        exported = cio.read_model(str(workdir / "sparse/0"))
        lines = []
        for image_id, image in exported.images.items():
            lines.extend([" ".join(map(str, [image_id, *image.qvec, *image.tvec, image.camera_id, image.name])), ""])
        (sparse / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        loaded()

    monkeypatch.setattr(colmap, "align_steps", plan)
    monkeypatch.setattr(window, "_run", run)
    window.align_photos()
    assert ch.aligned_count() == 5
    if gcps:
        assert ch.georef["control_rmse"]["total"] < 0.01
        assert ch.georef["check_rmse"]["total"] < 0.01
        assert len(ch.georef["marker_image_errors"]) == 4
        assert all(point["est"] is not None for point in ch.markers)
    else:
        assert all(point["est"] is None for point in ch.markers)
    assert bool(ch.georef) == (gps or gcps)
    if local:
        assert g.CRS.parse(ch.crs).kind == "wgs84"
        assert ch.reference_settings["gcp_crs"] == "LOCAL"


def test_local_model_allows_both_selections_and_explains_local_gcp_conflict(window, tmp_path):
    ch, _ = _data(window, tmp_path)
    ch.crs = "LOCAL"
    ch.georef = {"crs": "LOCAL", "offset": [26000, 22000, 1400]}
    dialog = AlignPhotosDialog(window)
    assert dialog.gps_frame.isEnabled() and dialog.use_gcps.isEnabled()
    assert not dialog.gps_frame.isChecked()
    dialog.gps_frame.setChecked(True)
    assert "WGS 84" in dialog.reference_note.text()
    dialog.validate_settings()
    dialog.use_gcps.setChecked(True)
    assert dialog.settings()["gps_frame"] and dialog.settings()["use_gcps"]
    assert "known transformation" in dialog.reference_note.text()
    with pytest.raises(ValueError, match="known transformation"):
        dialog.validate_settings()
    dialog.gps_frame.setChecked(False)
    dialog.validate_settings()


def test_alignment_reference_failure_cannot_fall_back_to_gps(window, tmp_path):
    ch, model = _data(window, tmp_path)
    settings = {"gps_frame": True, "use_gcps": True, "common_height_reference": "ellipsoidal"}
    mapping = {camera.path: Path(camera.path).name for camera in ch.cameras}
    cfg = alignment_reference_config(ch, settings, mapping, str(tmp_path))
    # A selected GCP can have enough source measurements but fail to register.
    cfg["markers"][2]["projections"] = {"missing.jpg": [200, 200, True]}
    cio.write_model(cfg["model"], model)
    path = Path(cfg["model"]) / "images.bin"
    previous = path.read_bytes()
    with pytest.raises(ValueError, match="Only 2 GCPs"):
        g.run_task(cfg)
    assert path.read_bytes() == previous
    assert not Path(cfg["result"]).exists()


def test_change_gcp_system_updates_validation_diagnostics_and_processing(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QLabel, QPushButton
    from photomesh.alignment_reference import gcp_crs
    from photomesh.dialogs import GcpCoordinatesDialog, OptimizeCamerasDialog
    from photomesh.project import Project
    from test_markers_manual import _scene

    ch = _scene(window, tmp_path)
    ch.crs = "EPSG:4326"
    ch.georef = {"crs": ch.crs, "offset": [450000, 4420000, 100],
                 "rows": [{"label": "P", "total": 1.0}], "control_rmse": {"total": 1.0}}
    model = window._sparse_model()
    for index, location in enumerate(((102, 95, 0), (118, 105, 0), (130, 91, 3))):
        observations = {}
        for image_id, image in model.images.items():
            uv = g.predict_projection(model, image_id, location)
            if uv:
                observations[image.name] = [*uv, True]
        ref = [44 + index * .001, 40 + index % 2 * .002, 100.0]
        ch.markers.append(g.Marker(f"P{index}", ref, projections=observations).to_dict())
    original_pose = list(ch.cameras[0].est)
    window.refresh_all()

    def choose_local(dialog):
        dialog.system.setCurrentIndex(dialog.system.findData("LOCAL"))
        return True

    monkeypatch.setattr(GcpCoordinatesDialog, "exec", choose_local)
    panel = window.docks["Markers"].widget()
    change = next(button for button in panel.findChildren(QPushButton) if button.text() == "Change…")
    assert change.isEnabled()
    change.click()
    assert gcp_crs(ch) == "LOCAL"
    assert ch.crs == "EPSG:4326" and ch.cameras[0].est == original_pose
    assert ch.georef["rows"] == [] and len(ch.georef["marker_image_errors"]) == 3
    assert any("Local Coordinates" in label.text() for label in panel.findChildren(QLabel))
    assert window.undo_markers() and gcp_crs(ch) == "EPSG:4326"
    assert window.redo_markers() and gcp_crs(ch) == "LOCAL"
    assert window.edit_marker_cell("P0", 1, "26100")
    assert ch.markers[0]["ref"][0] == 26100

    dialog = AlignPhotosDialog(window)
    assert dialog.use_gcps.isEnabled()
    dialog.use_gcps.setChecked(True)
    assert g.CRS.parse(dialog.settings()["gcp_crs"]).kind == "local"
    assert not dialog.gps_frame.isEnabled()
    monkeypatch.setattr(OptimizeCamerasDialog, "exec", lambda _: True)
    calls = []
    monkeypatch.setattr(window, "_run", lambda *args: calls.append(args))
    window.optimize_cameras()
    assert len(calls) == 1
    cfg = json.loads((Path(ch.workdir) / "georef_task.json").read_text(encoding="utf-8"))
    assert g.CRS.parse(cfg["crs"]).kind == "local"
    assert cfg["offset"] is None and not cfg["camera_gps"]
    path = tmp_path / "gcp_crs.pmsx"
    window.project.save(str(path))
    restored = Project.load(str(path)).dataset
    assert restored.crs == "EPSG:4326" and gcp_crs(restored) == "LOCAL"


def test_gcp_epsg_field_accepts_metric_projected_systems_only(qtbot):
    from photomesh.dialogs import GcpCoordinatesDialog

    dialog = GcpCoordinatesDialog("LOCAL")
    qtbot.addWidget(dialog)
    dialog.system.setCurrentIndex(dialog.system.count() - 1)
    assert dialog.code.isEnabled()
    dialog.code.setText("32638")
    assert dialog.selected_crs() == "EPSG:32638"
    dialog.code.setText("EPSG:2263")
    with pytest.raises(ValueError):
        dialog.selected_crs()

"""Regression coverage for issue #1, project switches and Ortho marker placement."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from photomesh import colmap_io as cio
from photomesh import georef as g
from test_markers_manual import _scene, _marker
from test_dense_mesh_gcp import synthetic_model


def prepare_test_optimize_exports(stage):
    """Supply the engine export files in tests that run just the Python stage."""
    from photomesh.plyio import save_ply

    model = cio.read_model(str(stage / "sparse" / "0"))
    lines = []
    for iid, image in model.images.items():
        lines.extend((" ".join(map(str, (iid, *image.qvec, *image.tvec, image.camera_id, image.name))), ""))
    (stage / "sparse_txt" / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (stage / "sparse_txt" / "points3D.txt").write_text("# points\n", encoding="utf-8")
    save_ply(str(stage / "tie_points.ply"), model.points.xyz)


@pytest.mark.parametrize("transition", ["new", "open"])
def test_project_switch_drops_marker_overlays_history_and_pending(window, tmp_path, transition):
    from photomesh.project import ProjectData, Project
    _scene(window, tmp_path)
    label = window.add_marker_at_photo(500, 400, k=0)
    window.edit_marker_cell(label, 1, "123")
    assert window._mk_undo and window._ref_pending
    assert window.model_view.overlay.markers is not None
    # Queued cell edits must not leak into another project with the same marker label.
    window._marker_edited(label, 2, "999")
    target = ProjectData(markers=[g.Marker(label).to_dict()])
    if transition == "new":
        window.new_project()
    elif transition == "open":
        project = Project()
        project.datasets = [target]
        path = str(tmp_path / "next.pmsx")
        project.save(path)
        window.open_project(path)
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()
    window.wait_view_loaded()
    assert window.model_view.overlay.markers is None
    assert window.ortho_view.markers == []
    assert window.ortho_view.active_marker is None
    assert window._ref_pending == {} and window.markers_pane.pending == {}
    assert not window.undo_markers() and not window.redo_markers()
    assert all(m["ref"] is None for m in window.dataset.markers)


def test_model_clear_drops_marker_overlay_and_backing_data(window):
    window.model_view.set_markers(np.array([[1., 2., 3.]]), labels=["old"])
    window.model_view.clear()
    assert window.model_view.overlay.markers is None
    assert window.model_view._marker_data[0] is None


def test_ortho_never_paints_point_cloud_as_an_orthomosaic(window):
    from PySide6.QtGui import QImage, QPainter
    ov = window.ortho_view
    ov.center, ov.scale = (0., 0.), 5.
    def render():
        image = QImage(ov.size(), QImage.Format_ARGB32)
        painter = QPainter(image)
        try:
            ov._paint(painter)
        finally:
            painter.end()
        return image
    before = render()
    ov.set_data(np.array([[1., 2., 3.], [3., 4., 5.]]), [[5., 6., 7.]])
    assert ov._bounds() is None
    assert before == render()
    ov.center = None
    ov.fit()
    assert ov.center is None


def test_ortho_context_selects_marker_without_any_edit_actions(window,tmp_path,monkeypatch):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QAction, QContextMenuEvent, QImage, QPainter
    from photomesh import orthoview
    from photomesh.marker_style import HOVER_COLOR, color
    ch=_scene(window,tmp_path)
    original=g.Marker("G1",ref=[700.,800.,20.],enabled=False,
                      projections={"IMG_0001.JPG":[500.,400.,True]})
    ch.markers=[original.to_dict()]
    window.refresh_all()
    view=window.ortho_view
    from PySide6.QtWidgets import QTabBar
    window.docks["Markers"].widget().findChild(QTabBar).setCurrentIndex(1)
    assert window.markers_pane.isRowHidden(0)
    view.center, view.scale = (110., 100.), 2.
    view.set_markers([("G1", 110., 100.)], {"G1": 5.})
    event_pos = QPoint(*map(round, view.to_screen(110., 100.)))
    monkeypatch.setattr(view,"coordinates_at",lambda *a:(115.,105.,9.))
    before=copy.deepcopy(ch.markers); history=len(window._mk_undo)
    actions=[]
    painted_flags, painted_labels = [], []
    draw_flag = orthoview.draw_flag

    def capture_flag(painter, x, y, colour, active):
        painted_flags.append(colour)
        draw_flag(painter, x, y, colour, active)

    monkeypatch.setattr(orthoview, "draw_flag", capture_flag)

    def render():
        image = QImage(view.size(), QImage.Format_ARGB32)
        class CapturePainter(QPainter):
            def drawText(self, *args):
                if args[-1] == "G1":
                    painted_labels.append(self.pen().color())
                return super().drawText(*args)
        painter = CapturePainter(image)
        try:
            view._paint(painter)
        finally:
            painter.end()

    class TestMenu:
        def __init__(self,parent): self.parent,self.items=parent,[]
        def addAction(self,text):
            action=QAction(text,self.parent); self.items.append(action); return action
        def addSeparator(self): pass
        def exec(self,*args):
            actions.extend(a.text() for a in self.items)
            if len(self.items) == 3:
                view.leaveEvent(None)
                render()
            self.items[0].trigger()
            if len(self.items) == 3:
                self.items[1].trigger()
    monkeypatch.setattr(orthoview,"QMenu",TestMenu)
    view.contextMenuEvent(QContextMenuEvent(QContextMenuEvent.Mouse, event_pos, event_pos))
    assert actions == ["Show GCP on table", "Copy GCP coordinate", "Reset view"]
    from PySide6.QtGui import QGuiApplication
    assert QGuiApplication.clipboard().text() == "110.000\t100.000\t5.000"
    assert not window.markers_pane.isRowHidden(0)
    assert painted_flags == painted_labels == [HOVER_COLOR]
    assert view._hovered_flag is None
    render()
    assert painted_flags == painted_labels == [HOVER_COLOR, color("predicted")]
    actions.clear()
    elsewhere = event_pos + QPoint(40, 40)
    view.contextMenuEvent(QContextMenuEvent(QContextMenuEvent.Mouse, elsewhere, elsewhere))
    assert actions == ["Copy coordinate", "Reset view"]
    assert QGuiApplication.clipboard().text() == "115.000\t105.000\t9.000"
    assert window.markers_pane.current_label() == "G1"
    assert ch.markers == before and len(window._mk_undo) == history


def test_imported_xyz_alone_is_not_a_usable_control(window, tmp_path):
    ch = _scene(window, tmp_path)
    markers = [g.Marker(f"G{i}", [i, i * i, 10.]) for i in range(94)]
    status = g.reference_status(window._sparse_model(), markers, g.CRS.parse(ch.crs))
    assert status == {"imported": 94, "references": 94, "usable": 0, "gps": 0}
    # Two differently spelled keys for one photo still count as only one observation.
    X, n = g.triangulate_marker(window._sparse_model(), {"IMG_0001.JPG": [500, 400],
                                                      "img_0001": [500, 400]})
    assert X is None and n == 1


@pytest.mark.parametrize("accept", [False, True])
def test_optimize_preflight_offers_local_mode_explicitly(window, tmp_path, monkeypatch, accept):
    from PySide6.QtWidgets import QMessageBox
    from photomesh import dialogs
    ch = _scene(window, tmp_path)
    ch.markers = [g.Marker(f"G{i}", [i, i * i, 10.]).to_dict() for i in range(94)]
    monkeypatch.setattr(dialogs.OptimizeCamerasDialog, "exec", lambda self: 1)
    prompts, ran = [], []
    def question(*a, **kw):
        prompts.append(a[2])
        return QMessageBox.Yes if accept else QMessageBox.No
    monkeypatch.setattr(QMessageBox, "question", question)
    monkeypatch.setattr(window, "_run", lambda *a, **k: ran.append((a, k)))
    window.optimize_cameras()
    assert "94 GCPs imported" in prompts[0]
    assert "0 usable controls" in prompts[0]
    assert bool(ran) is accept
    assert not ch.georef
    if accept:
        path = os.path.join(ch.workdir, "georef_task.json")
        cfg = json.load(open(path))
        assert cfg["local_only"] is True
        assert "local bundle adjustment" in ran[0][0][1][0][0]


@pytest.mark.parametrize("control_count", [2, 3])
def test_optimize_transfers_gps_alignment_to_local_survey_grid(window, tmp_path, monkeypatch, control_count):
    from PySide6.QtWidgets import QMessageBox
    from photomesh.dialogs import OptimizeCamerasDialog

    ch = _scene(window, tmp_path)
    ch.crs = "EPSG:4326"
    old_offset = [566200.0, 4280200.0, 1500.0]
    ch.georef = {"crs": ch.crs, "crs_kind": "wgs84", "offset": old_offset}
    old_reference_settings = {"use_cameras": True}
    ch.reference_settings = old_reference_settings.copy()
    model = window._sparse_model()
    translation = np.array([26000, 22000, 1400])
    for index, xyz in enumerate(((102, 95, 0), (118, 105, 0), (130, 91, 3))[:control_count]):
        projections = {}
        for image_id, image in model.images.items():
            pixel = g.predict_projection(model, image_id, xyz)
            if pixel is not None:
                projections[image.name] = [*pixel, True]
        assert len(projections) >= 2
        marker = g.Marker(f"P{index}", (np.asarray(xyz) + translation).tolist(), projections=projections)
        ch.markers.append(marker.to_dict())
    original_markers = copy.deepcopy(ch.markers)

    def choose_local(dialog):
        dialog.use_cameras.setChecked(True)
        dialog.adjustment_frame.setCurrentIndex(1)
        dialog.control_coordinates.setCurrentIndex(dialog.control_coordinates.findData("local"))
        assert not dialog.use_cameras.isEnabled()
        assert not dialog.adjustment_frame.isEnabled()
        dialog.bundle.setChecked(False)
        return 1

    monkeypatch.setattr(OptimizeCamerasDialog, "exec", choose_local)
    messages, submitted = [], []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: messages.append(args[2]))
    monkeypatch.setattr(window, "_run", lambda *args, **kwargs: submitted.append(args))
    window.optimize_cameras()
    assert ch.crs == "EPSG:4326" and ch.georef["offset"] == old_offset
    assert ch.markers == original_markers and ch.reference_settings == old_reference_settings
    if control_count == 2:
        assert not submitted and "2 usable control points" in messages[0]
        return

    assert len(submitted) == 1 and not messages
    cfg = json.loads((tmp_path / "work/georef_task.json").read_text())
    assert cfg["offset"] is None and cfg["coordinate_frame"] is None
    assert cfg["adjustment_frame"] == "projected"
    assert not cfg["use_cameras"] and not cfg["camera_gps"]
    result = g.run_task(cfg, log=lambda *_: None)
    assert result["crs_kind"] == "local"
    assert result["control_rmse"]["total"] < 1e-5
    assert result["offset"] == [26100.0, 22100.0, 1400.0]
    prepare_test_optimize_exports(Path(cfg["stage"]))
    monkeypatch.setattr(window, "_reload_alignment", lambda *_: None)
    submitted[0][2]()
    assert g.CRS.parse(ch.crs).kind == "local" and ch.georef["offset"] == result["offset"]
    assert [(m["ref"], m["projections"]) for m in ch.markers] == [
        (m["ref"], m["projections"]) for m in original_markers]


def test_local_optimization_reduces_error_and_retains_frame(tmp_path):
    model = synthetic_model(n_img=5, n_pts=250)
    rng = np.random.default_rng(94)
    for im in model.images.values():
        im.tvec = -im.R() @ (im.center() + rng.normal(0, .2, 3))
    g.retriangulate(model)
    ids = sorted(model.images)
    first = model.images[ids[0]]
    C0, R0 = first.center().copy(), first.R().copy()
    other = max(ids[1:], key=lambda i: np.linalg.norm(model.images[i].center() - C0))
    baseline = np.linalg.norm(model.images[other].center() - C0)
    err0 = np.mean(cio.observation_errors(model)[0] ** 2)
    assert err0 > .5  # enough pose error to distinguish adjustment from the 0.3 px noise floor
    path = str(tmp_path / "sparse")
    cio.write_model(path, model)
    cloud = tmp_path / "unchanged.ply"
    cloud.write_bytes(b"dense data must not be transformed in local mode")
    out = g.run_task({"model": path, "local_only": True, "fit_intrinsics": False,
                      "max_iterations": 8, "transform_files": [str(cloud)],
                      "markers": [g.Marker(f"G{i}", [i, i * i, 10.]).to_dict() for i in range(94)]},
                     log=lambda *_: None)
    result = cio.read_model(path)
    assert out["local_only"] and out["similarity"] is None
    assert np.allclose(result.images[ids[0]].center(), C0, atol=1e-8)
    assert np.allclose(result.images[ids[0]].R(), R0, atol=1e-8)
    assert np.linalg.norm(result.images[other].center() - C0) == pytest.approx(baseline, abs=1e-8)
    assert np.mean(cio.observation_errors(result)[0] ** 2) < err0 * .5
    assert cloud.read_bytes() == b"dense data must not be transformed in local mode"
    assert all(m["est"] is None for m in out["markers"].values())
    assert not os.path.exists(path + "_optimize_tmp")


def test_camera_exif_gps_can_be_used_in_utm(window, tmp_path):
    _scene(window, tmp_path)
    gps = {"IMG_0001.JPG": [69.763, 38.678, 1580.7]}
    got = g.camera_gps_metric(window._sparse_model(), gps, g.CRS.parse("EPSG:32642"))
    E, N = g.utm_forward(38.678, 69.763, 42)
    assert np.allclose(got[1], [E, N, 1580.7])
    assert g.camera_gps_metric(window._sparse_model(), gps, g.CRS.parse("")) == {}

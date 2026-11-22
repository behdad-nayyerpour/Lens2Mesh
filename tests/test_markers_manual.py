"""Manual ground control: Add GCP on a photo / on the 3D surface, prediction
into the other photos, editable Markers table with validation + undo, Rename /
Remove, Filter Photos by GCPs, Ctrl+click pinning, Refine/Optimize inputs."""
import json
import os

import numpy as np
import pytest

from photomesh import colmap_io as cio
from photomesh import georef as g

R_DOWN = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])      # looking down, image y = -Y


def _ground_ply(path, z=0.0, lo=-100.0, hi=300.0):
    from plyfile import PlyData, PlyElement
    v = np.zeros(4, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    v["x"] = [lo, hi, hi, lo]; v["y"] = [lo, lo, hi, hi]; v["z"] = z
    f = np.empty(2, dtype=[("vertex_indices", "i4", (3,))]); f["vertex_indices"] = [[0, 1, 2], [0, 2, 3]]
    PlyData([PlyElement.describe(v, "vertex"), PlyElement.describe(f, "face")]).write(path)
    return path


def _scene(window, tmp_path, centres=((100, 100, 60), (120, 100, 60), (140, 100, 60))):
    """Three nadir cameras 60 m above a flat ground (z = 0), 20 m apart in X."""
    window.project.path = str(tmp_path / "proj.pmsx")      # processing needs a saved project
    from PIL import Image
    from photomesh.project import Camera
    ch = window.dataset
    wd = tmp_path / "work"; (wd / "sparse" / "0").mkdir(parents=True)
    ch.workdir = str(wd)
    cam = cio.Camera(1, 1000, 800, np.array([1000.0, 1000.0, 500.0, 400.0]))
    imgs, cams = {}, []
    for k, C in enumerate(centres, 1):
        name = f"IMG_{k:04d}.JPG"
        photo = str(tmp_path / name)
        Image.new("RGB", (1000, 800), (90, 110, 90)).save(photo)
        C = np.asarray(C, float)
        imgs[k] = cio.Image(name, 1, cio.rot_to_qvec(R_DOWN), -R_DOWN @ C, np.zeros((0, 2)), np.zeros(0, np.int64))
        cams.append(Camera(path=photo, label=name, aligned=True, est=C.tolist(), rot=R_DOWN.tolist(),
                           width=1000, height=800))
    pts = cio.Points(np.array([1, 2], np.uint64), np.array([[100.0, 100, 0], [130.0, 95, 0]]),
                     np.zeros((2, 3), np.uint8), np.zeros(2), np.array([0, 0, 0]), np.zeros(0, np.uint32),
                     np.zeros(0, np.uint32))
    cio.write_model(str(wd / "sparse" / "0"), cio.Model({1: cam}, imgs, pts))
    ch.cameras = cams
    ch.tie_point_count = 2
    window.refresh_all(); window.reload_view(); window.wait_view_loaded()
    window._headless = True
    return ch


def _marker(window, label):
    return next(m for m in window.dataset.markers if m["label"] == label)


def test_ground_control_panel_placements_and_photo_buttons(window, tmp_path, monkeypatch):
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QLabel, QMenu, QPushButton
    from photomesh import panes

    ch = _scene(window, tmp_path)
    ch.markers = [g.Marker("P12", [100, 100, 0], projections={
        "img_0001": [500, 400, True],
        "img_0002": [510, 400, False, "refined"],
        "other": [520, 400, False, "predicted"],
    }).to_dict()]
    ch.georef = {"rows": [{"label": "P12", "pix": 0.5}]}
    pane = window.markers_pane
    pane.refresh(ch)
    header = pane.horizontalHeader()
    assert header.visualIndex(6) == header.visualIndex(7) + 1
    assert header.visualIndex(10) == header.visualIndex(6) + 1
    assert pane.columnWidth(1) >= pane.MIN_WIDTHS[1]
    assert pane.item(0, 9).text() == "Measured"
    assert pane.item(0, 10).text() == "1"

    pane.selectRow(0)
    panel = window.docks["Markers"].widget()
    assert any("Auto placements: 1 · Predictions: 0" in label.text() for label in panel.findChildren(QLabel))
    buttons = {button.text(): button for button in panel.findChildren(QPushButton)}
    buttons["Open photos"].click()
    assert window.photos.visible_indices() == [0, 1]
    buttons["All photos"].click()
    assert window.photos.visible_indices() == [0, 1, 2]

    menu_actions = []

    class InspectableMenu(QMenu):
        def exec(self, _pos):
            menu_actions.extend(action.text() for action in self.actions())

    monkeypatch.setattr(panes, "QMenu", InspectableMenu)
    pane._menu(QPoint(0, 0))
    assert menu_actions == ["Add GCP", "Rename GCP", "Remove GCP", "Ignore GCP (in all photos)",
                            "Open photos", "All photos"]
    from PySide6.QtCore import QSize
    icon = pane.item(0, 0).icon()
    for scale in (1., 1.25, 1.5, 2., 3.):
        pixmap = icon.pixmap(QSize(16, 16), scale)
        diameter = round(16 * scale)
        assert pixmap.devicePixelRatio() == scale
        assert pixmap.width() == pixmap.height() == diameter
        image = pixmap.toImage()
        assert image.pixelColor(diameter // 2, diameter // 2).name() == "#087a48"
        assert 0 < image.pixelColor(diameter // 2, round(1.5 * scale)).alpha()
        assert image.pixelColor(0, 0).alpha() == 0
    assert pane.item(0, 9).foreground().style() == panes.Qt.NoBrush
    assert {"Import GCPs…", "Add GCP", "Change…"} <= set(buttons)
    assert not any(label.text() == "Control coordinates" for label in panel.findChildren(QLabel))
    assert buttons["Clear auto placements"].isEnabled()
    buttons["Clear auto placements"].click()
    assert set(_marker(window, "P12")["projections"]) == {"img_0001", "other"}
    assert pane.item(0, 10).text() == "0"
    assert not buttons["Clear auto placements"].isEnabled()
    assert window.undo_markers()
    assert _marker(window, "P12")["projections"]["img_0002"][3] == "refined"


@pytest.mark.parametrize("offset", [None, [1000., 2000., 50.]])
def test_add_marker_on_photo_pins_and_predicts_other_photos(window, tmp_path, offset):
    from photomesh.project import Project

    ch = _scene(window, tmp_path)
    ch.model_ply = _ground_ply(str(tmp_path / "ground.ply"))
    if offset is not None:
        ch.georef = {"offset": offset, "crs": "LOCAL"}
        ch.reference_settings["gcp_crs"] = "EPSG:4326"
    window.markers_pane.clearSelection()
    assert window.markers_pane.current_label() is None          # no active marker needed
    lab = window.add_marker_at_photo(500.0, 400.0, k=0)
    assert lab == "point 1"
    m = _marker(window, lab)
    expected = np.array([100., 100., 0.]) + (offset if offset is not None else np.zeros(3))
    table = window.markers_pane
    assert [table.item(0, col).text() for col in (1, 2, 3)] == [f"{value:.4f}" for value in expected]
    assert table.item(0, 9).text() == "Model estimate"
    assert np.allclose(m["photo_model_xyz"], expected)
    assert m["ref"] is None
    saved_path = str(tmp_path / "estimated-gcp.pmsx")
    window.project.save(saved_path)
    assert Project.load(saved_path).dataset.markers[0] == m
    with open(saved_path, encoding="utf-8") as file:
        assert "photo_model_xyz" not in json.load(file)["datasets"][0]["markers"][0]
    assert window.undo_markers() and not window.dataset.markers
    assert window.redo_markers() and window.dataset.markers[0] == m
    p = m["projections"]
    assert p["img_0001"] == [500.0, 400.0, True]                 # pinned where clicked
    assert p["img_0002"][2] is False                             # predicted, stored unpinned
    assert p["img_0002"][0] == pytest.approx(1000 * -20 / 60 + 500, abs=1e-3)
    assert p["img_0002"][1] == pytest.approx(400.0, abs=1e-3)
    assert "img_0003" not in p                                   # -40 m at 60 m: outside the frame
    assert window.project.modified
    X, n = g.triangulate_marker(window._sparse_model(), p)       # provisional display position; prediction is not a measurement
    assert n == 2 and np.allclose(X, [100, 100, 0], atol=1e-3)
    labels = window.model_view.overlay.markers[1]
    assert lab in labels                                          # shown in the 3D view


def test_add_marker_falls_back_to_tie_points_and_reports_no_surface(window, tmp_path):
    _scene(window, tmp_path)                                      # no mesh, no DEM
    lab = window.add_marker_at_photo(500.0, 400.0, k=0)            # ray hits tie point (100, 100, 0)
    assert "img_0002" in _marker(window, lab)["projections"]
    assert [window.markers_pane.item(0, col).text() for col in (1, 2, 3)] == ["", "", ""]
    assert window.add_marker_at_ray(np.array([500.0, 500, 50]), np.array([0.0, 0, -1])) is None
    assert len(window.dataset.markers) == 1



def test_add_marker_uses_current_elevation_preview_for_model_estimate(window, tmp_path):
    ch = _scene(window, tmp_path)
    offset = [1000., 2000., 50.]
    ch.georef = {"offset": offset, "crs": "LOCAL"}
    ortho = window.ortho_view
    ortho.zgrid = (np.full((6, 6), 55.), 1097., 2103., 1., None)
    ortho._height_range = (55., 55.)
    ortho.height_surface = {"georef": True, "offset": offset}

    label = window.add_marker_at_photo(500., 400., k=0)
    point = _marker(window, label)
    assert point["photo_model_xyz"] == pytest.approx([1100., 2100., 55.])
    assert point["ref"] is None

    ortho.height_surface["stale"] = True
    other = _marker(window, window.add_marker_at_photo(500., 400., k=0))
    assert "photo_model_xyz" not in other
    assert other["projections"]["img_0001"] == [500., 400., True]


def test_model_view_cannot_add_markers_and_ray_math_stays_correct(window, tmp_path):
    ch = _scene(window, tmp_path)
    ch.model_ply = _ground_ply(str(tmp_path / "ground.ply"))
    lab = window.add_marker_at_ray(np.array([110.0, 100, 50]), np.array([0.0, 0, -1]))
    assert lab is None and window.dataset.markers == []
    # model-view picking: the pyqtgraph MVP ray goes through the projected pixel
    from photomesh.overlay3d import pixel_ray, project
    P = np.array([[1.2, 0, 0, 0], [0, 1.6, 0, 0], [0, 0, -1.0002, -0.20002], [0, 0, -1, 0]])
    V = np.eye(4); V[:3, 3] = [-110, -100, -150]
    mvp = P @ V
    xy, _z, ok = project(mvp, [[113.0, 97.0, 2.0]], 640, 480)
    o, d = pixel_ray(mvp, xy[0, 0], xy[0, 1], 640, 480)
    w = np.array([113.0, 97.0, 2.0]) - o
    assert ok[0] and np.linalg.norm(w - (w @ d) * d) < 1e-6


def test_prediction_skips_photos_where_the_surface_hides_the_marker(window, tmp_path):
    pytest.importorskip("open3d")
    from plyfile import PlyData, PlyElement
    ch = _scene(window, tmp_path)
    # ground + a 10 m high wall at x = 126 between the marker (x = 128) and camera 1 (x = 100)
    v = np.zeros(8, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    v["x"] = [-100, 300, 300, -100, 126, 126, 126, 126]
    v["y"] = [-100, -100, 300, 300, 0, 200, 200, 0]
    v["z"] = [0, 0, 0, 0, 0, 0, 10, 10]
    f = np.empty(4, dtype=[("vertex_indices", "i4", (3,))])
    f["vertex_indices"] = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]
    ch.model_ply = str(tmp_path / "wall.ply")
    PlyData([PlyElement.describe(v, "vertex"), PlyElement.describe(f, "face")]).write(ch.model_ply)
    assert window._raycast_scene() is not None
    pred = window.predicted_projections(np.array([128.0, 100.0, 0.0]))
    assert "img_0002" in pred and "img_0003" in pred
    assert "img_0001" not in pred                                 # behind the wall from camera 1


def test_markers_table_edits_validate_and_undo(window, tmp_path):
    _scene(window, tmp_path)
    lab = window.add_marker_at_photo(500.0, 400.0, k=0)
    assert window.edit_marker_cell(lab, 1, "abc") is False        # rejected, nothing stored
    assert _marker(window, lab)["ref"] is None
    assert window.edit_marker_cell(lab, 1, "12,5")                # decimal comma accepted
    assert _marker(window, lab)["ref"] is None                     # incomplete: pending only
    assert window._ref_pending[lab] == [12.5, None, None]
    assert window.markers_pane.item(0, 1).text() == "12.5000"
    window.edit_marker_cell(lab, 2, "20"); window.edit_marker_cell(lab, 3, "-3.25")
    assert _marker(window, lab)["ref"] == [12.5, 20.0, -3.25] and lab not in window._ref_pending
    assert window.edit_marker_cell(lab, 4, "0.02/0.05") and _marker(window, lab)["acc"] == [0.02, 0.05]
    assert window.edit_marker_cell(lab, 4, "-1") is False and _marker(window, lab)["acc"] == [0.02, 0.05]
    assert window.edit_marker_cell(lab, 4, "") and _marker(window, lab)["acc"] is None
    window.add_marker()                                           # Tools > Markers > Add GCP -> point 2
    assert window.edit_marker_cell(lab, 0, "point 2") is False    # duplicate label
    assert window.edit_marker_cell(lab, 0, "  ") is False         # empty label
    assert window.edit_marker_cell(lab, 0, "GCP 7")
    assert [m["label"] for m in window.dataset.markers] == ["GCP 7", "point 2"]
    assert window.undo_markers() and [m["label"] for m in window.dataset.markers] == [lab, "point 2"]
    assert window.undo_markers() and len(window.dataset.markers) == 1        # Add GCP undone
    assert window.redo_markers() and len(window.dataset.markers) == 2
    window.actions["Edit/Undo"].trigger()                          # Edit > Undo is wired
    assert len(window.dataset.markers) == 1


def test_table_double_click_edit_goes_through_signal(window, tmp_path, qtbot):
    _scene(window, tmp_path)
    lab = window.add_marker_at_photo(500.0, 400.0, k=0)
    pane = window.markers_pane
    from photomesh.panes import MarkersPane
    from PySide6.QtWidgets import QAbstractItemView
    assert pane.editTriggers() & QAbstractItemView.DoubleClicked
    from PySide6.QtCore import Qt
    assert all(pane.item(0, c).flags() & Qt.ItemIsEditable for c in MarkersPane.EDITABLE)
    assert not pane.item(0, 5).flags() & Qt.ItemIsEditable
    pane.item(0, 3).setText("101.5")                               # what the cell editor commits
    qtbot.waitUntil(lambda: window._ref_pending.get(lab) == [None, None, 101.5], timeout=2000)
    pane.item(0, 0).setText("renamed")
    qtbot.waitUntil(lambda: window.dataset.markers[0]["label"] == "renamed", timeout=2000)
    assert window._ref_pending.get("renamed") == [None, None, 101.5]


def test_rename_remove_filter_and_workspace_menu(window, tmp_path):
    ch = _scene(window, tmp_path)
    ch.model_ply = _ground_ply(str(tmp_path / "ground.ply"))
    a = window.add_marker_at_photo(500.0, 400.0, k=0)              # on photos 1, 2
    b = window.add_marker_at_photo(500.0, 400.0, k=2)              # on photos 3, 2
    assert window.rename_marker(a, "G1") and window.rename_marker(b, "G2")
    shown = window.filter_photos_by_markers(["G1"])
    assert shown == {0, 1} and window.photos.visible_indices() == [0, 1]
    window._workspace_action("marker:Filter Photos by GCPs:G2", window.project.active)
    assert window.photos.visible_indices() == [1, 2]
    window._workspace_action("marker:Reset Filter:", window.project.active)
    assert window.photos.visible_indices() == [0, 1, 2]
    window._workspace_action("marker:Remove GCP:G1", window.project.active)
    assert [m["label"] for m in window.dataset.markers] == ["G2"]
    assert window.undo_markers() and [m["label"] for m in window.dataset.markers] == ["G1", "G2"]
    assert window.remove_markers(["G1", "G2"], confirm=False) == 2 and window.dataset.markers == []
    rows = []
    window.workspace.refresh(window.project)
    assert not any(it.text(0).startswith("GCPs (") for it in window.workspace._iter())


def test_photo_flags_kinds_ctrl_click_and_context(window, tmp_path):
    from PySide6.QtCore import QPointF
    _scene(window, tmp_path)
    window.open_photo(1)
    pv = window.photo_view
    got = []
    pv.addMarker.connect(lambda u, v: got.append(("add", u, v)))
    pv.pinMarker.connect(lambda l, u, v: got.append(("pin", l, u, v)))
    pv.active_marker = None; pv.ctrl_click(10, 20)
    pv.active_marker = "G9"; pv.ctrl_click(30, 40)
    assert got == [("add", 10.0, 20.0), ("pin", "G9", 30.0, 40.0)]
    pv.set_markers([("A", 100, 100, "pinned"), ("B", 200, 100, "placed"), ("C", 300, 100, "predicted")])
    flags = [item for item in pv._overlay if hasattr(item, "colour") and item.zValue() == 7]
    assert [item.colour.name() for item in flags] == ["#28dc3c", "#3c8cff", "#bebebe"]
    assert all(item.pos().y() == 100 for item in flags)  # exact ground position is the item origin
    assert pv._flag_at(pv.view.mapFromScene(QPointF(105, 100))) == "A"
    # pinning a blue flag through the window keeps one projection per photo and pins it
    lab = window.add_marker_at_photo(500.0, 400.0, k=0)
    assert _marker(window, lab)["projections"]["img_0002"][2] is False
    window.pin_marker(lab, 170.0, 401.0, k=1)
    assert _marker(window, lab)["projections"]["img_0002"] == [170.0, 401.0, True]


def test_refine_and_optimize_inputs_include_new_markers(window, tmp_path, monkeypatch):
    ch = _scene(window, tmp_path)
    lab = window.add_marker_at_photo(500.0, 400.0, k=0)
    window.edit_marker_cell(lab, 1, "100"); window.edit_marker_cell(lab, 2, "100"); window.edit_marker_cell(lab, 3, "0")
    ran = []
    monkeypatch.setattr(window, "_run", lambda name, steps, *a, **k: ran.append((name, steps)))
    window.refine_markers()
    assert ran and ran[0][0] == "Refine GCPs"
    cfg = json.load(open(os.path.join(window.project.default_workdir(ch), "refine_task.json")))
    assert any(m["label"] == lab and len(m["projections"]) == 2 for m in cfg["markers"])
    assert np.allclose(cfg["positions"][lab], [100, 100, 0], atol=1e-3)
    # the refine task uses the GUI position (CRS-aware) before the raw reference
    from photomesh import tasks
    called = {}
    def fake_predict(model, iid, X):
        called["X"] = np.asarray(X, float)
        return None
    monkeypatch.setattr(g, "predict_projection", fake_predict)
    monkeypatch.setattr("photomesh.marker_refine.crop_views", lambda *a, **k: {})
    cfg["output"] = str(tmp_path / "out.json")
    tasks.cmd_refine_markers(cfg)
    assert np.allclose(called["X"], [100, 100, 0])
    # Optimize Cameras sees it as a control point with two projections
    m = g.Marker(**_marker(window, lab))
    assert m.enabled and m.ref == [100.0, 100.0, 0.0] and len(m.projections) == 2


def test_parse_helpers():
    from photomesh.markers_ui import parse_coordinate, parse_accuracy, ray_mesh
    assert parse_coordinate(" 1 234.5 ") == 1234.5 and parse_coordinate("") is None
    with pytest.raises(ValueError):
        parse_coordinate("nan")
    assert parse_accuracy("0.01") == [0.01, 0.01] and parse_accuracy("0.01/0.02") == [0.01, 0.02]
    with pytest.raises(ValueError):
        parse_accuracy("0/1")
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    assert ray_mesh(V, np.array([[0, 1, 2]]), [0.2, 0.2, 5], [0, 0, -1]) == pytest.approx(5.0)
    assert ray_mesh(V, np.array([[0, 1, 2]]), [2, 2, 5], [0, 0, -1]) is None


def test_dynamic_prediction_filter_and_scoped_ignore_survive_reopen(window, tmp_path, monkeypatch):
    from photomesh import tasks
    from photomesh.project import Project

    ch = _scene(window, tmp_path)
    model = window._sparse_model()
    xyz = [120., 100., 0.]
    pixels = {i: g.predict_projection(model, i, xyz) for i in model.images}
    marker = g.Marker("G1", xyz, projections={
        "IMG_0001.JPG": [*pixels[1], True], "IMG_0003.JPG": [*pixels[3], True]})
    ch.crs = "LOCAL"
    ch.georef = {"offset": [0., 0., 0.], "crs": "LOCAL"}
    window._commit_markers([marker])
    window.open_photo(1)
    assert window.marker_prediction_count("G1") == 1
    assert window.photo_view.markers[0][-1] == "predicted"
    assert window.filter_photos_by_markers(["G1"]) == {0, 1, 2}
    assert window.pin_marker("G1", *pixels[2])
    assert window.photo_view.markers[0][-1] == "pinned"

    assert window.marker_view_action("ignore_photo", "G1")
    assert not window.marker_view_action("ignore_photo", "G1")
    saved = _marker(window, "G1")
    assert saved["ignored_photos"] == ["img_0002"]
    assert set(map(g._stem, saved["projections"])) == {"img_0001", "img_0003"}
    assert window.photo_view.markers[0][-1] == "ignored"
    assert window.marker_prediction_count("G1") == 0
    assert window.photos.visible_indices() == [0, 1, 2]
    assert window.dataset.georef["marker_image_errors"]["G1"]["n"] == 2

    project_path = str(tmp_path / "placement.pmsx")
    window.project.save(project_path)
    with open(project_path, encoding="utf-8") as project_file:
        dataset_record = json.load(project_file)["datasets"][0]
    assert "ignored_photos" not in dataset_record["markers"][0]
    assert dataset_record["gcp_ignored_photos"] == {"G1": ["img_0002"]}
    reloaded = Project.load(project_path).dataset.markers[0]
    assert reloaded == saved
    xml_path = tmp_path / "placement.xml"
    g.write_markers_xml(xml_path, [g.Marker(**reloaded)], "LOCAL")
    imported, _ = g.read_markers_xml(xml_path)
    assert imported[0].ignored_photos == ["img_0002"]
    assert g.merge_markers(imported, [g.Marker("G1", projections={
        "img_0002": [*pixels[2], True]})])[0].projections == saved["projections"]

    planned = {}
    def capture_plan(_model, _root, plan, **_options):
        planned.update(plan)
        return {}
    monkeypatch.setattr("photomesh.marker_refine.crop_views", capture_plan)
    monkeypatch.setattr("photomesh.marker_refine.auto_place", lambda *args, **kwargs: {})
    tasks.cmd_refine_markers({"model": window._model_dir(), "image_root": str(tmp_path),
                              "markers": [saved], "positions": {"G1": xyz},
                              "output": str(tmp_path / "refined.json")})
    assert planned and 2 not in planned
    assert window.undo_markers()
    assert window.photo_view.markers[0][-1] == "pinned"
    assert window.redo_markers()
    assert window.photo_view.markers[0][-1] == "ignored"

    assert window.marker_view_action("ignore", "G1")
    assert window.dataset.markers[0]["disabled"] and not window.dataset.markers[0]["projections"]
    assert window.dataset.markers[0]["ref"] == xyz
    assert not window.dataset.georef["marker_image_errors"]
    assert window.pin_marker("G1", *pixels[2])
    assert not window.dataset.markers[0]["disabled"]
    assert window.dataset.markers[0]["projections"] == {"img_0002": [*pixels[2], True]}


def test_photo_context_hover_and_navigation(window, tmp_path, monkeypatch, qtbot):
    from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
    from PySide6.QtWidgets import QApplication, QGraphicsSimpleTextItem, QGraphicsView, QMenu
    from photomesh import views
    from photomesh.marker_style import HOVER_COLOR, color

    _scene(window, tmp_path)
    window.open_photo(0)
    view = window.photo_view
    canvas = view.view.viewport()
    assert view.view.dragMode() == QGraphicsView.NoDrag
    assert canvas.cursor().shape() == Qt.ArrowCursor
    view.set_markers([("pinned", 500, 400, "pinned"), ("local", 620, 400, "ignored"),
                      ("global", 380, 400, "ignored"), ("auto", 750, 400, "placed")],
                     available=["pinned", "local", "global", "auto"],
                     ignored_here={"local"}, ignored_all={"global"})

    labels = {item.text(): item for item in view._overlay if isinstance(item, QGraphicsSimpleTextItem)}
    menus = []
    during_popup = []

    class InspectableMenu(QMenu):
        def exec(self, _position):
            QApplication.sendEvent(canvas, QEvent(QEvent.Leave))
            during_popup.append((view._hovered_flag, tuple(
                name for name, item in labels.items() if item.brush().color() == HOVER_COLOR)))
            menus.append([(action.text(), action.isEnabled(),
                           [(sub.text(), sub.isEnabled()) for sub in action.menu().actions()]
                           if action.menu() else None) for action in self.actions()])
    monkeypatch.setattr(views, "QMenu", InspectableMenu)

    def on_point(u):
        return view.view.mapFromScene(QPointF(u, 400)) + QPoint(3, -17)

    qtbot.mouseMove(canvas, on_point(500))
    assert view._flags["pinned"].hovered
    assert labels["pinned"].brush().color() == HOVER_COLOR
    qtbot.mouseMove(canvas, on_point(620))
    assert view._flags["local"].hovered and not view._flags["pinned"].hovered
    assert labels["local"].brush().color() == HOVER_COLOR
    assert labels["pinned"].brush().color() == color("pinned")
    for u in (500, 620, 380, 750):
        view._context(on_point(u))
        assert view._hovered_flag is None
    assert during_popup == [(name, (name,)) for name in ("pinned", "local", "global", "auto")]
    assert labels["auto"].brush().color() == color("placed")
    colors = [menu[0][1] for menu in menus]
    assert colors == [False, True, True, True]
    assert [name for name, _, _ in menus[0]] == [
        "Pin GCP", "Move GCP", "Ignore GCP", "Delete GCP", "Show GCP on table"]
    assert menus[1][2][2] == [("Only from this photo", False), ("From all photos", True)]
    assert menus[2][2][2] == [("Only from this photo", False), ("From all photos", False)]
    blank = view.view.mapFromScene(QPointF(500, 550))
    qtbot.mouseMove(canvas, blank)
    assert not any(flag.hovered for flag in view._flags.values())
    view._context(blank)
    assert [name for name, _, _ in menus[-1]] == ["Add GCP", "Place Existing GCP"]

    view.zoom(3)
    scrollbar = view.view.horizontalScrollBar()
    start_value = scrollbar.value()
    from PySide6.QtGui import QMouseEvent
    start = canvas.rect().center()
    qtbot.mousePress(canvas, Qt.LeftButton, pos=start)
    end = QPointF(start + QPoint(40, 0))
    QApplication.sendEvent(canvas, QMouseEvent(QEvent.MouseMove, end, end, Qt.NoButton,
                                               Qt.LeftButton, Qt.NoModifier))
    qtbot.mouseRelease(canvas, Qt.LeftButton, pos=end.toPoint())
    assert scrollbar.value() < start_value
    assert canvas.cursor().shape() == Qt.ArrowCursor



def test_gcp_table_selection_highlights_visible_flags_and_restores_their_colors(window, tmp_path, qtbot, monkeypatch):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QImage, QPainter
    from photomesh import orthoview
    from photomesh.marker_style import HOVER_COLOR, color

    ch = _scene(window, tmp_path)
    ch.markers = [
        g.Marker("A", projections={"img_0001": [500, 400, True]}).to_dict(),
        g.Marker("B", projections={"img_0001": [680, 400, False, "refined"]}).to_dict(),
        g.Marker("not in view").to_dict(),
    ]
    window.refresh_all()
    window.open_photo(0)
    window._show_dock("Markers", True)
    photo, ortho, model = window.photo_view, window.ortho_view, window.model_view
    table = window.markers_pane
    ortho.center, ortho.scale = (10., 0.), 3.
    ortho.marker_states = model.marker_states = {"A": "pinned", "B": "placed"}
    ortho.set_markers([("A", 0., 0.), ("B", 20., 0.)])
    model.set_markers([[-.5, 0, 0], [.5, 0, 0]], labels=["A", "B"], control=[True, False])
    first_photo_flag = photo._flags["A"]

    painted_ortho_flags = []
    monkeypatch.setattr(orthoview, "draw_flag", lambda _p, _x, _y, col, _active: painted_ortho_flags.append(col))

    def assert_colors(a, b):
        expected = [a, b]
        assert [photo._flags[name].display_color() for name in ("A", "B")] == expected
        assert [photo._flags[name].label_item.brush().color() for name in ("A", "B")] == expected
        image = QImage(ortho.size(), QImage.Format_ARGB32)
        painter = QPainter(image)
        try:
            painted_ortho_flags.clear()
            ortho._paint(painter)
        finally:
            painter.end()
        assert painted_ortho_flags == expected
        assert [item[3] for item in model.overlay.items(np.eye(4), 200, 100)] == expected

    def click(row, modifier=Qt.NoModifier):
        cell = table.visualItemRect(table.item(row, 0)).center() + QPoint(10, 0)
        qtbot.mouseClick(table.viewport(), Qt.LeftButton, modifier, pos=cell)

    green, blue = color("pinned"), color("placed")
    assert_colors(green, blue)
    click(0)
    assert table.selected_labels() == ["A"]
    assert_colors(HOVER_COLOR, blue)
    click(1, Qt.ControlModifier)
    assert table.selected_labels() == ["A", "B"]
    assert_colors(HOVER_COLOR, HOVER_COLOR)
    click(1)
    assert table.selected_labels() == ["B"]
    assert_colors(green, HOVER_COLOR)
    assert photo._flags["A"] is first_photo_flag

    # A hover can still highlight another flag during a table selection.
    qtbot.mouseMove(photo.view.viewport(), photo.view.mapFromScene(QPointF(500, 400)))
    assert photo._flags["A"].display_color() == HOVER_COLOR
    table.clearSelection()
    assert photo._flags["A"].display_color() == HOVER_COLOR
    assert photo._flags["B"].display_color() == blue
    photo._clear_photo_cursor()
    assert_colors(green, blue)
    click(2)
    assert table.selected_labels() == ["not in view"]
    assert_colors(green, blue)


def test_photo_reports_mesh_coordinates_and_original_image_pixels(window, tmp_path, qtbot):
    import re
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication

    ch = _scene(window, tmp_path)
    ch.model_ply = _ground_ply(str(tmp_path / "ground.ply"))
    ch.georef = {"offset": [1000., 2000., 50.], "crs": "LOCAL"}
    window.open_photo(0)
    view = window.photo_view

    def move_cursor(screen):
        viewport = view.view.viewport()
        event = QMouseEvent(QEvent.MouseMove, QPointF(screen), QPointF(viewport.mapToGlobal(screen)),
                            Qt.NoButton, Qt.NoButton, Qt.NoModifier)
        QApplication.sendEvent(viewport, event)

    def assert_cursor():
        screen = view.view.mapFromScene(QPointF(500, 400))
        pixel = view.view.mapToScene(screen)
        move_cursor(screen)
        qtbot.waitUntil(lambda: "Z: 50.000 m" in view.coordinates.text(), timeout=30000)
        lines = view.coordinates.text().splitlines()
        assert len(lines) == 2
        model_xyz = [float(value.replace(",", "")) for value in re.findall(r": ([0-9,.]+)", lines[0])]
        assert model_xyz == pytest.approx([1100 + (pixel.x() - 500) * .06,
                                          2100 - (pixel.y() - 400) * .06, 50], abs=.002)
        assert lines[1] == f"Photo: X: {pixel.x():.2f} px  Y: {pixel.y():.2f} px"

    assert_cursor()
    view.zoom(2)
    view.rotate(90)
    view._clear_photo_cursor()
    move_cursor(view.view.mapFromScene(QPointF(520, 410)))
    assert_cursor()
    ch.cameras[1].aligned = False
    window.open_photo(1)
    move_cursor(view.view.mapFromScene(QPointF(540, 400)))
    qtbot.waitUntil(lambda: "px" in view.coordinates.text(), timeout=3000)
    assert view.coordinates.text().splitlines()[0] == "Model: —"

"""UI: markers pane, marker import/placement, new processing dialogs."""
import numpy as np

from tests.test_dense_mesh_gcp import GCP_TXT


def test_import_markers_and_place(window, image_dir, tmp_path, no_dialogs):
    window.add_photos(sorted(str(p) for p in image_dir.iterdir()))
    p = tmp_path / "GCP.txt"; p.write_text(GCP_TXT, encoding="utf-8")
    window.import_markers(str(p))
    n = window.markers_pane.rowCount()
    from photomesh.alignment_reference import gcp_crs
    assert n >= 3 and gcp_crs(window.dataset).startswith("LOCAL_CS")
    assert not window.dataset.crs
    window.open_photo(0)
    window.markers_pane.selectRow(0)
    window.place_marker(12.5, 34.0)
    lab = window.markers_pane.item(0, 0).text()
    m = [m for m in window.dataset.markers if m["label"] == lab][0]
    assert list(m["projections"].values()) == [[12.5, 34.0, True]]
    assert any(x[0] == lab and x[3] == "pinned" for x in window.photo_view.markers)
    window.remove_marker_projection(lab)
    m = [m for m in window.dataset.markers if m["label"] == lab][0]
    assert m["projections"] == {}
    window.toggle_marker(lab, False)
    assert not [m for m in window.dataset.markers if m["label"] == lab][0]["enabled"]
    window.add_marker()
    assert window.markers_pane.rowCount() == n + 1


def test_new_dialogs_settings(window):
    from photomesh import dialogs as d
    s = d.BuildPointCloudDialog(window).settings()
    assert s["quality"] == "Medium" and s["max_sources"] == 8 and s["window_step"] == 2
    assert s["filters"]["min_confidence"] == 3
    s = d.BuildModelDialog(window).settings()
    assert s["face_count"] == "Medium" and s["interpolation"] == "Enabled"
    s = d.FilterPointCloudDialog(window, True).settings()
    assert s["min_confidence"] == 3 and not s["statistical"]
    s = d.OptimizeCamerasDialog(window).settings()
    assert abs(s["accuracies"]["marker_accuracy"] - 0.005) < 1e-9
    m = {"Reprojection error": np.array([0.1, 0.5, 2.0]), "Image count": np.array([2.0, 3, 5])}
    g = d.GradualSelectionDialog(m, window)
    g.crit.setCurrentText("Image count"); g.level.setValue(3)
    assert g.selection().tolist() == [True, False, False]


def test_import_and_export_use_gcp_reference_independently(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from photomesh.alignment_reference import gcp_crs
    from photomesh.filedialogs import QFileDialog
    from photomesh.georef import CRS, import_markers

    ch = window.dataset
    ch.crs = "EPSG:4326"
    ch.georef = {"crs": ch.crs, "offset": [450000, 4420000, 100]}
    ch.reference_settings["gcp_crs"] = "EPSG:32638"
    txt = tmp_path / "survey.csv"
    txt.write_text("Label,X,Y,Z\nP1,450000,4420000,101\n", encoding="utf-8")
    window.import_markers(str(txt))
    assert ch.crs == "EPSG:4326" and CRS.parse(gcp_crs(ch)).epsg == 32638
    assert ch.markers[0]["ref"] == [450000, 4420000, 101]

    conflicting = tmp_path / "conflicting.txt"
    conflicting.write_text("# CoordinateSystem: EPSG:4326\nLabel,X,Y,Z\nP2,44,40,100\n", encoding="utf-8")
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    window.import_markers(str(conflicting))
    assert warnings and len(ch.markers) == 1

    xml = tmp_path / "gcps.xml"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(xml), "GCP XML"))
    window.export_markers()
    saved, crs = import_markers(str(xml))
    assert crs.epsg == 32638 and saved[0].ref == [450000, 4420000, 101]

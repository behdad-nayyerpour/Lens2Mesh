"""1.0.0 follow-up: readable GL text, inline errors, owned Properties and input cursors."""
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QColor
from PySide6.QtWidgets import QApplication
from photomesh import georef as g
from photomesh.project import Camera


@pytest.mark.parametrize("dpr", [1., 2.])
def test_gl_target_never_receives_font_glyph_draws(qapp, dpr):
    from photomesh.overlay3d import Overlay
    class ImageOnlyTextPainter(QPainter):
        image_calls = 0
        def drawText(self, *args):
            raise AssertionError("Text must be rasterized before reaching the GL target")
        def drawImage(self, *args):
            self.image_calls += 1
            return super().drawImage(*args)
    image = QImage(int(320*dpr), int(240*dpr), QImage.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(dpr); image.fill(QColor(150, 130, 100))
    overlay = Overlay()
    target = ImageOnlyTextPainter(image)
    try:
        rectangles = overlay.paint(target, [("marker", (120., 100.), "GCP65",
                                           QColor(40, 220, 60), False, None)], 320, 240)
    finally:
        target.end()
    assert len(rectangles) == 1 and target.image_calls >= 2
    assert overlay._text_cache
    assert overlay._legend_cache[1].devicePixelRatio() == dpr
    assert "hollow" not in str(overlay._legend_cache[0])
    assert all(p.width() > 0 for p in overlay._text_cache.values())


def test_marker_errors_have_separate_noneditable_summary(window):
    ch = window.dataset
    ch.markers = [g.Marker("A").to_dict(), g.Marker("B").to_dict(),
                  g.Marker("C", enabled=False).to_dict(), g.Marker("D", disabled=True).to_dict()]
    ch.georef = {"rows": [{"label": "A", "total": 3., "pix": 2., "n": 1},
                         {"label": "B", "total": 4., "pix": 4., "n": 3},
                         {"label": "C", "total": 12., "pix": 6., "n": 1},
                         {"label": "D", "total": 1000., "pix": 1000., "n": 1}]}
    window.refresh_all()
    table = window.markers_pane
    assert window.docks["Markers"].windowTitle() == "Ground Control Point"
    assert table.data_row_count == 4 and table.rowCount() == 4
    assert table.horizontalHeader().logicalIndex(1) == 8
    assert table.item(0, 8).text() == "Control" and table.item(2, 8).text() == "Check"
    assert table.item(0, 0).checkState() == Qt.Checked and table.item(2, 0).checkState() == Qt.Unchecked
    assert table.item(0, 0).icon().isNull()
    summary = table.summary_table
    assert summary.rowCount() == 2 and summary.columnCount() == 3
    assert summary.item(0, 1).text() == "3.536" and summary.item(0, 2).text() == "3.606"
    assert summary.item(1, 1).text() == "12.000" and summary.item(1, 2).text() == "6.000"
    for row in range(2):
        assert summary.item(row, 0).data(Qt.UserRole) is None
        assert not summary.item(row, 0).flags() & (Qt.ItemIsEditable | Qt.ItemIsSelectable)
    for group in (1, 2, 0):
        table.filter_group(group)
        assert not summary.isRowHidden(0) and not summary.isRowHidden(1)


def test_camera_error_summary_is_separate_and_updates_with_gps_policy(window):
    east, north = g.utm_forward(np.array([40.]), np.array([44.]), 38, False)
    E, N = float(east[0]), float(north[0])
    ch = window.dataset
    ch.crs = "EPSG:32638"
    ch.georef = {"crs": ch.crs, "offset": [0., 0., 0.]}
    ch.cameras = [Camera("A.jpg", lon=44., lat=40., alt=100., est=[E+3, N+4, 100.]),
                  Camera("B.jpg", lon=44., lat=40., alt=100., est=[E+1000, N, 100.],
                         use_reference=False, enabled=False)]
    window.refresh_all()
    table = window.reference
    assert table.data_row_count == table.rowCount() == 2
    assert table.rowHeight(0) == window.markers_pane.verticalHeader().defaultSectionSize()
    summary = table.summary_table
    assert window.docks["Cameras"].widget().isAncestorOf(summary)
    assert summary.rowCount() == 1 and summary.item(0, 0).text() == "Cameras"
    assert summary.item(0, 1).text() == "—"  # conservative migration of legacy GPS opt-outs
    assert not summary.item(0, 1).flags() & (Qt.ItemIsEditable | Qt.ItemIsSelectable)
    ch.reference_settings["use_photo_gps"] = True
    table.refresh(ch)
    assert table.item(0, 5).text() == summary.item(0, 1).text() == "5.000"
    assert table.error_summary == {"error": 5.0, "measured": 1}
    activated = []
    table.cameraActivated.connect(activated.append)
    table._activate_camera(2, 0)
    table._activate_camera(0, 0)
    assert activated == [0]


def test_properties_belongs_to_workspace_only_and_uses_compact_rows(window):
    assert "Properties" not in window.docks
    assert "View/Properties" not in window.actions
    workspace = window.docks["Project"].widget()
    assert workspace is window.workspace_splitter
    assert workspace.isAncestorOf(window.properties)
    assert not window.docks["Cameras"].isAncestorOf(window.properties)
    assert window.docks["Cameras"].widget().isAncestorOf(window.reference)
    window.properties.show_props({f"Property {i}": f"Value {i}" for i in range(30)})
    assert window.properties.rowCount() == 30
    assert window.properties.rowHeight(0) <= max(18, window.properties.fontMetrics().height()+2)
    assert not window.properties.wordWrap()


@pytest.mark.parametrize("kind", ["PreferencesDialog", "AlignPhotosDialog"])
def test_interactive_settings_release_wait_cursor_and_restore_remaining_work(window, qtbot, kind):
    from photomesh import dialogs
    busy = window.busy_indicator
    dialog = getattr(dialogs, kind)(window); qtbot.addWidget(dialog)
    busy.begin("Preparing settings")
    try:
        assert QApplication.overrideCursor() is not None
        dialog.show(); QApplication.processEvents()
        assert QApplication.overrideCursor() is None
        assert busy.active
        dialog.hide(); QApplication.processEvents()
        assert QApplication.overrideCursor() is not None
    finally:
        dialog.hide()
        busy.end()
    assert QApplication.overrideCursor() is None


def test_finishing_background_work_during_settings_does_not_revive_cursor(window, qtbot):
    from photomesh.dialogs import PreferencesDialog
    busy = window.busy_indicator
    dialog = PreferencesDialog(window); qtbot.addWidget(dialog)
    busy.background("preview", "Preparing preview")
    try:
        dialog.show(); QApplication.processEvents()
        assert QApplication.overrideCursor() is None
        busy.background("preview")
        dialog.hide(); QApplication.processEvents()
        assert not busy.active and QApplication.overrideCursor() is None
    finally:
        dialog.hide()
        busy.background("preview")


def test_processing_dialog_keeps_wait_cursor(window, qtbot):
    from photomesh.dialogs import ProgressDialog
    dialog = ProgressDialog("Processing", window); qtbot.addWidget(dialog)
    busy = window.busy_indicator
    busy.begin("Processing")
    try:
        dialog.show(); QApplication.processEvents()
        assert QApplication.overrideCursor() is not None
    finally:
        dialog.running = False; dialog.close()
        busy.end()
    assert QApplication.overrideCursor() is None


def test_theme_reuses_native_style_when_stylesheet_wraps_it(qapp, monkeypatch):
    from photomesh.theme import apply_theme
    apply_theme(qapp, "Dark")
    calls = []
    original = qapp.setStyle
    def record(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(qapp, "setStyle", record)
    for name in ("Dark", "Light", "Dark"):
        apply_theme(qapp, name)
    assert calls == []


def test_side_panels_and_floating_diagnostics(window):
    assert window.corner(Qt.BottomLeftCorner) == Qt.LeftDockWidgetArea
    assert window.dockWidgetArea(window.docks["Photos"]) == Qt.RightDockWidgetArea
    for name in ("Project", "Markers", "Cameras"):
        assert window.dockWidgetArea(window.docks[name]) == Qt.LeftDockWidgetArea
    for name in ("Console", "Jobs"):
        window.actions[name].setChecked(True)
        QApplication.processEvents()
        assert window.docks[name].isFloating()
        assert not window.docks[name].isHidden()
        window.actions[name].setChecked(False)

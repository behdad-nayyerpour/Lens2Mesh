import json
from pathlib import Path

import numpy as np
import pytest

from photomesh import colmap_io, georef, plyio, survey_capacity
from photomesh.dialogs import OptimizeCamerasDialog
from test_dense_mesh_gcp import _markers_for, synthetic_model
from test_markers_manual import _scene
from test_reference_lifecycle import prepare_test_optimize_exports


def test_report_failure_never_replaces_model(tmp_path):
    model = synthetic_model(n_img=5, n_pts=100)
    path = tmp_path / "sparse"
    colmap_io.write_model(str(path), model)
    landmarks = _markers_for(model, lambda x: 2 * x + [1000, 2000, 300],
                             np.random.default_rng(5), 7, 0)
    original = {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(FileNotFoundError):
        georef.run_task({"model": str(path), "crs": "LOCAL", "bundle": False,
                         "markers": [m.to_dict() for m in landmarks],
                         "report": str(tmp_path / "missing" / "report.txt")})
    assert {p.name: p.read_bytes() for p in path.iterdir()} == original


@pytest.mark.parametrize("finish", ["publish", "cancel", "failed_export", "failed_activation"])
def test_optimize_publication_protects_imported_file_and_previous_frame(window, tmp_path, monkeypatch, finish):
    dataset = _scene(window, tmp_path)
    model = window._sparse_model()
    dataset.crs = "EPSG:32638"
    dataset.georef = {"crs": "EPSG:32638", "offset": [450000., 4400000., 10.]}
    for i, point in enumerate(((102, 95, 0), (118, 105, 0), (130, 91, 3))):
        projections = {im.name: [*uv, True] for iid, im in model.images.items()
                       if (uv := georef.predict_projection(model, iid, point)) is not None}
        dataset.markers.append(georef.Marker(f"P{i}", (2 * np.asarray(point) + [26000, 22000, 1400]).tolist(),
                                             projections=projections).to_dict())
    original_cloud = tmp_path / "imported.ply"
    plyio.save_ply(str(original_cloud), np.array([[101., 101., 2.], [104., 103., 3.]]))
    dataset.point_cloud_ply = str(original_cloud)
    cloud_bytes = original_cloud.read_bytes()
    shape_path = tmp_path / "project-shapes.json"
    shape_path.write_text("{}", encoding="utf-8")
    dataset.shapes = {"path": str(shape_path)}
    live_model = Path(window._model_dir()) / "images.bin"
    model_bytes = live_model.read_bytes()

    def choices(dialog):
        dialog.control_coordinates.setCurrentIndex(dialog.control_coordinates.findData("local"))
        dialog.bundle.setChecked(False)
        return True

    monkeypatch.setattr(OptimizeCamerasDialog, "exec", choices)
    pipeline = []
    monkeypatch.setattr(window, "_run", lambda *args, **kwargs: pipeline.append((args, kwargs)))
    window.optimize_cameras()
    args, kwargs = pipeline[0]
    cfg = json.loads((Path(dataset.workdir) / "georef_task.json").read_text(encoding="utf-8"))
    stage = Path(cfg["stage"])
    georef.run_task(cfg, log=lambda *_: None)
    assert live_model.read_bytes() == model_bytes and original_cloud.read_bytes() == cloud_bytes
    if finish == "cancel":
        kwargs["on_fail"]()
    elif finish == "failed_export":
        with pytest.raises(ValueError, match="camera list is empty"):
            args[2]()
    else:
        prepare_test_optimize_exports(stage)
        if finish == "failed_activation":
            def fail(*_):
                raise ValueError("view activation failed")
            monkeypatch.setattr(window, "_reload_alignment", fail)
            with pytest.raises(ValueError, match="activation failed"):
                args[2]()
        else:
            args[2]()
    assert not stage.exists()
    assert original_cloud.read_bytes() == cloud_bytes
    assert shape_path.read_text() == "{}"
    if finish == "publish":
        assert live_model.read_bytes() != model_bytes
        assert Path(dataset.point_cloud_ply).is_relative_to(dataset.workdir)
        assert Path(dataset.point_cloud_ply).read_bytes() != cloud_bytes
        assert dataset.shapes["stale_frame"]
        assert window._shape_set(dataset) is None
    else:
        assert live_model.read_bytes() == model_bytes
        assert dataset.crs == "EPSG:32638"
        assert dataset.point_cloud_ply == str(original_cloud)


def test_gps_only_from_local_uses_wgs84_and_discards_old_offset(window, tmp_path, monkeypatch):
    dataset = _scene(window, tmp_path)
    dataset.crs = "Local Coordinates (m)"
    dataset.georef = {"crs": "Local Coordinates (m)", "offset": [26000., 22000., 1400.]}
    dataset.reference_settings.update(use_photo_gps=True, use_markers=False, use_cameras=True)
    target = window._model_dir()
    model = window._sparse_model()
    image = model.images[2]
    image.tvec = -image.R() @ (image.center() + [0, 10, 0])
    colmap_io.write_model(target, model)
    for i, camera in enumerate(dataset.cameras):
        camera.lon, camera.lat, camera.alt = 44.001 + i * .001, 40.0 + (i % 2) * .001, 500.
    monkeypatch.setattr(OptimizeCamerasDialog, "exec", lambda _: True)
    pipeline = []
    monkeypatch.setattr(window, "_run", lambda *args, **kwargs: pipeline.append(kwargs))
    window.optimize_cameras()
    assert pipeline
    cfg = json.loads((Path(dataset.workdir) / "georef_task.json").read_text())
    assert cfg["crs"] == "EPSG:4326" and cfg["offset"] is None and not cfg.get("local_only")


def test_weights_and_incompatible_frame_follow_selected_inputs(window):
    window.dataset.crs = "EPSG:3857"
    dialog = OptimizeCamerasDialog(window)
    assert not dialog.adjustment_frame.isEnabled()
    dialog.tie_acc.setValue(0)
    dialog.proj_acc.setValue(0)
    assert dialog.tie_acc.value() > 0 and dialog.proj_acc.value() > 0
    dialog.bundle.setChecked(False)
    assert not dialog.tie_acc.isEnabled() and not dialog.proj_acc.isEnabled()
    assert not dialog.cam_acc.isEnabled() and not dialog.intrinsic_preset.isEnabled()


def test_local_control_unchecks_gps_and_requires_reselecting_it(window):
    dataset = window.dataset
    dataset.crs = "EPSG:32638"
    dataset.reference_settings.update(use_photo_gps=True, gcp_crs="EPSG:32638")
    dataset.markers = [georef.Marker("P1", [450000., 4400000., 100.]).to_dict()]
    dialog = OptimizeCamerasDialog(window, {"use_markers": False, "use_cameras": True})
    dialog.use_markers.setChecked(True)
    assert dialog.use_cameras.isEnabled() and dialog.use_cameras.isChecked()
    assert dialog.settings()["use_cameras"]
    dialog.control_coordinates.setCurrentIndex(dialog.control_coordinates.findData("local"))
    assert not dialog.use_cameras.isEnabled() and not dialog.use_cameras.isChecked()
    assert not dialog.settings()["use_cameras"] and not dialog.reported_accuracy.isEnabled()
    dialog.control_coordinates.setCurrentIndex(dialog.control_coordinates.findData("current"))
    assert dialog.use_cameras.isEnabled() and not dialog.use_cameras.isChecked()

    dataset.reference_settings["gcp_crs"] = "LOCAL"
    dialog = OptimizeCamerasDialog(window, {"use_markers": True, "use_cameras": True})
    assert not dialog.use_cameras.isEnabled() and not dialog.use_cameras.isChecked()


def test_optimize_caption_and_full_height_with_local_grid_note(window, qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialogButtonBox

    monkeypatch.setattr(georef, "bundle_memory_status", lambda *_: (2**30, 2**29, 1000))
    model = synthetic_model(n_img=5, n_pts=100)
    dialog = OptimizeCamerasDialog(window, model=model)
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.button_box.button(QDialogButtonBox.Ok).text() == "Optimize"
    qtbot.waitUntil(lambda: dialog.settings_scroll.verticalScrollBar().maximum() == 0)
    dialog.control_coordinates.setCurrentIndex(dialog.control_coordinates.findData("local"))
    qtbot.waitUntil(lambda: dialog.local_control_note.isVisible()
                    and dialog.settings_scroll.verticalScrollBar().maximum() == 0)


def test_similarity_checks_only_effective_height_references():
    cfg = {"bundle": False, "strict_heights": True, "camera_height_reference": "unknown",
           "marker_height_reference": "ellipsoidal", "adjustment_frame": "enu"}
    crs = georef.CRS.parse("EPSG:32638")
    assert georef.validate_adjustment_references(cfg, crs, "GCPs", 3, 5) == (True, False)
    with pytest.raises(ValueError, match="height references"):
        georef.validate_adjustment_references(dict(cfg, bundle=True), crs, "GCPs", 3, 5)


def test_pixel_only_report_keeps_type_and_projection_count(tmp_path):
    report = tmp_path / "reference.txt"
    georef.write_report(report, [], georef.CRS.parse("LOCAL"), {"Q": .5}, {"Q": 2}, {"Q": "check"})
    row = report.read_text().splitlines()[2].split("\t")
    assert row == ["Q", "check", *([""] * 10), "0.500", "2"]


def test_batched_normal_blocks_have_the_same_numerical_solution(monkeypatch):
    args = {"fit_intrinsics": True, "intrinsics": "f,cx,cy,k1", "max_iterations": 3}
    model = synthetic_model(n_img=6, n_pts=300, two_cameras=False)
    batched = synthetic_model(n_img=6, n_pts=300, two_cameras=False)
    georef.gcp_bundle_adjust(model, {}, {}, args, log=lambda *_: None)
    monkeypatch.setattr(survey_capacity, "BUNDLE_OBSERVATION_BATCH", 120)
    georef.gcp_bundle_adjust(batched, {}, {}, args, log=lambda *_: None)
    np.testing.assert_allclose(batched.points.xyz, model.points.xyz, rtol=1e-9, atol=1e-7)
    np.testing.assert_allclose(batched.cameras[1].params, model.cameras[1].params, rtol=1e-9, atol=1e-7)


def test_all_direct_solver_attempts_failing_is_an_error(monkeypatch):
    model = synthetic_model(n_img=5, n_pts=100)

    def fail(*_):
        raise np.linalg.LinAlgError("singular synthetic camera block")

    monkeypatch.setattr(survey_capacity, "solve_camera_system", fail)
    with pytest.raises(ValueError, match="could not solve iteration 1"):
        georef.gcp_bundle_adjust(model, {}, {}, {"fit_intrinsics": False})

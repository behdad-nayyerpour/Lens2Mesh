"""Reference selection for alignment, before and after camera reconstruction."""
import os
from copy import deepcopy

import numpy as np

from . import georef as g
from .photo_gps import camera_accuracies, camera_gps
from .photonames import marker_camera_keys


# Persist the same result fields after Align Photos and Optimize Cameras.
GEOREF_FIELDS = (
    "offset", "similarity", "crs", "crs_kind", "zone", "rows", "control_rmse",
    "check_rmse", "coordinate_frame", "height_reference", "camera_height_reference",
    "marker_height_reference", "reference_heights_verified", "marker_image_errors",
)
HEIGHT_REFERENCES = (
    ("Unknown / not confirmed", "unknown"),
    ("WGS84 ellipsoidal", "ellipsoidal"),
    ("Orthometric / same geoid", "orthometric"),
    ("Local height datum", "local"),
)


def model_crs(dataset):
    return (dataset.georef or {}).get("crs") or dataset.crs


def gcp_crs(dataset):
    """Old projects used a single CRS for both the model and GCPs."""
    return dataset.reference_settings.get("gcp_crs") or model_crs(dataset) or "LOCAL"


def gcp_frame_matches_model(dataset):
    return g.CRS.parse(gcp_crs(dataset)) == g.CRS.parse(model_crs(dataset))


def control_availability(dataset, crs_text=None):
    """Check source observations even when there is no aligned model yet."""
    if dataset is None:
        return False, "Add at least three non-collinear GCPs with reference XYZ and measurements on two photos each."
    keys = marker_camera_keys(dataset)
    enabled_keys = {key for index, key in keys.items() if dataset.cameras[index].enabled}
    controls = []
    try:
        crs = g.CRS.parse(crs_text or gcp_crs(dataset))
        for record in dataset.markers or []:
            if not record.get("enabled", True) or record.get("disabled") or record.get("ref") is None:
                continue
            point = g.Marker(**record)
            observed = {g._stem(name) for name, value in point.projections.items()
                        if len(value) >= 2 and g.projection_is_measured(value)}
            if len(observed & enabled_keys) >= 2:
                g.validate_control_reference(point, crs)
                controls.append(point.ref)
        if len(controls) >= 3:
            coords = crs.to_metric(np.asarray(controls, float))
            if np.linalg.matrix_rank(coords - coords.mean(0)) >= 2:
                return True, f"{len(controls)} GCPs have measured observations. Coordinates use {crs.name}."
    except (TypeError, ValueError, np.linalg.LinAlgError) as error:
        return False, str(error)
    return False, (f"{len(controls)} GCPs have reference XYZ and measurements on two enabled photos each. "
                   "At least three non-collinear GCPs are required.")


def alignment_reference_config(dataset, settings, mapping, workdir):
    """Snapshot the explicit reference choices for the georeference worker."""
    use_gps = bool(settings.get("gps_frame"))
    use_gcps = bool(settings.get("use_gcps"))
    if not (use_gps or use_gcps):
        return None
    reference = dataset.reference_settings
    selected_crs = (settings.get("gcp_crs") or gcp_crs(dataset)) if use_gcps else None
    existing_frame = dataset.georef or {}
    if selected_crs and g.CRS.parse(selected_crs) != g.CRS.parse(model_crs(dataset)):
        existing_frame = {}
    camera_height = reference.get("camera_height_reference", existing_frame.get("camera_height_reference") or "unknown")
    gcp_height = reference.get("marker_height_reference", existing_frame.get("marker_height_reference") or "unknown")
    if use_gps and use_gcps and settings.get("common_height_reference"):
        camera_height = gcp_height = settings["common_height_reference"]
    crs = g.CRS.parse(selected_crs or model_crs(dataset) or "EPSG:4326")
    if use_gps and crs.kind == "local":
        raise ValueError("Camera GPS requires a geographic/projected CRS. For a local grid, use only GCPs.")
    if use_gcps:
        ready, reason = control_availability(dataset, selected_crs)
        if not ready:
            raise ValueError(reason)
        if use_gps and (camera_height != gcp_height or camera_height == "unknown"):
            raise ValueError("To combine photo GPS and GCPs, confirm their common height reference in Align Photos. "
                             "You can also run alignment with just one reference source.")
    priors = use_gps and (use_gcps or settings.get("gps_priors"))
    return {
        "model": os.path.join(workdir, "sparse", "0"),
        "markers": deepcopy(dataset.markers) if use_gcps else [],
        "crs": selected_crs or model_crs(dataset) or "EPSG:4326",
        "use_markers": use_gcps,
        "use_cameras": use_gps,
        "require_gcps": use_gcps,
        "require_camera_gps": use_gps,
        "camera_gps": camera_gps(dataset, mapping) if use_gps else {},
        "camera_accuracies": camera_accuracies(dataset, mapping,
                                                reported=settings.get("reported_gps_accuracy", False)) if priors else {},
        "accuracies": {key: reference.get(key, default) for key, default in g.DEFAULTS.items()},
        "camera_height_reference": camera_height,
        "marker_height_reference": gcp_height,
        "strict_heights": use_gcps,
        "offset": existing_frame.get("offset"),
        "adjustment_frame": "enu" if existing_frame.get("coordinate_frame") else "projected",
        "coordinate_frame": existing_frame.get("coordinate_frame"),
        "fit_intrinsics": False,
        "bundle": bool(use_gcps or priors),
        "result": os.path.join(workdir, "alignment_georef.json"),
        "report": os.path.join(workdir, "alignment_georef.txt"),
    }

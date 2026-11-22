# Local control grid candidate for Lens2Mesh 1.0.6

Optimize Cameras now accepts a local survey grid as an explicit interpretation
of the control XYZ. This path fits the existing reconstruction using measured
controls, excludes camera GPS and does not reuse the geographic offset or ENU
origin. The ordinary project-reference workflow keeps its default. Three
non-collinear controls observed in at least two aligned photos each are required.
The project reference and settings are updated by the existing success callback.

Reviewed the dialog, reference preflight, existing solver and product
invalidation. Kept the coordinate range check and the existing transformation
logic. No dependencies or engine changes. Tests cover both an independently
known local translation from a geographic project and insufficient controls.
The PDF section for Optimize Cameras should be reviewed before a full release.

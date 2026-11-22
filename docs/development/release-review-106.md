# Lens2Mesh 1.0.6 review

## Scope and source contracts

Reviewed pair-file writers/readers, native command lifecycle, final-database
matching, dynamic mapper fallbacks, camera snapshots, model export, project
identity, cancellation, frame preservation, GPS postprocessing, and view/picking
matrix conventions. Source photos and the last published reconstruction remain
unchanged until successful alignment publication. Native processes remain direct
QProcess children so cancellation does not orphan a wrapper's child.

The repository-wide inventory parsed 258 Python files, including 127 application
and vendored modules. Undefined-name/export checks also cover packaging, tools
and tests. The inventory found no exact nontrivial duplicate function bodies and
no configured application static diagnostics. This is a broad automated sweep
plus targeted manual contract review, not proof that every line is bug-free.

There are 129 unreferenced-name candidates (mostly vendored interfaces and Qt or
Python hooks), and 53 size/complexity candidates. Inspected the application-only
candidates against signal handlers, entry points, compatibility and script APIs.
Unused-looking helpers such as dem_mesh, external_workdirs and read_projections_csv
remain available; no speculative deletion or broad numerical refactor is included.
Large orthomosaic, texture, DEM and adjustment functions remain maintainability
follow-up areas. CPU/GPU and legacy/modern engine paths deliberately stay separate.

## Corrections

- Parse complete photo names exactly; use journalled temporary database aliases
  only during native pair import. Restore names on success, failure, launch error,
  cancellation or a subsequent recovery after an interrupted process.
- Refuse an empty verified match graph before starting camera estimation.
- Resume final matching/mapping using saved feature data and valid camera models.
  Reject changed inputs/reference metadata and incomplete snapshots. Keep prior
  sparse models; preserve calibration groups already keyed by engine aliases.
- Native integration exposed COLMAP's different output layout with input_path;
  resumed models now explicitly target sparse/0 before export.
- Re-probe matching device flags on resume; retain saved feature/pair selection.
- Restrict resume snapshots to the final sparse model: coarse snapshots belong
  to another feature database and cannot seed final-resolution tracks.
- Keep mapping eligible through global/hierarchical validation and fallback;
  enter postprocessing only when a postprocessing command actually starts.
- Restore raw GPS backup before repeated postprocessing to avoid double transform.
- Pan using the screen basis at both poles; allow complete elevation/azimuth turns
  while retaining the existing picking and axis-indicator coordinate contract.
- Reuse the identical optional NVIDIA payload for 1.0.6; keep its installer separate.

## Validation retained outside the repository

Evidence directory: Lens2Mesh-validation-evidence/release-106.

- 170 focused regression tests passed before the final resume review; 54 affected
  tests passed after calibration and output-layout review, including an added
  device-change regression. Full Windows/Linux suite and installed candidate
  checks are required before acceptance; their results will be recorded below.
- Local complete regression run: 1023 passed, 14 skipped in 330.90 seconds;
  subsequent final-recovery selection/lifecycle regressions: 10 passed. Skips
  explicitly require real OpenGL, external processed data, or filesystem link
  privileges. A test-only DXF oracle was installed in an isolated evidence folder;
  the personal installation and its running reconstruction were not changed.
- Personal-baseline delta: 3,021,864 bytes before the final QA-log correction;
  applied to a disposable copy, exact installed source identity and offline
  Help/About/licence in both themes passed (169 application files). Native
  cancelled/resumed alignment registered all 16 cameras, centre RMS 0.02497 m,
  reprojection RMS 0.22216 px. Installed console validation is UTF-8 explicitly;
  this avoids a QA-only cp1252 console error for Persian photo names.
- Native COLMAP 4.2.1 CPU test: 16 synthetic photos with spaces and Persian names;
  cancelled after a valid 12-camera snapshot, resumed to all 16 cameras; camera
  centre RMS after independent similarity fit 0.02550 m, reprojection RMS 0.22589 px.
  This synthetic result does not establish survey accuracy on arbitrary projects.
- Ruff application checks and repository-wide undefined-name/export checks passed;
  packaging PowerShell parsed without errors.
- Approved limited manual change: original complete 119-page 1.0.5 PDF retained;
  current PDF has 121 pages. All 114 unaffected body pages rendered identically;
  changed cross-references, contents and appended recovery/navigation pages were
  rendered and visually checked. Offline Help validation uses its exact checksum.

## Gates and limitations

No complete Setup is built before the user's explicit acceptance of the tested
incremental candidate. Public publication was requested separately. The user's
currently running 1.0.5 recovery is not modified or stopped.

Recovery starts at final matching, not during initial feature extraction/coarse
alignment. Work since the last valid snapshot may repeat; global/hierarchical
solver internals are not serialised. Earlier versions cannot gain snapshots
mid-run. Real GPU, driver-specific rendering, real-survey validation and exact
installed apply/rollback checks remain separate from source tests. Existing
third-party native source/build-provenance follow-up remains open; this change
does not assert redistribution or trademark clearance.

## Incremental processing progress

The review traced the runner, modal background jobs and their callers through
alignment, dense reconstruction, interrupted fusion, validation, result
publication and the separate 3D loader. Recalculating weighted totals as the
runner inserted new steps could leave the displayed bar ahead of the active
stage. Depth-map and fusion starts were also mixed with completed work.

Progress now reports the active stage and observed counts. Matching uses native
completion logs. The incremental mapper shows the largest reported registration
count while global bundle adjustment has no time estimate. Depth-map progress
counts newly written depth and normal files after checking their headers and
lengths. Each pass and fusion attempt has a fresh estimate. Validation and
publication remain indeterminate and only successful publication completes the
dialog. These changes do not modify the numerical algorithms, engine options,
project files or the 3D view loader.

Verification covers split process output, counter completion, stage and retry
resets, incomplete maps, caches and publication failures. Native runs use the
same small alignment and dense inputs for the previous build and the candidate.
The installed test update is checked on a disposable copy with a Unicode path.
Per-stage estimates still depend on stable measured throughput. Step totals and
time estimates are unavailable when the native engine only reports a phase name.

"""Progress presentation and ETA behavior under changing real workloads."""
from photomesh.progress import WorkProgress, fmt_eta


def test_eta_needs_samples_and_expires_without_new_work():
    now = [0.0]
    meter = WorkProgress(clock=lambda: now[0])
    for when, done in ((0, 1), (4, 3)):
        now[0] = when
        meter.update(done, 10)
        assert meter.remaining() is None
    now[0] = 8
    meter.update(5, 10)
    assert meter.remaining() == 10
    now[0] = 10
    assert meter.remaining() == 10  # Do not invent throughput while idle.
    now[0] = 30
    assert meter.remaining() is None
    assert meter.done == 5 and "estimating" in meter.text()


def test_cache_burst_and_new_workload_hide_unreliable_eta():
    now = [0.0]
    meter = WorkProgress(clock=lambda: now[0])
    for when, done in ((0, 1), (5, 51), (10, 52)):
        now[0] = when
        meter.update(done, 100)
    assert meter.remaining() is None
    meter.update(1, 200)
    assert meter.remaining() is None and len(meter.samples) == 1
    meter.update(float("nan"), 200)
    assert meter.done == 1
    assert fmt_eta(float("inf")) == ""


def test_stage_counts_hide_percent_during_publication(qtbot):
    from photomesh.dialogs import ProgressDialog
    from photomesh.progress import StageProgress
    dialog = ProgressDialog("Build")
    qtbot.addWidget(dialog)
    dialog.set_stage_progress(StageProgress("Building depth maps", 7, 10, "maps written"))
    assert dialog.stage_bar.value() == 700
    assert "7/10 maps written" in dialog.timing.text()
    dialog.set_stage_progress(StageProgress("Building depth maps", 10, 10, "maps written"))
    assert dialog.stage_bar.maximum() == 0
    assert "10/10 maps written" in dialog.timing.text()
    assert "100" not in dialog.timing.text()
    dialog.begin_stage("Finalizing results...")
    assert dialog.stage_bar.maximum() == 0
    assert "Finalizing" in dialog.stage.text()
    dialog.finish_success()
    assert dialog.stage_bar.value() == 1000
    assert dialog.stage.text() == "Finished"
    dialog.running = False


def test_new_phase_or_restarted_step_discards_previous_rate(qtbot):
    from photomesh.dialogs import ProgressDialog
    from photomesh.progress import StageProgress
    dialog = ProgressDialog("Cloud")
    qtbot.addWidget(dialog)
    now = [0.0]
    dialog._meter = WorkProgress(clock=lambda: now[0])
    dialog._started = now[0]
    for when, done in ((0, 1), (5, 2), (10, 3)):
        now[0] = when
        dialog.set_stage_progress(StageProgress("Photometric depth maps", done, 10))
    assert dialog._meter.remaining() is not None
    dialog.set_stage_progress(StageProgress("Geometric depth maps", 1, 10))
    assert dialog._meter.remaining() is None
    assert dialog.stage_bar.value() == 100
    dialog.begin_stage("Geometric depth maps")
    assert dialog.stage_bar.maximum() == 0 and not dialog._meter.samples
    dialog.running = False


def test_mapper_reports_confirmed_count_without_guessing_eta():
    from photomesh.progress import MappingMeter
    m = MappingMeter(3569)
    first = m.feed("Registering image #1130 (num_reg_frames=3531)")
    assert first.done == 3531 and first.total == 3569 and not first.estimate
    assert m.feed("=> Could not register, trying another image.") is None
    m.feed("Registering image #1130 (num_reg_frames=3532)")
    ba = m.feed("Retriangulation and Global bundle adjustment")
    assert "Global bundle adjustment" in ba.label
    assert ba.done == 3532 and ba.fraction is None
    next_cluster = m.feed("Registering image #78 (num_reg_frames=2)")
    assert next_cluster.done == 3532


def test_matching_counts_completed_blocks_and_handles_two_index_logs():
    from photomesh.progress import MatchingMeter
    m = MatchingMeter()
    assert m.feed("pairing.cc:214] Processing block [1/3, 1/3]").done == 0
    assert m.feed("feature_matching.cc:218] in 94.679s").done == 1
    assert m.feed("pairing.cc:214] Processing block [1/3, 2/3]").total == 0
    assert m.feed("feature_matching.cc:218] in 5s").done == 2
    m = MatchingMeter()
    a = m.feed("pairing.cc:938] Processing block [1/26]")
    assert (a.done, a.total) == (0, 26)
    b = m.feed("feature_matching.cc:218] in 94.679s")
    assert (b.done, b.total) == (1, 26)

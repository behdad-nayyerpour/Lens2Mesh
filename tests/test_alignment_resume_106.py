from pathlib import Path
from types import SimpleNamespace

import pytest

from photomesh import alignment_resume as resume
from photomesh.colmap import Cmd,PipelineRunner
from photomesh.project import Camera,ProjectData


@pytest.fixture
def pending(tmp_path):
    image=tmp_path/'photo (1).JPG';image.write_bytes(b'original photo')
    dataset=ProjectData(cameras=[Camera(str(image))])
    work=tmp_path/'alignment-test';work.mkdir()
    record=resume.create(work,dataset,{'feature_method':'SIFT'},{str(image):image.name})
    return dataset,work,record


def test_only_unfinished_ready_alignments_with_unchanged_sources_are_offered(pending):
    dataset,work,record=pending
    assert resume.pending(work.parent,dataset) is None
    record['phase']='mapping';resume.save(work/'alignment-resume.json',record)
    assert resume.pending(work.parent,dataset)[0]==str(work)
    # Cached display metadata is not a source identity change.
    dataset.cameras[0].desc='new cache';dataset.cameras[0].width=4000
    assert resume.pending(work.parent,dataset)
    dataset.reference_settings['reported_gps_accuracy']=False
    assert resume.pending(work.parent,dataset)
    Path(dataset.cameras[0].path).write_bytes(b'changed source photo')
    assert resume.pending(work.parent,dataset) is None


def test_completed_disabled_missing_or_reference_changed_is_not_resumed(pending):
    dataset,work,record=pending
    record['phase']='complete';resume.save(work/'alignment-resume.json',record)
    assert resume.pending(work.parent,dataset) is None
    record['phase']='mapping';resume.save(work/'alignment-resume.json',record)
    dataset.crs='EPSG:32611'
    assert resume.pending(work.parent,dataset) is None
    dataset.crs='';dataset.cameras[0].enabled=False
    assert resume.pending(work.parent,dataset) is None
    dataset.cameras[0].enabled=True;Path(dataset.cameras[0].path).unlink()
    assert resume.pending(work.parent,dataset) is None


def test_final_matching_and_mapper_checkpoints_are_scoped_to_final_database(pending):
    dataset,work,record=pending
    def action(name,db):return Cmd([name,'--database_path',str(work/db)])
    coarse=action('matches_importer','coarse.db')
    final=action('matches_importer','database.db')
    mapper=action('mapper','database.db')
    items=resume.instrument([('coarse',coarse,1),('final',final,1),('map',mapper,1)],str(work),record)
    assert not hasattr(items[0][1],'on_start')
    final.on_start();assert record['phase']=='matching'
    final.on_success();assert record['phase']=='mapping'
    mapper.on_start();assert record['mapper_kind']=='Incremental'
    mapper.on_success();assert record['phase']=='postprocessing'


def test_dynamic_callback_and_fallback_steps_keep_instrumentation(pending):
    dataset,work,record=pending
    from photomesh.colmap import Try
    fallback=lambda:[('retry',Cmd(['mapper','--database_path',str(work/'database.db')]),1)]
    command=Try(['global_mapper','--database_path',str(work/'database.db')],fallback)
    planned=resume.instrument([('plan',lambda:[('map',command,1)],1)],str(work),record)
    expanded=planned[0][1]()
    assert hasattr(expanded[0][1],'on_start')
    retry=expanded[0][1].fallback()
    assert hasattr(retry[0][1],'on_success')


def test_resume_reuses_features_and_matches_and_preserves_old_models(pending,monkeypatch):
    dataset,work,record=pending
    from photomesh import fastpipe
    record['phase']='mapping'
    sparse=work/'sparse';sparse.mkdir();(sparse/'keep.bin').write_bytes(b'partial model')
    monkeypatch.setattr(resume,'choose_input',lambda wd:None)
    called=[]
    monkeypatch.setattr(fastpipe,'mapper_steps',lambda *args,**kw:called.append(args) or [('map',Cmd(['mapper']),1)])
    planned=resume.steps('engine',str(work),record,dataset)
    assert called and not any('feature_extractor' in a for _,a,_ in planned if isinstance(a,list))
    assert len(list(work.glob('previous-sparse-*')))==1
    assert (next(work.glob('previous-sparse-*'))/'keep.bin').read_bytes()==b'partial model'


def test_postprocessing_uses_raw_backup_without_repeating_gps_transform(pending):
    dataset,work,record=pending
    record['phase']='postprocessing'
    backup=work/'sparse/0_original';backup.mkdir(parents=True)
    current=work/'sparse/0';current.mkdir()
    (backup/'model.bin').write_bytes(b'raw');(current/'model.bin').write_bytes(b'transformed')
    planned=resume.steps('engine',str(work),record,dataset)
    assert (current/'model.bin').read_bytes()==b'raw'
    assert not any('mapper' in a for _,a,_ in planned if isinstance(a,list))


def test_corrupt_newest_snapshot_is_skipped(pending,monkeypatch):
    import struct
    from photomesh import engine
    dataset,work,record=pending
    for number,count in ((1,10),(2,20)):
        folder=work/'checkpoints'/'sparse'/str(number);folder.mkdir(parents=True)
        (folder/'images.bin').write_bytes(struct.pack('<Q',count))
    def validate(path):
        if Path(path).name=='2':raise EOFError('interrupted write')
        return SimpleNamespace()
    monkeypatch.setattr(engine,'validate_model',validate)
    assert resume.choose_input(work).name=='1'


def test_coarse_camera_snapshots_cannot_seed_final_feature_tracks(pending,monkeypatch):
    import struct
    from photomesh import engine
    dataset,work,record=pending
    for stage,count in (('sparse_coarse',100),('sparse',10)):
        folder=work/'checkpoints'/stage/'snapshot';folder.mkdir(parents=True)
        (folder/'images.bin').write_bytes(struct.pack('<Q',count))
    monkeypatch.setattr(engine,'validate_model',lambda path:SimpleNamespace())
    assert resume.choose_input(work).parent.name=='sparse'


def test_success_hook_runs_only_after_successful_native_command(qtbot):
    import sys
    events=[]
    action=Cmd(['-c','print("done")'])
    action.on_start=lambda:events.append('start')
    action.on_success=lambda:events.append('complete')
    runner=PipelineRunner(sys.executable,[('saved stage',action,1)])
    results=[];runner.finished.connect(lambda *a:results.append(a))
    runner.start();qtbot.waitUntil(lambda:bool(results),timeout=10000)
    assert events==['start','complete'] and results[0][0]


def test_resume_matching_reprobes_device_and_keeps_saved_pair_selection(pending,monkeypatch):
    from photomesh import engine,fastpipe
    dataset,work,record=pending
    record.update(phase='matching',pairs=None,matching_args=['exhaustive_matcher',
        '--database_path',str(work/'database.db'),'--FeatureMatching.use_gpu','1',
        '--FeatureMatching.gpu_index','0','--SiftMatching.guided_matching','1'])
    monkeypatch.setattr(engine,'matching_arguments',lambda *a:['--FeatureMatching.use_gpu','0'])
    monkeypatch.setattr(fastpipe,'mapper_steps',lambda *a:[])
    monkeypatch.setattr(resume,'choose_input',lambda *a:None)
    planned=resume.steps('engine',str(work),record,dataset,{'use_gpu':False})
    args=planned[0][1]
    assert '--FeatureMatching.gpu_index' not in args
    assert args[args.index('--FeatureMatching.use_gpu')+1]=='0'
    assert args[args.index('--SiftMatching.guided_matching')+1]=='1'

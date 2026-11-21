"""Resume final matching/mapping without touching the last published alignment."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import uuid


def signature(dataset):
    photos=[]
    for camera in dataset.cameras:
        if not camera.enabled:continue
        stat=os.stat(camera.path)
        photos.append([os.path.abspath(camera.path),stat.st_size,stat.st_mtime_ns,
                       camera.lon,camera.lat,camera.alt,camera.use_reference])
    reference=dict(dataset.reference_settings)
    reference.setdefault('reported_gps_accuracy',False)
    payload=dict(photos=photos,reference=reference,crs=dataset.crs,
                 markers=dataset.markers,georef=dataset.georef)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()


def save(path,record):
    path=Path(path);temp=path.with_suffix('.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2)
        stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def create(workdir,dataset,settings,mapping):
    record=dict(schema=1,signature=signature(dataset),settings=dict(settings),mapping=mapping,
                phase='preparing')
    save(Path(workdir)/'alignment-resume.json',record)
    return record


def pending(data_dir,dataset):
    try:expected=signature(dataset)
    except OSError:return None
    for path in sorted(Path(data_dir).glob('alignment-*/alignment-resume.json'),
                       key=lambda p:p.stat().st_mtime_ns,reverse=True):
        try:
            record=json.loads(path.read_text(encoding='utf-8'))
            if (record.get('schema')==1 and record.get('signature')==expected and
                    record.get('phase') in ('matching','mapping','postprocessing')):
                return str(path.parent),record
        except (OSError,ValueError):continue
    return None


def instrument(steps,workdir,record):
    from .colmap import Cmd,PYTHON
    database=str(Path(workdir)/'database.db')
    path=Path(workdir)/'alignment-resume.json'
    result=[]
    def wrap(callback):
        return lambda:instrument(callback() or [],workdir,record)
    for label,action,weight in steps:
        if callable(action):action=wrap(action)
        elif isinstance(action,list) and action:
            if not hasattr(action,'meter'):action=Cmd(action)
            if getattr(action,'fallback',None):action.fallback=wrap(action.fallback)
            args=list(action)
            def option(key,args=args):
                return args[args.index(key)+1] if key in args else None
            final_db=option('--database_path')==database
            matching=final_db and args[0] in ('matches_importer','exhaustive_matcher','sequential_matcher')
            mapping=final_db and args[0] in ('mapper','global_mapper','hierarchical_mapper')
            post=((args[0]==PYTHON and len(args)>1 and args[1] in ('georef','preserve_frame')) or
                  (args[0]=='model_converter' and label=='Exporting tie points'))
            if matching or mapping or post:
                def before(args=args,matching=matching,mapping=mapping,post=post,action=action):
                    if matching:
                        record.update(phase='matching',matching_args=args,
                                      pairs=str(action.pairs) if hasattr(action,'pairs') else None)
                    elif mapping:
                        record.update(phase='mapping',mapper_kind={'mapper':'Incremental',
                            'global_mapper':'Global','hierarchical_mapper':'Hierarchical'}[args[0]])
                    elif post:record['phase']='postprocessing'
                    save(path,record)
                action.on_start=before
            if matching or mapping:
                def after():
                    # Global/hierarchical plans may still validate and fall
                    # back after the native command exits successfully.
                    record['phase']='mapping'
                    save(path,record)
                action.on_success=after
        result.append((label,action,weight))
    return result


def choose_input(workdir):
    """Ignore half-written snapshots; prefer the largest valid model."""
    from .engine import validate_model
    root=Path(workdir)
    # Coarse-to-fine uses a different feature database. Its camera snapshots
    # must never seed final-resolution tracks, even if they contain more cameras.
    candidates=list((root/'checkpoints'/'sparse').glob('**/images.bin'))
    candidates.extend(image for image in (root/'sparse').glob('*/images.bin') if image.parent.name.isdecimal())
    ranked=[]
    for image in candidates:
        try:
            with image.open('rb') as stream:count=struct.unpack('<Q',stream.read(8))[0]
            ranked.append((count,image.stat().st_mtime_ns,image.parent))
        except (OSError,ValueError,struct.error):continue
    for _count,_time,path in sorted(ranked,reverse=True):
        try:
            validate_model(str(path))
            return path
        except (OSError,ValueError,EOFError,IndexError,struct.error):continue
    return None


def steps(exe,workdir,record,dataset,execution=None):
    from . import colmap as cm
    from .fastpipe import mapper_steps
    from .pair_matching import PairImport
    root=Path(workdir);settings=dict(record['settings']);planned=[]
    if execution is not None:settings.update(execution)
    if record['phase']=='matching':
        args=list(record['matching_args'])
        if execution is not None:
            # Re-probe the device without changing the saved feature method.
            # A resumed laptop may no longer have the original GPU available.
            from .engine import matching_arguments
            current=matching_arguments(exe,args[0],settings)
            replace=set(current[::2])|{'--FeatureMatching.gpu_index','--SiftMatching.gpu_index'}
            args=[args[0]]+[part for key,value in zip(args[1::2],args[2::2])
                            if key not in replace for part in (key,value)]+current
        action=(PairImport(args,root/'database.db',record['pairs'],cm.image_list(workdir))
                if record.get('pairs') else args)
        planned.append(('Matching points (resume)',action,15))
    if record['phase']!='postprocessing':
        snapshot=choose_input(workdir)
        if snapshot is not None:
            safe=root/('resume-input-'+uuid.uuid4().hex[:8])
            shutil.copytree(snapshot,safe)
            settings['_resume_input']=str(safe)
            settings['mapper']='Incremental'
            planned.append(('log','Resuming registered cameras from the last valid saved model.',0))
        else:
            settings['mapper']=record.get('mapper_kind',settings.get('mapper','Incremental'))
            planned.append(('log','Reusing saved features and matches; no valid camera snapshot is available yet.',0))
        sparse=root/'sparse'
        if sparse.exists():sparse.rename(root/('previous-sparse-'+uuid.uuid4().hex[:8]))
        sparse.mkdir()
        planned+=mapper_steps(exe,str(root/'database.db'),cm.image_dir(workdir),str(sparse),
                              settings,len([c for c in dataset.cameras if c.enabled]),cm.image_list(workdir))
    else:
        # Georeferencing preserves this raw model before transforming its
        # working copy. Reuse it to avoid a second transform after interruption.
        backup=root/'sparse/0_original'
        if backup.exists():
            current=root/'sparse/0'
            if current.exists():current.rename(root/('previous-final-'+uuid.uuid4().hex[:8]))
            shutil.copytree(backup,current)
    (root/'sparse_txt').mkdir(exist_ok=True)
    planned += [
        ('Exporting tie points',['model_converter','--input_path',str(root/'sparse/0'),
            '--output_path',str(root/'tie_points.ply'),'--output_type','PLY'],1),
        ('Exporting cameras',['model_converter','--input_path',str(root/'sparse/0'),
            '--output_path',str(root/'sparse_txt'),'--output_type','TXT'],1)]
    return planned

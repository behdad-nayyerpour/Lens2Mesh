"""Native whitespace-name alignment and saved-camera resume on synthetic photos."""
import argparse
import json
from pathlib import Path
import sys
import time

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app',type=Path,required=True)
    parser.add_argument('--exe',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(args.app.resolve()))
    from photomesh import alignment_resume, colmap as cm, colmap_io as cio, engine
    from photomesh.project import Camera,ProjectData
    from PySide6.QtCore import QCoreApplication,QTimer
    import numpy as np

    assert Path(cm.__file__).resolve().is_relative_to(args.app.resolve())
    import benchmark_engine_102 as benchmark

    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    scene=benchmark.make_scene(root/'scene')
    truth=cio.read_images(Path(scene['model'])/'images.bin')
    rename={}
    photos=[]
    for index,path in enumerate(scene['photos'].values(),1):
        path=Path(path);new=path.with_name(f'Aerial image ({index}) - نقشه.JPG')
        path.rename(new);rename[path.name]=new.name;photos.append(new)
    from PIL import Image
    dataset=ProjectData()
    for path in photos:
        with Image.open(path) as image:width,height=image.size
        dataset.cameras.append(Camera(str(path),width=width,height=height))
    work=root/'alignment-native';work.mkdir()
    mapping=cm.prepare_images(dataset,str(work))
    settings=dict(mode='Fast',feature_method='SIFT',accuracy='High',key_point_limit=2048,
                  use_gpu=False,single_camera=True,mapper='Incremental',coarse_to_fine=False,
                  generic=False,reference=False,window=5)
    names=[mapping[c.path] for c in dataset.cameras]
    planned=cm.align_steps(args.exe,dataset,str(work),settings,names=names,infos=[c.info() for c in dataset.cameras])
    record=alignment_resume.create(work,dataset,settings,mapping)
    app=QCoreApplication.instance() or QCoreApplication([])

    def run(steps,cancel_at_snapshot=False):
        runner=cm.PipelineRunner(args.exe,alignment_resume.instrument(engine.bind_steps(args.exe,steps),str(work),record))
        done=[];runner.finished.connect(lambda ok,error:done.append((ok,error)))
        runner.log.connect(lambda text:print(text,flush=True))
        timer=QTimer()
        def check():
            if (cancel_at_snapshot and record['phase']=='mapping' and
                    list((work/'checkpoints').glob('**/images.bin')) and alignment_resume.choose_input(work)):
                timer.stop();runner.cancel()
        timer.timeout.connect(check);timer.start(100)
        runner.start()
        while not done:
            app.processEvents();time.sleep(.005)
        timer.stop()
        if cancel_at_snapshot:
            assert not done[0][0] and runner.cancelled,'Snapshot cancellation was not exercised'
        else:assert done[0][0],done[0][1]

    run(planned,True)
    snapshot=alignment_resume.choose_input(work)
    assert snapshot is not None
    saved_cameras=len(cio.read_images(snapshot/'images.bin'))
    assert 2<=saved_cameras<len(photos)
    restored=alignment_resume.pending(root,dataset)
    assert restored and restored[1]['phase']=='mapping'
    run(alignment_resume.steps(args.exe,str(work),restored[1],dataset))
    model=engine.validate_model(str(work/'sparse/0'))
    assert len(model.images)>=12
    assert all(image.name in mapping.values() and ' ' in image.name for image in model.images.values())
    poses=cm.read_images_txt(work/'sparse_txt/images.txt')
    assert set(poses)=={image.name for image in model.images.values()}
    from photomesh.georef import umeyama
    expected={rename[image.name]:image.center() for image in truth.values()}
    src=np.array([image.center() for image in model.images.values()])
    dst=np.array([expected[image.name] for image in model.images.values()])
    scale,rotation,translation=umeyama(src,dst)
    rms=float(np.sqrt(np.mean(np.sum((src@rotation.T*scale+translation-dst)**2,axis=1))))
    assert rms<.5,rms
    errors=cio.observation_errors(model)[0]
    result=dict(passed=True,version='1.0.6',module=str(cm.__file__),registered=len(model.images),
                photos=len(photos),saved_cameras_before_cancel=saved_cameras,
                camera_rms_m=rms,reprojection_rms_px=float(np.sqrt(np.mean(errors**2))),
                whitespace_names_preserved=True,native_snapshot_resumed=True)
    (root/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

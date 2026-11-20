from contextlib import closing
import json
import sqlite3
import sys

import pytest
from PySide6.QtCore import QProcess

from photomesh.colmap import PipelineRunner, read_images_txt
from photomesh.engine import bind_steps
from photomesh.fastpipe import covisible_pairs, read_pairs
from photomesh.pair_matching import PairImport, exact_pairs


@pytest.fixture
def command(tmp_path):
    db=tmp_path/'database.db'
    with closing(sqlite3.connect(db)) as c,c:
        c.execute('CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT UNIQUE)')
        c.executemany('INSERT INTO images VALUES(?,?)',[(1,'DJI1 (1).JPG'),(2,'عکس دوم.JPG')])
        c.execute('CREATE TABLE two_view_geometries(rows INTEGER)')
        c.execute('CREATE TABLE keypoints(image_id INTEGER,data BLOB)')
        c.execute("INSERT INTO keypoints VALUES(1,x'01020304')")
    images=tmp_path/'images.txt';images.write_text('DJI1 (1).JPG\nعکس دوم.JPG\nmissing photo.JPG\n',encoding='utf-8')
    pairs=tmp_path/'pairs.txt';pairs.write_text('DJI1 (1).JPG عکس دوم.JPG\nDJI1 (1).JPG missing photo.JPG\n',encoding='utf-8')
    return PairImport(['matches_importer','--database_path',str(db),'--match_list_path',str(pairs)],db,pairs,images)


def names(command):
    with closing(sqlite3.connect(command.database)) as c:
        return dict(c.execute('SELECT image_id,name FROM images'))


def test_alias_scope_and_features_unchanged(command):
    original=names(command)
    messages=command.prepare()
    assert '1 valid pairs' in messages[0] and '1 pairs excluded' in messages[0]
    assert all(not any(ch.isspace() for ch in n) for n in names(command).values())
    native=open(command[command.index('--match_list_path')+1],encoding='utf-8').read().split()
    assert len(native)==2 and set(native)==set(names(command).values())
    with closing(sqlite3.connect(command.database)) as c,c:
        c.execute('INSERT INTO two_view_geometries VALUES(15)')
    command.cleanup(True)
    assert names(command)==original and not command.journal.exists()
    with closing(sqlite3.connect(command.database)) as c:
        assert c.execute('SELECT data FROM keypoints').fetchone()[0]==b'\x01\x02\x03\x04'


def test_crash_journal_restores_before_next_attempt(command):
    original=names(command);command.prepare()
    second=PairImport(list(command),command.database,command.pairs,command.image_list)
    second.prepare();second.cleanup(False)
    assert names(command)==original


def test_zero_verified_matches_stop_before_mapper_but_restore_names(command):
    original=names(command);command.prepare()
    with pytest.raises(ValueError,match='No geometrically verified'):
        command.cleanup(True)
    assert names(command)==original


def test_ambiguous_or_unknown_names_fail():
    with pytest.raises(ValueError,match='unambiguously'):
        exact_pairs(['a b c'],['a','b c','a b','c'],{})
    with pytest.raises(ValueError,match='unambiguously'):
        exact_pairs(['a unknown'],['a','b'],{})


def test_text_pose_and_covisibility_preserve_whitespace(tmp_path):
    text=tmp_path/'images.txt'
    text.write_text('1 1 0 0 0 1 2 3 1 DJI1 (1).JPG\n0 0 1 1 1 2\n'
                    '2 1 0 0 0 2 3 4 1 عکس دوم.JPG\n0 0 1 1 1 2\n',encoding='utf-8')
    assert set(read_images_txt(text))=={'DJI1 (1).JPG','عکس دوم.JPG'}
    pairs,registered=covisible_pairs(text,min_shared=2)
    assert pairs=={('DJI1 (1).JPG','عکس دوم.JPG')} and len(registered)==2
    pairs_file=tmp_path/'pairs.txt';pairs_file.write_text('DJI1 (1).JPG عکس دوم.JPG\n',encoding='utf-8')
    assert read_pairs(pairs_file,list(registered))==[('DJI1 (1).JPG','عکس دوم.JPG')]


@pytest.mark.parametrize('outcome',['success','failure','cancel','failed-start'])
def test_runner_always_restores_photo_identities(command,qtbot,outcome):
    original=names(command)
    if outcome=='success':
        code="import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.execute('INSERT INTO two_view_geometries VALUES(22)'); c.commit(); c.close()"
    elif outcome=='cancel':code='import time; time.sleep(60)'
    else:code='raise SystemExit(7)'
    command[:]=['-c',code,str(command.database),'--match_list_path',str(command.pairs)]
    exe='missing-pair-matcher-106.exe' if outcome=='failed-start' else sys.executable
    runner=PipelineRunner(exe,bind_steps(exe,[('matching',command,1)]))
    results=[];runner.finished.connect(lambda ok,error:results.append((ok,error)))
    runner.start()
    if outcome=='cancel':
        qtbot.waitUntil(lambda:runner.proc.state()==QProcess.Running,timeout=10000)
        runner.cancel()
    qtbot.waitUntil(lambda:bool(results),timeout=15000)
    assert names(command)==original
    assert results[0][0] is (outcome=='success')
    assert not command.journal.exists()


def test_tampered_identity_not_overwritten(command):
    command.prepare()
    with closing(sqlite3.connect(command.database)) as c,c:
        c.execute("UPDATE images SET name='unexpected' WHERE image_id=1")
    with pytest.raises(ValueError,match='identities changed'):command.cleanup(False)
    assert names(command)[1]=='unexpected' and command.journal.exists()
    assert isinstance(json.loads(command.journal.read_text()),list)

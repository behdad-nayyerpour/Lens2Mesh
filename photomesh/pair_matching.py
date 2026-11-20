"""Native pair imports with stable photo identities, including whitespace names."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import uuid

from .colmap import Cmd


def exact_pairs(lines, known, aliases):
    """The writer separates two complete names by one ASCII space."""
    known=set(known)
    result=set();skipped=0
    for number,line in enumerate(lines,1):
        line=line.rstrip('\r\n')
        if not line:continue
        choices=[(line[:i],line[i+1:]) for i,char in enumerate(line)
                 if char==' ' and line[:i] in known and line[i+1:] in known]
        if len(choices)!=1:
            raise ValueError(f'Image pair line {number} cannot be resolved unambiguously')
        a,b=choices[0]
        if a not in aliases or b not in aliases:
            skipped+=1;continue
        if a!=b:result.add(tuple(sorted((aliases[a],aliases[b]))))
    if not result:raise ValueError('No selected pairs have extracted features for both photos')
    return sorted(result),skipped


class PairImport(Cmd):
    """Scoped native command: aliases exist only while its QProcess runs.

    The runner calls prepare before starting, and cleanup on success, failure,
    failed launch and cancellation. A small journal permits crash recovery.
    No photo files, camera IDs, keypoints or descriptors are renamed/rewritten.
    """
    def __init__(self,args,database,pairs,image_list):
        super().__init__(args)
        self.database=Path(database)
        self.pairs=Path(pairs)
        self.image_list=Path(image_list)
        self.journal=self.database.with_name(self.database.name+'.pair-names.json')
        self.records=None

    def _rename(self,aliased):
        with closing(sqlite3.connect(self.database)) as connection, connection:
            for row in self.records:
                name=connection.execute('SELECT name FROM images WHERE image_id=?',(row['id'],)).fetchone()
                if name is None or name[0] not in (row['original'],row['alias']):
                    raise ValueError('Photo identities changed during pair matching; retained recovery journal')
            connection.executemany('UPDATE images SET name=? WHERE image_id=?',
                [(row['alias'] if aliased else row['original'],row['id']) for row in self.records])

    def prepare(self):
        if self.journal.exists():
            self.records=json.loads(self.journal.read_text(encoding='utf-8'))
            self._rename(False)
            self.journal.unlink()
        with closing(sqlite3.connect(self.database)) as connection:
            images=connection.execute('SELECT image_id,name FROM images ORDER BY image_id').fetchall()
        prefix='lm_pair_'+uuid.uuid4().hex[:12]+'_'
        self.records=[dict(id=i,original=name,alias=prefix+str(i)+'.jpg') for i,name in images]
        known={name for _,name in images}|set(self.image_list.read_text(encoding='utf-8-sig').splitlines())
        aliases={row['original']:row['alias'] for row in self.records}
        pairs,skipped=exact_pairs(self.pairs.read_text(encoding='utf-8-sig').splitlines(),known,aliases)
        safe=self.pairs.with_name(self.pairs.stem+'.native.txt')
        safe.write_text(''.join(a+' '+b+'\n' for a,b in pairs),encoding='utf-8')
        self[self.index('--match_list_path')+1]=str(safe)
        temp=self.journal.with_suffix('.tmp')
        temp.write_text(json.dumps(self.records,ensure_ascii=False),encoding='utf-8')
        temp.replace(self.journal)
        self._rename(True)
        return [f'Pair matching: {len(pairs)} valid pairs; {skipped} pairs excluded because a photo has no features.']

    def cleanup(self,success):
        if self.records is not None and self.journal.exists():
            self._rename(False)
            self.journal.unlink()
        if success:
            with closing(sqlite3.connect(self.database)) as connection:
                count=connection.execute('SELECT coalesce(sum(rows),0) FROM two_view_geometries').fetchone()[0]
            if not count:
                raise ValueError('No geometrically verified photo matches were produced; camera estimation was not started')

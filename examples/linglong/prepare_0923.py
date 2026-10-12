"""Isolated 0923 train/test conversion, repairing raw feedback before alignment."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tarfile
import time
import numpy as np
from PIL import Image

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK / 'linglong_rawdata_spec'))
import linglong_parser as parser
from prepare_sft import prepare, write

SOURCE = Path('/iag_ad_01/ad/zhengyuan/yyzj/data/linglong/0923_100_sample')
TEST = WORK / 'linglong_rawdata_spec/processed_0923_10'
ROOT = WORK / 'openpi/data/linglong_0923_processed'
TRAIN_OUT = WORK / 'openpi/data/linglong_0923_sft_train_15hz'
TEST_OUT = WORK / 'openpi/data/linglong_0923_sft_test_15hz'
PROMPT = 'Use the right arm to pick up the blue block from the table and place it in the box.'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def process(raw, eid, split, source, tar_hash):
    ep=parser.read_episode(raw)
    with sqlite3.connect(f"file:{ep['db']}?mode=ro",uri=True) as c:
        assert c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    for k in parser.TOPICS.values():
        assert len(ep[k])>0, (eid,k,'empty')
        assert all(b['timestamp_ns']>a['timestamp_ns'] for a,b in zip(ep[k],ep[k][1:])), (eid,k,'nonmonotonic')
    drivers=ep['driver']
    q=np.array([parser.flatten_joints(x['limbs']) for x in drivers])
    actions=np.array([parser.flatten_joints(x['limbs'],'q_exp') for x in drivers])
    assert np.isfinite(q).all() and np.isfinite(actions).all()
    fixed,events=parser.repair_isolated_gripper_feedback(q[:,15],actions[:,15])
    if split=='test':
        expected={1:list(range(534,539)),6:[490],8:[398,399]}.get(int(eid),[])
        assert np.flatnonzero(q[:,15]!=fixed).tolist()==expected
    # Standard exporter uses the same raw-rate repair before camera matching.
    dest=ROOT/split/'decoded'/eid
    parser.export_episode(ep,dest,images=True,repair_gripper_feedback=True)
    indices={k:{id(x):i for i,x in enumerate(ep[k])} for k in ('head','left','right')}
    rows=parser.aligned_frames(ep)
    with (dest/'alignment.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps({'timestamp_ns':r['timestamp_ns'],
                'source_timestamps':{k:r[k]['timestamp_ns'] for k in parser.TOPICS.values()},
                'images':{k:f'{k}/{indices[k][id(r[k])]:06d}.jpg' for k in indices}})+'\n')
    image_hash=hashlib.sha256()
    for cam in indices:
        for x in ep[cam]:
            with Image.open(io.BytesIO(x['data'])) as im: im.load()
            image_hash.update(x['data'])
    # Preserve source values. Only q15 is repaired; base2 is masked by the model.
    rec=dict(episode=eid,archive_member=str(source),metadata=ep['metadata'],
             tar_sha256=tar_hash,db_sha256=sha(Path(ep['db'])),images_sha256=image_hash.hexdigest(),
             counts={k:len(ep[k]) for k in parser.TOPICS.values()},repair_events=events,
             repaired_frames=int(np.sum(q[:,15]!=fixed)),
             remaining_feedback_jumps=int(np.sum(np.abs(np.diff(fixed))>.1)))
    write(dest/'quality.json',rec)
    return rec

def main():
    started=time.monotonic()
    ROOT.mkdir(exist_ok=False,parents=True)
    test=[]
    for eid in range(1,11):
        raw=TEST/'raw'/str(eid)
        test.append(process(raw,str(eid),'test',raw/'episode.tar',sha(raw/'episode.tar')))
        print(f'test {eid}/10',flush=True)
    write(ROOT/'test/inventory.json',test)
    test_hashes={r['tar_sha256'] for r in test}
    test_db={r['db_sha256'] for r in test}; test_images={r['images_sha256'] for r in test}
    train=[]; excluded=[]; seen=set()
    for path in sorted(SOURCE.glob('*/*.tar')):
        digest=sha(path)
        if digest in test_hashes:
            excluded.append({'path':str(path),'reason':'identical_test_tar','sha256':digest}); continue
        eid=path.parent.name
        raw=ROOT/'train/raw'/eid;raw.mkdir(parents=True)
        with tarfile.open(path) as tar:
            for member in tar:
                if not member.isfile(): continue
                rel=Path(member.name)
                if rel.is_absolute() or '..' in rel.parts: raise ValueError('unsafe archive path')
                dest=raw/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as inp,dest.open('wb') as out: shutil.copyfileobj(inp,out)
        db=sha(raw/'bag/bag_0.db3')
        if db in test_db or db in seen:
            excluded.append({'path':str(path),'reason':'duplicate_db','sha256':db}); continue
        seen.add(db)
        rec=process(raw,eid,'train',path,digest)
        assert rec['images_sha256'] not in test_images,'test image leakage'
        train.append(rec)
        write(ROOT/'progress.json',{'train_done':len(train),'seconds':time.monotonic()-started})
        print(f'train {len(train)} {eid} repaired={rec["repaired_frames"]}',flush=True)
    write(ROOT/'train/inventory.json',train);write(ROOT/'excluded_sources.json',excluded)
    prepare(ROOT/'train',TRAIN_OUT,prompt=PROMPT,train_fraction=1.)
    prepare(ROOT/'test',TEST_OUT,prompt=PROMPT,train_fraction=0.,
            norm_reference=TRAIN_OUT/'assets/linglong_h/norm_stats.json')
    # Explicit test manifest: validation alias is only for compatibility.
    write(TEST_OUT/'test_episodes.json',json.loads((TEST_OUT/'validation_episodes.json').read_text()))
    info=json.loads((TEST_OUT/'meta/info.json').read_text()); info['splits']={'test':f'0:{info["total_episodes"]}'}
    write(TEST_OUT/'meta/info.json',info)
    write(ROOT/'completion.json',dict(status='complete',train_sources=len(train),test_sources=len(test),
        duplicates_excluded=len(excluded),prompt=PROMPT,seconds=time.monotonic()-started))
    print('COMPLETE',flush=True)

if __name__=='__main__': main()

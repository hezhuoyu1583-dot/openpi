"""Build a local image-reference LeRobot v2.1 dataset; never alter source records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import datasets
from openpi.policies.linglong_policy import PROMPT

# Joint export layout. Training now expects the 14D end-effector contract instead.
JOINT_LAYOUT = "left7,left_gripper,right7,right_gripper,waist4,head2,base2"

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

def prepare(source,output,resume=False,*,prompt=PROMPT,train_fraction=.8,norm_reference=None):
    if not 0 <= train_fraction <= 1: raise ValueError('train_fraction must be in [0,1]')
    output.mkdir(parents=True,exist_ok=resume)
    (output/'meta').mkdir(exist_ok=resume)
    inventory=json.loads((source/'inventory.json').read_text())
    inventory.sort(key=lambda r:r['metadata']['startAt'])
    train_sources={r['episode'] for r in inventory[:int(train_fraction*len(inventory))]}
    features={k:{'dtype':'float32','shape':[24],'names':None} for k in ('observation.state','action')}
    for cam,shape in [('head',[3,720,960]),('left',[3,540,960]),('right',[3,540,960])]:
        features['observation.images.'+cam]={'dtype':'image','shape':shape,'names':['channels','height','width']}
    for k in ('timestamp','frame_index','episode_index','index','task_index'):
        features[k]={'dtype':'float32' if k=='timestamp' else 'int64','shape':[1],'names':None}
    hf=datasets.Features({k:(datasets.Image() if v['dtype']=='image' else
        datasets.Sequence(datasets.Value('float32'),length=24) if v['shape']==[24] else datasets.Value(v['dtype'])) for k,v in features.items()})
    episodes=[]; stats_rows=[]; mapping=[]; train=[]; val=[]; stats_arrays=[]; offset=0
    rejected=0; short=0
    for rec in inventory:
        eid=rec['episode']; path=source/'decoded'/eid
        rows=list(csv.DictReader((path/'aligned.csv').open()))
        alignment=[json.loads(s) for s in (path/'alignment.jsonl').open()]
        ts=np.array([int(r['timestamp_ns']) for r in rows],dtype=np.int64)
        if not np.all(np.diff(ts)>0): raise ValueError('nonmonotonic '+eid)
        # 15 Hz, whole-source filtering; never join across missing data.
        grid=ts[0]+np.rint(np.arange(int((ts[-1]-ts[0])*15/1e9)+1)*1e9/15).astype(np.int64)
        right=np.minimum(np.searchsorted(ts,grid),len(ts)-1); left=np.maximum(right-1,0)
        indices=np.where(abs(ts[left]-grid)<=abs(ts[right]-grid),left,right)
        q=np.array([[float(r[f'q_{i}']) for i in range(24)] for r in rows],dtype=np.float32)
        a=np.array([[float(r[f'q_exp_{i}']) for i in range(24)] for r in rows],dtype=np.float32)
        good=[]
        for tick,i in zip(grid,indices):
            match=alignment[int(i)]
            valid=int(rows[i]['timestamp_ns'])==match['timestamp_ns'] and abs(int(ts[i])-int(tick))<=20_000_000
            valid=valid and all(t is not None and abs(t-int(tick))<=80_000_000 for t in match['source_timestamps'].values())
            valid=valid and np.isfinite(q[i]).all() and np.isfinite(a[i]).all()
            good.append(valid)
        rejected+=len(good)-sum(good)
        raw_ok=all(all(t is not None and abs(t-r['timestamp_ns'])<=80_000_000 for t in r['source_timestamps'].values()) for r in alignment) and np.all(np.diff(ts)<=50_000_000)
        if not all(good) or not raw_ok:
            print(f'{eid}: excluded whole episode',flush=True)
            continue
        bounds=np.flatnonzero(np.diff(np.r_[False,good,False])).reshape(-1,2)
        for start,end in bounds:
            if end-start<25: short+=int(end-start); continue
            selection=indices[start:end]; n=len(selection); ep=len(episodes)
            values={'observation.state':q[selection].tolist(),'action':a[selection].tolist(),
                    'timestamp':(np.arange(n,dtype=np.float32)/15).tolist(),'frame_index':list(range(n)),
                    'episode_index':[ep]*n,'index':list(range(offset,offset+n)),'task_index':[0]*n}
            dest=output/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet'; dest.parent.mkdir(parents=True,exist_ok=True)
            if not (resume and dest.exists()):
                for cam in ('head','left','right'):
                    paths=[path/alignment[int(i)]['images'][cam] for i in selection]
                    if not all(p.is_file() for p in paths): raise FileNotFoundError(eid+' '+cam)
                    values['observation.images.'+cam]=[{'bytes':None,'path':str(p.resolve())} for p in paths]
                datasets.Dataset.from_dict(values,features=hf).to_parquet(str(dest))
            else:
                import pyarrow.parquet as pq
                saved=pq.read_table(dest,columns=['observation.state','action'])
                if saved.num_rows!=n or not np.array_equal(np.array(saved['action'].to_pylist()),a[selection]) or not np.array_equal(np.array(saved['observation.state'].to_pylist()),q[selection]):
                    raise ValueError('resume content mismatch: '+str(dest))
            episodes.append({'episode_index':ep,'tasks':[prompt],'length':n})
            sr={}
            for k,v in [('observation.state',q[selection]),('action',a[selection])]:
                sr[k]={name:arr.tolist() for name,arr in {'min':v.min(0),'max':v.max(0),'mean':v.mean(0),'std':v.std(0),'count':np.array([n])}.items()}
            stats_rows.append({'episode_index':ep,'stats':sr})
            is_train=eid in train_sources; (train if is_train else val).append(ep)
            if is_train: stats_arrays.append(np.concatenate([q[selection],a[selection]],axis=1))
            mapping.append({'episode_index':ep,'source_id':eid,'archive_member':rec['archive_member'],
                            'split':'train' if is_train else 'validation','grid_start_ns':int(grid[start]),
                            'source_row_indices':selection.tolist(),'length':n})
            offset+=n
        print(f'{eid}: segments={len(episodes)} frames={offset}',flush=True)
    for name,values in [('episodes',episodes),('episodes_stats',stats_rows),('tasks',[{'task_index':0,'task':prompt}])]:
        (output/f'meta/{name}.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))
    write(output/'meta/info.json',{'codebase_version':'v2.1','robot_type':'linglong_h',
        'total_episodes':len(episodes),'total_frames':offset,'total_tasks':1,'total_videos':0,
        'total_chunks':(len(episodes)+999)//1000,'chunks_size':1000,'fps':15,
        'splits':{'train':f'0:{len(episodes)}'},'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'video_path':None,'features':features})
    write(output/'train_episodes.json',train); write(output/'validation_episodes.json',val)
    write(output/'source_mapping.json',mapping)
    norm={}; stat_frames=0
    if norm_reference is not None:
        norm=json.loads(Path(norm_reference).read_text())['norm_stats']
    else:
        if not stats_arrays: raise ValueError('No training frames for normalization')
        x=np.concatenate(stats_arrays).astype(np.float64); stat_frames=len(x)
        for key,v in [('state',x[:,:24]),('actions',x[:,24:])]:
            norm[key]={k:arr.tolist() for k,arr in {'mean':v.mean(0),'std':v.std(0),'q01':np.quantile(v,.01,axis=0),'q99':np.quantile(v,.99,axis=0)}.items()}
    write(output/'assets/linglong_h/norm_stats.json',{'norm_stats':norm})
    write(output/'manifest.json',{'status':'complete','prompt':prompt,'layout':JOINT_LAYOUT,'source_root':str(source.resolve()),
        'source_inventory_sha256':hashlib.sha256((source/'inventory.json').read_bytes()).hexdigest(),
        'source_episodes':len(inventory),'segments':len(episodes),'train_segments':len(train),'validation_segments':len(val),
        'frames':offset,'train_stat_frames':stat_frames,'norm_reference':str(norm_reference) if norm_reference else None,'rejected_grid_rows':rejected,'discarded_short_segment_rows':short,
        'resampling':'15Hz nearest head <=20ms; all streams absolute age <=80ms; reject whole source if any invalid grid row, original head gap >50ms or original aligned age >80ms',
        'image_storage':'absolute original JPEG references; source must remain accessible',
        'action_semantics':'absolute q_exp','success_filter':'none; success labels unavailable',
        'split':f'chronological source episodes train fraction {train_fraction}; train/validation JSON overrides meta full coverage'})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Reuse verified existing Parquet files with unchanged source and conversion settings')
    args=p.parse_args(); prepare(args.source,args.output,args.resume)

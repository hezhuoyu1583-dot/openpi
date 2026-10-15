"""Build a local image-reference LeRobot v2.1 dataset; never alter source records.

Joint columns stay 24D. state_eef is the logged /end_pos pose plus the measured
gripper. action_eef is end_frames() of the commanded joints: wrist_yaw_link
orientation and the wrist-frame TCP point. Both are 16D xyz + quaternion wxyz
+ gripper.
"""
import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
import datasets
from end_frames import end_frames
import linglong_parser as parser
from openpi.policies.linglong_policy import PROMPT
from quantile_outliers import filter_dataset

# Joint export layout. The training entry reads state_eef and action_eef.
JOINT_LAYOUT = "left7,left_gripper,right7,right_gripper,waist4,head2,base2"
EEF_LAYOUT = "left_xyz3_quat_wxyz4_gripper,right_xyz3_quat_wxyz4_gripper"
EEF_DIM = 16
MAX_END_AGE_NS = 80_000_000

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

def load_eef_modules():
    return parser, end_frames

def bag_path(source, record):
    episode = record["episode"]
    local = source / "raw" / episode / "bag" / "bag_0.db3"
    if local.is_file():
        return local
    archived = Path(record["archive_member"]).parent / "bag" / "bag_0.db3"
    if archived.is_file():
        return archived
    raise FileNotFoundError(f"no bag for {episode}: tried {local} and {archived}")

def load_end_poses(parser, db_path):
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        topics = {name: topic_id for topic_id, name, *_ in connection.execute(
            "SELECT id,name,type,serialization_format FROM topics")}
        topic_id = topics["/end_pos"]
        poses = {}
        for payload, in connection.execute("SELECT data FROM messages WHERE topic_id=?", (topic_id,)):
            message = parser.decode_end_pos(payload)
            poses[int(message["timestamp_ns"])] = message
    finally:
        connection.close()
    if not poses:
        raise ValueError(f"no /end_pos messages in {db_path}")
    return poses

def rpy_to_matrix(roll, pitch, yaw):
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll), the end_frames convention."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)

def matrix_to_quat(rotation):
    """Rotation matrix to wxyz quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array([
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ])
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quat = np.array([
            (matrix[2, 1] - matrix[1, 2]) / scale,
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
        ])
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quat = np.array([
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
        ])
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quat = np.array([
            (matrix[1, 0] - matrix[0, 1]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
        ])
    return quat / np.linalg.norm(quat)

def continuous_quaternions(quaternions):
    out = np.asarray(quaternions, dtype=np.float64).copy()
    for index in range(1, len(out)):
        if np.dot(out[index], out[index - 1]) < 0.0:
            out[index] *= -1.0
    return out

def recorded_pose(message, arm):
    values = np.asarray(message["ee_pose_l" if arm == "left" else "ee_pose_r"], dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError(f"recorded {arm} pose must be finite (7,)")
    return values[:3], rpy_to_matrix(*values[3:6])

def pack(position, rotation, gripper):
    return np.concatenate([position, matrix_to_quat(rotation), [gripper]])

def geodesic_deg(first, second):
    relative = np.swapaxes(first, -1, -2) @ second
    trace = relative[..., 0, 0] + relative[..., 1, 1] + relative[..., 2, 2]
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))

def end_message(alignment_row, poses):
    head_ns = int(alignment_row["timestamp_ns"])
    end_ns = int(alignment_row["source_timestamps"]["end_pos"])
    if abs(end_ns - head_ns) > MAX_END_AGE_NS:
        raise ValueError(f"end_pos is more than 80 ms from the head frame at {head_ns}")
    message = poses.get(end_ns)
    if message is None:
        raise ValueError(f"aligned end_pos timestamp {end_ns} is missing")
    return message

def eef_for_rows(state, action, messages, end_frames):
    """Logged state_eef and forward-kinematics action_eef, both (T, 16)."""
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    state_eef = np.empty((len(state), EEF_DIM), dtype=np.float64)
    recorded_rot = np.empty((len(state), 2, 3, 3), dtype=np.float64)
    for row, message in enumerate(messages):
        for hand, arm, gripper_index in ((0, "left", 7), (1, "right", 15)):
            position, rotation = recorded_pose(message, arm)
            recorded_rot[row, hand] = rotation
            state_eef[row, hand * 8:(hand + 1) * 8] = pack(position, rotation, state[row, gripper_index])
    positions, rotations = end_frames(action)
    action_eef = np.empty_like(state_eef)
    for hand, gripper_index in ((0, 7), (1, 15)):
        for row in range(len(action)):
            action_eef[row, hand * 8:(hand + 1) * 8] = pack(
                positions[row, hand], rotations[row, hand], action[row, gripper_index])
    for values in (state_eef, action_eef):
        for hand in (0, 1):
            values[:, hand * 8 + 3:hand * 8 + 7] = continuous_quaternions(values[:, hand * 8 + 3:hand * 8 + 7])
    if not np.isfinite(state_eef).all() or not np.isfinite(action_eef).all():
        raise ValueError("non-finite end-effector values")
    state_positions, state_rotations = end_frames(state)
    recorded_positions = np.stack([state_eef[:, 0:3], state_eef[:, 8:11]], axis=1)
    position_mm = np.linalg.norm(recorded_positions - state_positions, axis=-1) * 1000.0
    orientation_deg = geodesic_deg(recorded_rot, state_rotations)
    return state_eef, action_eef, position_mm, orientation_deg

def _saved_rows_match(saved_rows, source_rows):
    """Resume accepts the original rows or the ordered subset left by the outlier filter."""
    saved_action=np.asarray(saved_rows['action'],dtype=np.float64)
    source_action=np.asarray(source_rows['action'],dtype=np.float64)
    if len(saved_action)>len(source_action): return False
    chosen=[]; cursor=0
    for row in saved_action:
        while cursor<len(source_action) and not np.allclose(source_action[cursor],row,atol=1e-5,rtol=0):
            cursor+=1
        if cursor==len(source_action): return False
        chosen.append(cursor); cursor+=1
    for key,source in source_rows.items():
        if not np.allclose(np.asarray(saved_rows[key],dtype=np.float64),np.asarray(source,dtype=np.float64)[chosen],atol=1e-5,rtol=0):
            return False
    return True

def quantile_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {name: arr.tolist() for name, arr in {
        "mean": values.mean(0), "std": values.std(0),
        "q01": np.quantile(values, 0.01, axis=0), "q99": np.quantile(values, 0.99, axis=0),
    }.items()}

def prepare(source,output,resume=False,*,prompt=PROMPT,train_fraction=.8,norm_reference=None,split_note=None):
    if not 0 <= train_fraction <= 1: raise ValueError('train_fraction must be in [0,1]')
    output.mkdir(parents=True,exist_ok=resume)
    (output/'meta').mkdir(exist_ok=resume)
    inventory=json.loads((source/'inventory.json').read_text())
    inventory.sort(key=lambda r:r['metadata']['startAt'])
    train_sources={r['episode'] for r in inventory[:int(train_fraction*len(inventory))]}
    features={k:{'dtype':'float32','shape':[24],'names':None} for k in ('observation.state','action')}
    features.update({k:{'dtype':'float32','shape':[EEF_DIM],'names':None} for k in ('state_eef','action_eef')})
    for cam,shape in [('head',[3,720,960]),('left',[3,540,960]),('right',[3,540,960])]:
        features['observation.images.'+cam]={'dtype':'image','shape':shape,'names':['channels','height','width']}
    for k in ('timestamp','frame_index','episode_index','index','task_index'):
        features[k]={'dtype':'float32' if k=='timestamp' else 'int64','shape':[1],'names':None}
    def feature(spec):
        if spec['dtype']=='image': return datasets.Image()
        if spec['dtype']=='float32' and spec['shape']!=[1]:
            return datasets.Sequence(datasets.Value('float32'),length=spec['shape'][0])
        return datasets.Value(spec['dtype'])
    hf=datasets.Features({k:feature(v) for k,v in features.items()})
    parser,kinematics=load_eef_modules()
    episodes=[]; stats_rows=[]; mapping=[]; train=[]; val=[]; offset=0
    task_list=[]
    def task_id(text):
        if text not in task_list: task_list.append(text)
        return task_list.index(text)
    position_errors=[]; orientation_errors=[]
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
        poses=load_end_poses(parser,bag_path(source,rec))
        bounds=np.flatnonzero(np.diff(np.r_[False,good,False])).reshape(-1,2)
        for start,end in bounds:
            if end-start<25: short+=int(end-start); continue
            selection=indices[start:end]; n=len(selection); ep=len(episodes)
            text=rec.get('prompt') or prompt; tid=task_id(text)
            messages=[end_message(alignment[int(i)],poses) for i in selection]
            state_eef,action_eef,pos_mm,ang_deg=eef_for_rows(q[selection],a[selection],messages,kinematics)
            position_errors.append(pos_mm); orientation_errors.append(ang_deg)
            values={'observation.state':q[selection].tolist(),'action':a[selection].tolist(),
                    'state_eef':state_eef.astype(np.float32).tolist(),'action_eef':action_eef.astype(np.float32).tolist(),
                    'timestamp':(np.arange(n,dtype=np.float32)/15).tolist(),'frame_index':list(range(n)),
                    'episode_index':[ep]*n,'index':list(range(offset,offset+n)),'task_index':[tid]*n}
            dest=output/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet'; dest.parent.mkdir(parents=True,exist_ok=True)
            if not (resume and dest.exists()):
                for cam in ('head','left','right'):
                    paths=[path/alignment[int(i)]['images'][cam] for i in selection]
                    if not all(p.is_file() for p in paths): raise FileNotFoundError(eid+' '+cam)
                    values['observation.images.'+cam]=[{'bytes':None,'path':str(p.resolve())} for p in paths]
                datasets.Dataset.from_dict(values,features=hf).to_parquet(str(dest))
            else:
                import pyarrow.parquet as pq
                saved=pq.read_table(dest,columns=['observation.state','action','state_eef','action_eef'])
                source_rows={'observation.state':q[selection],'action':a[selection],'state_eef':state_eef,'action_eef':action_eef}
                saved_rows={key:np.asarray(saved.column(key).to_pylist(),dtype=np.float64) for key in source_rows}
                if not _saved_rows_match(saved_rows, source_rows): raise ValueError('resume content mismatch: '+str(dest))
            episodes.append({'episode_index':ep,'tasks':[text],'length':n})
            sr={}
            for k,v in [('observation.state',q[selection]),('action',a[selection]),('state_eef',state_eef),('action_eef',action_eef)]:
                sr[k]={name:arr.tolist() for name,arr in {'min':v.min(0),'max':v.max(0),'mean':v.mean(0),'std':v.std(0),'count':np.array([n])}.items()}
            stats_rows.append({'episode_index':ep,'stats':sr})
            is_train=eid in train_sources; (train if is_train else val).append(ep)
            mapping.append({'episode_index':ep,'source_id':eid,'archive_member':rec['archive_member'],
                            'split':'train' if is_train else 'validation','grid_start_ns':int(grid[start]),
                            'source_row_indices':selection.tolist(),'length':n})
            offset+=n
        print(f'{eid}: segments={len(episodes)} frames={offset}',flush=True)
    for name,values in [('episodes',episodes),('episodes_stats',stats_rows),
                        ('tasks',[{'task_index':i,'task':text} for i,text in enumerate(task_list)])]:
        (output/f'meta/{name}.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))
    write(output/'meta/info.json',{'codebase_version':'v2.1','robot_type':'linglong_h',
        'total_episodes':len(episodes),'total_frames':offset,'total_tasks':len(task_list),'total_videos':0,
        'total_chunks':(len(episodes)+999)//1000,'chunks_size':1000,'fps':15,
        'splits':{'train':f'0:{len(episodes)}'},'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
        'video_path':None,'features':features})
    write(output/'train_episodes.json',train); write(output/'validation_episodes.json',val)
    write(output/'source_mapping.json',mapping)
    reference_stats=None
    if norm_reference is not None:
        reference=Path(norm_reference)
        copied=json.loads(reference.read_text())['norm_stats']
        eef_reference=reference.parent/'eef_norm_stats.json'
        if not eef_reference.is_file(): raise ValueError(f'missing {eef_reference}')
        copied_eef=json.loads(eef_reference.read_text())['norm_stats']
        reference_stats={**copied,**copied_eef}
    outlier_report=filter_dataset(output, train, reference_stats=reference_stats)
    norm={}; eef_norm={}; stat_frames=0; qa={}
    if position_errors:
        position_errors=np.concatenate(position_errors); orientation_errors=np.concatenate(orientation_errors)
        if np.median(position_errors)>10.0:
            raise ValueError('recorded end position disagrees with end_frames by more than 10 mm median')
        if np.median(orientation_errors)>5.0:
            raise ValueError('recorded end orientation disagrees with wrist_yaw end_frames by more than 5 deg median')
        qa={'left_position_mm':float(np.median(position_errors[:,0])),'right_position_mm':float(np.median(position_errors[:,1])),
            'left_orientation_deg':float(np.median(orientation_errors[:,0])),'right_orientation_deg':float(np.median(orientation_errors[:,1]))}
    if norm_reference is not None:
        norm=copied; eef_norm=copied_eef
    else:
        columns=outlier_report['train_columns']
        if len(columns['action'])==0: raise ValueError('No training frames for normalization')
        stat_frames=len(columns['action'])
        norm={'state':quantile_stats(columns['observation.state']),'actions':quantile_stats(columns['action'])}
        eef_norm={'state_eef':quantile_stats(columns['state_eef']),'action_eef':quantile_stats(columns['action_eef'])}
    write(output/'assets/linglong_h/norm_stats.json',{'norm_stats':norm})
    write(output/'assets/linglong_h/eef_norm_stats.json',{'norm_stats':eef_norm})
    write(output/'manifest.json',{'status':'complete','prompt':task_list[0] if len(task_list)==1 else task_list,
        'layout':JOINT_LAYOUT,'eef_layout':EEF_LAYOUT,'source_root':str(source.resolve()),
        'source_inventory_sha256':hashlib.sha256((source/'inventory.json').read_bytes()).hexdigest(),
        'source_episodes':len(inventory),'segments':len(episodes),'train_segments':len(train),'validation_segments':len(val),
        'frames':outlier_report['frames'],'train_stat_frames':stat_frames,'norm_reference':str(norm_reference) if norm_reference else None,'rejected_grid_rows':rejected,'discarded_short_segment_rows':short,
        'quantile_outliers':{k:v for k,v in outlier_report.items() if k!='train_columns'},
        'resampling':'15Hz nearest head <=20ms; all streams absolute age <=80ms; reject whole source if any invalid grid row, original head gap >50ms or original aligned age >80ms',
        'image_storage':'absolute original JPEG references; source must remain accessible',
        'action_semantics':'absolute q_exp','state_eef':'logged /end_pos xyz and wrist orientation, gripper from measured joints 7 and 15',
        'action_eef':'end_frames(action) wrist_yaw_link pose plus wrist-frame TCP, gripper from commanded joints 7 and 15',
        'recorded_vs_end_frames':qa,'success_filter':'none; success labels unavailable',
        'split':split_note or f'chronological source episodes train fraction {train_fraction}; train/validation JSON overrides meta full coverage'})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Reuse verified existing Parquet files with unchanged source and conversion settings')
    args=p.parse_args(); prepare(args.source,args.output,args.resume)

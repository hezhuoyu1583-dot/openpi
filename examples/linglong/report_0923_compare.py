"""Plot training loss and summarize held-out action errors (not success rate)."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'data/training_logs/linglong/linglong0923_b64_2000'
OUT = ROOT / 'data/eval/linglong0923_compare_batched'

def main():
    rows = list(csv.DictReader((LOG / 'metrics.csv').open()))
    run = rows[-1]['run_id']
    rows = [r for r in rows if r['run_id'] == run]
    x = np.array([int(r['step']) for r in rows])
    y = np.array([float(r['loss']) for r in rows])
    sm = np.array([y[(x > t-200) & (x <= t)].mean() for t in x])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax in axes:
        ax.plot(x, y, color='#9ebbd8', label='Logged training loss')
        ax.plot(x, sm, color='#15548b', label='200-step mean (past logging points)')
        ax.set(xlabel='Training step (zero-based log)', ylabel='Flow-matching loss')
        ax.grid(alpha=.2)
        for s in (500, 1000, 1500, 2000): ax.axvline(s, alpha=.15, color='gray')
    axes[0].legend(fontsize=8)
    axes[1].set(xlim=(1000, 2000), ylim=(0, max(y[x >= 1000])*1.15))
    fig.suptitle('Linglong 0923 | pi0.5 base SFT | batch 64 | 2,000 updates')
    fig.tight_layout()
    fig.savefig(LOG/'loss_completed_run.png', dpi=180)
    plt.close(fig)
    blocks = {f'{s}-{s+199}': float(y[(x>=s)&(x<s+200)].mean()) for s in range(0,2000,200)}
    (LOG/'loss_summary.json').write_text(json.dumps({'run_id': run, 'first_logged_loss': float(y[0]),
        'last_logged_loss': float(y[-1]), '200_step_blocks': blocks,
        'caveat': 'Rank-0 logged training averages; not held-out loss or all-rank average.'}, indent=2))
    print(json.dumps(blocks, indent=2))
    if not (OUT/'summary.json').exists(): return
    summary = json.loads((OUT/'summary.json').read_text())
    mapping=json.loads((ROOT/'data/linglong_0923_sft_test_15hz/source_mapping.json').read_text())
    per_episode={}
    for name in summary['models']:
        arrays=np.load(OUT/f'{name}.npz')
        error=np.abs(arrays['predictions']-arrays['targets'])
        offset=0; entries=[]
        for rec in mapping:
            n=rec['length']-24
            e=error[offset:offset+n]
            entries.append({'source_id':rec['source_id'],'windows':n,
                'right_arm_mae':float(e[:,:,8:15].mean()),
                'right_gripper_mae':float(e[:,:,15].mean()),
                'right_arm_first_step_mae':float(e[:,0,8:15].mean())})
            offset+=n
        assert offset==len(error)
        per_episode[name]={'episodes':entries,'right_arm_by_horizon':error[:,:,8:15].mean(axis=(0,2)).tolist(),
            'right_gripper_by_horizon':error[:,:,15].mean(axis=0).tolist()}
    (OUT/'per_episode.json').write_text(json.dumps(per_episode,indent=2))
    groups = ['left_arm','left_gripper','right_arm','right_gripper','waist','head']
    metrics = {k:v['mae_by_group'] for k,v in summary['models'].items()}
    metrics['hold_position'] = next(iter(summary['models'].values()))['hold_baseline_mae']
    with (OUT/'comparison.csv').open('w') as f:
        writer=csv.writer(f); writer.writerow(['model',*groups])
        writer.writerows([k,*[v[g] for g in groups]] for k,v in metrics.items())
    fig, axes = plt.subplots(2,3,figsize=(15,8))
    for ax,g in zip(axes.flat,groups):
        ax.bar(range(len(metrics)),[v[g] for v in metrics.values()])
        ax.set_xticks(range(len(metrics)),list(metrics),rotation=65,ha='right',fontsize=7)
        ax.set_title(g); ax.set_ylabel('MAE (gripper raw units)' if 'gripper' in g else 'MAE (rad)')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('0923 held-out | all full 25-step windows | lower MAE is better')
    fig.tight_layout(); fig.savefig(OUT/'comparison.png',dpi=180); plt.close(fig)
    lines=['# Linglong 0923 SFT evaluation', '',
        'Training: 98 episodes / 17,339 frames; held-out: 10 episodes / 2,189 frames.',
        'Evaluation: all 1,949 full 25-step windows, 15 Hz, seed 42; same observations, task prompt and noise for all models.',
        'Base uses 0923 train-only normalization (not a native zero-shot Linglong policy). Each SFT uses its own checkpoint normalization.',
        'Previous 0714 SFT is evaluated for cross-task transfer, not its original task performance.',
        'Overlapping windows are correlated. This is action imitation MAE, not closed-loop task success.', '',
        '| Model | '+' | '.join(groups)+' |', '|---|'+'---:|'*len(groups)]
    lines += ['| '+k+' | '+' | '.join(f'{v[g]:.6f}' for g in groups)+' |' for k,v in metrics.items()]
    lines += ['', 'Arms/waist/head: rad; grippers: original normalized opening units. Base placeholders and padding excluded.',
              '', 'Training loss 200-step block means:', '```json',json.dumps(blocks,indent=2),'```',
              '', 'Loss logs are rank-0 interval averages, not validation loss or all-rank averages.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': main()

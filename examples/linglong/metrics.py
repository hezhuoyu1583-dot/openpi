"""Persist existing trainer log metrics without changing its loss computation."""
import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import uuid

FIELDS = ['run_id', 'timestamp_utc', 'step', 'loss', 'learning_rate', 'grad_norm', 'interval_seconds']
PATTERN = re.compile(r'^step=(\d+) loss=(\S+) lr=(\S+)(?: grad_norm=(\S+))? time=(\S+)s$')

class MetricsHandler(logging.Handler):
    def __init__(self, directory):
        super().__init__()
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
        self.run_id=uuid.uuid4().hex
        self.json_file=(directory/'metrics.jsonl').open('a',buffering=1)
        csv_path=directory/'metrics.csv'
        header=not csv_path.exists() or csv_path.stat().st_size==0
        self.csv_file=csv_path.open('a',newline='',buffering=1)
        self.writer=csv.DictWriter(self.csv_file,fieldnames=FIELDS)
        if header: self.writer.writeheader()

    def emit(self,record):
        match=PATTERN.fullmatch(record.getMessage())
        if not match: return
        try:
            step,loss,lr,grad,seconds=match.groups()
            row=dict(zip(FIELDS,[self.run_id,datetime.now(timezone.utc).isoformat(),int(step),
                float(loss),float(lr),float(grad) if grad else None,float(seconds)]))
            self.json_file.write(json.dumps(row)+'\n'); self.writer.writerow(row)
            self.json_file.flush(); self.csv_file.flush()
        except Exception:
            self.handleError(record)

    def close(self):
        self.json_file.close(); self.csv_file.close(); super().close()

def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory)
    path=directory/'metrics.jsonl'
    if not path.exists(): return
    runs={}
    for line in path.read_text().splitlines():
        try: row=json.loads(line)
        except json.JSONDecodeError: continue  # interrupted last write
        runs.setdefault(row['run_id'],[]).append(row)
    if not runs: return
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for ax,key in zip(axes.flat,('loss','learning_rate','grad_norm','interval_seconds')):
        for run,rows in runs.items():
            ax.plot([r['step'] for r in rows],[r[key] for r in rows],label=run[:8])
        ax.set(xlabel='Trainer step (zero-based log index)',ylabel=key)
        ax.grid(alpha=.3)
    axes[0,0].legend(title='Process run'); fig.tight_layout()
    fig.savefig(directory/'training_curves.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    plot(parser.parse_args().directory)

import importlib.util
import json
import logging
from pathlib import Path

spec=importlib.util.spec_from_file_location('linglong_metrics',Path(__file__).with_name('metrics.py'))
metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)

def test_append_resume_and_plot(tmp_path):
    for message in ('step=20 loss=0.1234 lr=1.00e-05 grad_norm=2.30 time=40.0s',
                    'step=20 loss=0.1000 lr=1.00e-05 time=1.0s'):
        handler=metrics.MetricsHandler(tmp_path)
        handler.emit(logging.LogRecord('test',logging.INFO,'',0,'unrelated log',(),None))
        handler.emit(logging.LogRecord('test',logging.INFO,'',0,message,(),None))
        handler.close()
    rows=[json.loads(s) for s in (tmp_path/'metrics.jsonl').read_text().splitlines()]
    assert len(rows)==2 and rows[0]['run_id']!=rows[1]['run_id']
    assert rows[0]['loss']==.1234 and rows[1]['grad_norm'] is None
    assert len((tmp_path/'metrics.csv').read_text().splitlines())==3
    metrics.plot(tmp_path)
    assert (tmp_path/'training_curves.png').stat().st_size>1000

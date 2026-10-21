#!/usr/bin/env python3
import argparse, datetime, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--directory',required=True)
p.add_argument('--kernel',required=True)
p.add_argument('--seconds',type=float,default=0)
p.add_argument('--status',required=True)
p.add_argument('--note',default='')
a=p.parse_args()
f=Path(a.directory)/'trial_log.jsonl'
f.parent.mkdir(parents=True,exist_ok=True)
trial=sum(1 for _ in f.open())+1 if f.exists() else 1
with f.open('a') as out:
 out.write(json.dumps({'ts':datetime.datetime.now().isoformat(timespec='seconds'),'kernel':a.kernel,'trial':trial,'jit_s':a.seconds,'gpu_us':0,'verify':a.status,'note':a.note})+'\n')

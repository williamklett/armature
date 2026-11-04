"""Authenticated observe/action client for a local assistant; no key is printed."""
import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['observe','start','stop','move','jog','note','connect','disconnect'])
    p.add_argument('--mode', choices=['sandbox','so101'], default='sandbox')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--state-dir', type=Path, default=ROOT/'.armature')
    p.add_argument('--experiment')
    p.add_argument('--revision', type=int)
    p.add_argument('--joint')
    p.add_argument('--delta', type=float)
    p.add_argument('--dx', type=float)
    p.add_argument('--dy', type=float)
    p.add_argument('--note', default='')
    args = p.parse_args()
    family='arm' if args.mode=='so101' else 'robot'
    path='observe' if args.action=='observe' else 'action'
    data=None if path=='observe' else {k:v for k,v in {'action':args.action,'experiment_id':args.experiment,'revision':args.revision,'joint':args.joint,'delta':args.delta,'dx':args.dx,'dy':args.dy,'note':args.note}.items() if v is not None}
    token=(args.state_dir/'bridge-token').read_text().strip()
    req=urllib.request.Request('http://127.0.0.1:%d/api/%s/%s'%(args.port,family,path),data=None if data is None else json.dumps(data).encode(),headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=10) as response:
        print(json.dumps(json.load(response),indent=2))


if __name__=='__main__': main()

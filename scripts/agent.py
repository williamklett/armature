"""Authenticated observe/action client for a local assistant; no key is printed."""
import argparse
import base64
import json
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['observe','start','stop','move','jog','note','connect','disconnect','camera','plan','execute','clear_fault'])
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
    p.add_argument('--output', type=Path, help='Save the current camera image here (camera action only)')
    p.add_argument('--waypoints', type=Path, help='JSON array of joint waypoints for plan')
    p.add_argument('--plan-id', help='Previously reviewed, unexpired motion plan')
    p.add_argument('--speed', type=float)
    p.add_argument('--acceleration', type=float)
    args = p.parse_args()
    if args.action == 'camera' and not args.output:
        p.error('camera requires --output so image bytes are not printed')
    if args.action in ('plan', 'execute') and args.mode != 'so101':
        p.error('plan and execute require --mode so101')
    if args.action == 'plan' and not args.waypoints:
        p.error('plan requires --waypoints')
    if args.action == 'execute' and not args.plan_id:
        p.error('execute requires --plan-id')
    family='arm' if args.mode=='so101' else 'robot'
    path='observe' if args.action=='observe' else 'action'
    data=None if path=='observe' else {k:v for k,v in {'action':args.action,'experiment_id':args.experiment,'revision':args.revision,'joint':args.joint,'delta':args.delta,'dx':args.dx,'dy':args.dy,'note':args.note}.items() if v is not None}
    if args.action == 'plan':
        data['waypoints'] = json.loads(args.waypoints.read_text())
        for key in ('speed', 'acceleration'):
            if getattr(args, key) is not None:
                data[key] = getattr(args, key)
    if args.action == 'execute':
        data['plan_id'] = args.plan_id
    if args.action == 'camera':
        family, path, data = 'camera', 'frame', None
    token=(args.state_dir/'bridge-token').read_text().strip()
    req=urllib.request.Request('http://127.0.0.1:%d/api/%s/%s'%(args.port,family,path),data=None if data is None else json.dumps(data).encode(),headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=30 if args.action == 'connect' else 10) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        print(error.read().decode())
        raise SystemExit(1)
    if args.action == 'camera':
        url = result.pop('data_url')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(base64.b64decode(url.split(',', 1)[1], validate=True))
        result['image_path'] = str(args.output.resolve())
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()

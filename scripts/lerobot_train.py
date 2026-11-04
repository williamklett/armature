"""LeRobot SFT runner. Launch only inside the prepared training environment."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def command(config, output, device):
    if config['recipe'] not in ('smolvla_sft', 'pi05_sft'):
        raise ValueError('Use an SFT recipe')
    if not config.get('dataset'):
        raise ValueError('A LeRobot dataset is required')
    checkpoint = config.get('checkpoint') or ('lerobot/smolvla_base' if config['recipe'] == 'smolvla_sft' else 'lerobot/pi05_base')
    return [sys.executable, '-m', 'lerobot.scripts.lerobot_train',
            '--policy.path=' + checkpoint, '--dataset.repo_id=' + config['dataset'],
            '--policy.device=' + device, '--output_dir=' + str(output),
            '--steps=' + str(config['steps']), '--batch_size=' + str(config['batch_size']),
            '--seed=' + str(config['seed']), '--policy.optimizer_lr=' + str(config['learning_rate']),
            '--policy.push_to_hub=false', '--trackio.enable=false', '--log_freq=10']


def metric_from_log(line):
    # LeRobot's human-readable AverageMeter output. Raw logs are also retained.
    clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
    step = re.search(r'\bstep:\s*([\d.]+)([KMG]?)\b', clean)
    loss = re.search(r'\bloss:\s*([-+\d.eE]+)', clean)
    if step and loss:
        return {'step': float(step[1]) * {'': 1, 'K': 1000, 'M': 1000000, 'G': 1000000000}[step[2]], 'loss': float(loss[1])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default='cuda')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    cmd = command(config, args.output, args.device)
    print(json.dumps({'backend': 'lerobot', 'argv': cmd}), flush=True)
    if args.dry_run:
        return
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
        try:
            metric = metric_from_log(line)
            if metric:
                print('ARMATURE_METRIC ' + json.dumps(metric, allow_nan=False), flush=True)
        except (ValueError, OverflowError):
            pass
    process.stdout.close()
    raise SystemExit(process.wait())


if __name__ == '__main__':
    main()

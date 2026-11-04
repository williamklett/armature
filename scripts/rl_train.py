"""Simulation-RL adapters. Each backend runs in its own prepared environment."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def build(config, root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    checkpoint = config.get('checkpoint')
    if not checkpoint:
        raise ValueError('A matching simulation SFT checkpoint is required')
    env = dict(os.environ)
    env['MUJOCO_GL'] = env.get('MUJOCO_GL', 'egl')
    env['PYOPENGL_PLATFORM'] = env.get('PYOPENGL_PLATFORM', 'egl')
    if config['recipe'] == 'pi05_rl':
        embodied = root / 'examples/embodiment'
        env.update(EMBODIED_PATH=str(embodied), ROBOT_PLATFORM='LIBERO')
        env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
        # JSON quoting is for Hydra's override grammar; this argv never goes through a shell.
        cmd = [sys.executable, str(embodied / 'train_embodied_agent.py'), '--config-path', str(embodied / 'config'),
               '--config-name', 'libero_spatial_ppo_openpi_pi05',
               'actor.model.model_path=' + json.dumps(checkpoint), 'rollout.model.model_path=' + json.dumps(checkpoint),
               'runner.logger.log_path=' + json.dumps(str(output)),
               'runner.max_steps=' + str(config['steps']), 'actor.seed=' + str(config['seed']),
               'actor.micro_batch_size=' + str(config['batch_size']),
               'actor.global_batch_size=' + str(config['batch_size'] * 4),
               'actor.optim.lr=' + str(config['learning_rate'])]
    elif config['recipe'] == 'smolvla_rl':
        if config['batch_size'] < 2:
            raise ValueError('GRPO group size must be at least two')
        env['PYTHONPATH'] = str(root / 'src') + os.pathsep + env.get('PYTHONPATH', '')
        cmd = [sys.executable, str(root / 'scripts/train_grpo_libero.py'), '--suite', 'libero_object',
               '--checkpoint', checkpoint, '--groups', str(config['steps']), '--group-size', str(config['batch_size']),
               '--seed', str(config['seed']), '--lr', str(config['learning_rate']), '--run-root', str(output)]
    else:
        raise ValueError('Use a simulation-RL recipe')
    return cmd, env


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    cmd, env = build(json.loads(Path(args.config).read_text()), args.root, args.output)
    print(json.dumps({'backend_argv': cmd}), flush=True)
    if args.dry_run:
        return
    if not Path(cmd[1]).is_file():
        raise SystemExit('Backend checkout is missing: ' + cmd[1])
    # exec preserves exit status and the bridge process-group cancellation behavior.
    os.chdir(args.root)
    os.execve(sys.executable, cmd, env)


if __name__ == '__main__':
    main()

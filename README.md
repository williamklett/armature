# Armature

A small experiment workspace for SO-ARM101 research. Static dashboard on GitHub Pages; authenticated Python bridge on the laptop. The interface keeps only labels, controls, statuses, and experiment data.

## Run locally

Python 3.9+ is sufficient for the dashboard, storage, toy learner, and optional API grader. No JavaScript dependencies or build step.

```bash
python3 -B -m bridge.server --port 8787
```

Open `http://127.0.0.1:8787`. The locally served dashboard pairs automatically. Data is stored under `.armature/`, which is excluded from Git.

Create a **Q-learning · test** experiment and start it to exercise actual process execution, metrics, logs, cancellation, and persistent run history. This is a CPU pipeline check, not VLA training or robot physics.

## Camera

Control room → **Enable camera** requests camera permission only after the click. Choose the laptop camera or a connected UVC device, capture up to eight frames, select an experiment, and save the episode. Audio is never requested. **Turn off**, page navigation, hiding the tab, and closing the page release camera tracks. Pending permission results are also stopped if the user has already left.

A live preview stays on the laptop. Saving an episode sends its captured frames to the selected bridge, or stores them in the browser when disconnected. **Grade with OpenAI** is a separate upload action. Camera frames are included in backup exports. Browser permission, physical UVC availability, and supported modes must be checked on the actual device.

**Agent view** explicitly shares fresh preview frames with the authenticated bridge at up to two frames per second. Only the latest frame is held in memory; it is unavailable after three seconds, and stopping sharing clears it. Live frames are not saved as episodes or included in backups. Turning off the camera, changing its device, leaving Control room, hiding the tab, or disconnecting the bridge ends sharing. Frozen video frames and failed uploads also stop sharing. Clients with the bridge token can retrieve the shared image; opening the dashboard alone does not start sharing.

An assistant can retrieve an image for visual inspection without a separate paid grading request:

```bash
python3 scripts/agent.py camera --port 8787 --output .armature/current-view.jpg
```

This explicitly saves the retrieved frame locally. For the physical controller, set `robot.require_camera` to `true` (as in the example). Connecting, starting, and jogging then require a fresh shared frame. Losing the feed requests a hold and stops the session. This checks feed freshness; it does not detect collisions, prove that an assistant inspected the image, or replace calibration and joint limits. Keep the whole arm, base, gripper, and nearby workspace in view.

## GitHub Pages and roommates

The workflow in `.github/workflows/pages.yml` runs tests and deploys only `web/`. Set repository **Settings → Pages → Source: GitHub Actions**. The dashboard works without a bridge using IndexedDB; each browser has its own library. Export/import transfers browser libraries.

For the hosted dashboard, copy `configs/bridge.example.json` to `bridge.local.json`, retain only the runners you intend to use, and replace the paths. Set `allowed_origins` to the exact website origin. Restart the bridge, copy the pairing token from the local dashboard's Connections screen, and connect from the hosted page. Tokens are kept in session storage, not URLs. Some browsers restrict hosted-site access to localhost; use the local dashboard if local-network permission is unavailable.

For a shared library, teammates must connect to the same bridge. The bridge binds only to `127.0.0.1`. A teammate with SSH access to the laptop can forward its port:

```bash
ssh -N -L 8787:127.0.0.1:8787 USER@LAPTOP
```

Then open `http://127.0.0.1:8787` on their computer. This is a small trusted-lab deployment: one pairing token grants access to all experiments, saved images, configured jobs, and enabled arm controls. It has no individual user accounts or role permissions.

## Model training

Model choice: π0.5 is the primary RL research candidate; SmolVLA is the lightweight baseline. See [RESEARCH.md](RESEARCH.md) for evidence and limitations.

### Demonstration training

On the training machine, with [uv](https://docs.astral.sh/uv/) installed:

```bash
python3 scripts/setup_training.py smolvla
python3 scripts/setup_training.py pi05
```

These create separate Python 3.12 environments using LeRobot revision `9e5ee9859480e425a97a6836bb3fee479539beb5`. They install dependencies, not checkpoints. Training downloads the selected checkpoint on first launch. Model licenses and any gated Hub access still apply. The setup commands have a `--dry-run` option.

On a quota-limited machine, install one environment at a time. `--budget-root /path/to/armature --max-disk-gib 8` stops installation when that directory reaches the allocation. Put `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` inside that directory so downloads and Python count toward the same limit. Check the cluster's actual quota first and leave additional headroom for in-flight writes; this guard is not a filesystem quota.

Configure the corresponding runner paths in `bridge.local.json`. Choose `cuda` for NVIDIA training; `mps` is an option for SmolVLA on an appropriate Mac, subject to backend compatibility and measured latency. Record demonstrations with [LeRobot](https://huggingface.co/docs/lerobot/so101) before starting SFT. The dataset must have matching camera names, action units, frame rate, and calibration. Armature's sparse camera review frames are **not** a LeRobot training dataset.

The SFT adapter passes the selected dataset, checkpoint, step count, batch size, seed, and optimizer learning rate. It retains raw logs and translates LeRobot's logged loss into chart metrics. Each run stores its configuration and resolved argv under `.armature/runs/<id>/`. Saved configurations are immutable; clone one to change it.

An SFT experiment also exports a Slurm job. Set `ARMATURE_VENV` and `ARMATURE_OUTPUT_ROOT` before `sbatch`, and adjust the partition/account/GPU/memory/time directives to the actual allocation. The example's single-GPU request is not a claim that π0.5 fits any GPU. Exported cluster jobs are not automatically monitored or cancelled by the local bridge.

### Launch on a cluster from the dashboard

Copy `configs/slurm.example.json` to the ignored `slurm.local.json`. Configure the SSH destination (or existing control socket), writable remote directory, Slurm resources, and each environment's Python path. The examples need adapting to your cluster. [Adroit's documentation](https://researchcomputing.princeton.edu/systems/adroit) describes its GPU constraints and scratch storage.

Register this command for each recipe in `bridge.local.json`, using absolute local paths:

```json
{
  "cwd": "/path/to/armature",
  "stop_timeout": 60,
  "command": ["/path/to/python3", "-B", "/path/to/armature/scripts/slurm_train.py", "--profile", "/path/to/armature/slurm.local.json", "--config", "{config_path}", "--output", "{output_dir}", "--run-id", "{run_id}"]
}
```

Set `enabled: false` and a `reason` while preparing an environment. The profile is a trusted operator file; the web client cannot change it. Experiment values travel through JSON over SSH, not through shell interpolation.

The runner stages an immutable configuration and adapter source, submits one Slurm job, saves its job ID in the run record and `remote.json`, streams logs/metrics, and reports the scheduler's terminal state. **Stop** requests `scancel`. The laptop must remain awake and the bridge must stay running: a compute-node supervisor terminates training if its lease has not been renewed for 120 seconds. A queued job whose lease expired exits before starting training. An SSH outage can prevent immediate cancellation; inspect Slurm before relaunching after an uncertain failure. Checkpoints remain in the remote run directory; they are not copied to the browser.

For a CPU-only integration check, run the adapter with `--smoke`, a toy experiment JSON, a fresh 32-character lowercase hexadecimal `--run-id`, and `smoke_resources` suited to the cluster. This exercises actual submission and log streaming without downloading a VLA or using a GPU.

### Simulation RL

Both RL recipes currently target LIBERO, using simulator rewards. Their environments are separate from SFT:

- **π0.5 PPO:** [RLinf instructions](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html), reviewed revision `77a521298f1d215971246c2d7a423720d9e76652`. Prepare its `openpi`/LIBERO environment and matching `RLinf-Pi05-LIBERO-SFT` checkpoint. The adapter invokes `train_embodied_agent.py` directly so its exit status reaches Armature. Batch size maps to actor microbatch size; global batch size is four times that value. Steps map to `runner.max_steps`.
- **SmolVLA GRPO:** experimental [FlowVLA-RL](https://github.com/BlackMirean/FlowVLA-RL), reviewed revision `50e73afc07212f48bed370ea4c88a3131d6f914a`. Follow its pinned environment instructions (including LeRobot v0.6.0) and use a compatible LIBERO SFT checkpoint. Batch size maps to GRPO group size, steps to completed updates, with the backend's default task rotation on LIBERO-Object.

Point the runner commands at those prepared Python environments and checkouts. `scripts/rl_train.py --dry-run` prints the resolved argv without starting training. Raw RL logs are available in Armature and upstream output directories; charts require `ARMATURE_METRIC` output as described below. The adapters have not yet been exercised on the user's GPU allocation.

**Physical SO101 online VLA RL and LLM rewards inside the training loop still require a task adapter, synchronized trajectories, a reset protocol, and a validated reward integration.** Saving a grade does not update model weights. RLinf's SO101 guide currently provides hardware checks, not a supported real-world training workflow. This MVP supplies tracking, model launch adapters, grading, and bounded manual/agent control without claiming that these separate pieces already form a validated physical RL algorithm.

## Vision-language grading

Export `OPENAI_API_KEY` and `ARMATURE_GRADER_MODEL` in the bridge's terminal, using a vision-capable model that supports structured output. Restart the bridge. `.env.example` lists the names; environment files are not automatically loaded.

Episode review → **Grade with OpenAI** sends the saved frames, task, and rubric to the Responses API. Responses are requested with `store: false`. API charges may apply. The saved result includes the model identifier, response ID, rubric and frame hashes, evidence, score, and estimated confidence. Unclear outcomes have no reward score. Refused, incomplete, malformed, and failed responses do not create a grade. Human labels remain independent.

This adapter is tested with mocked API responses. No real camera frame or paid API request has been sent during development. Validate grader agreement against held-out human labels before using scores for learning.

## SO101 and assistant control

Physical control is opt-in and disabled by default. No motor connection is opened when the app or bridge starts.

1. Prepare a LeRobot robot environment with `python3 scripts/setup_training.py robot`.
2. Calibrate the arm with `.venv-robot/bin/python scripts/calibrate_arm.py --port /dev/cu.YOUR_ARM`. This uses LeRobot's interactive calibration with torque off throughout and saves under `.armature/calibration`. Set the serial port, matching calibration ID and directory, and measured joint limits in `bridge.local.json`. Limits are intentionally blank in the example. Body joint units are degrees; the gripper uses 0–100. This adapter targets the pinned `so_follower` API.
3. Run the bridge with that environment's Python. Set `robot.enabled` to true only after local calibration and limit checks.
4. In Control room, **Connect motors** opens the bus, configures the motors, verifies that the measured pose is within limits, and seeds and verifies current-position targets before enabling holding torque. A failed configuration leaves torque off. Select an experiment and start a session before jogging. The UI issues 0.5-unit steps; server and LeRobot impose per-step limits.

The controller rejects stale observations, replayed revisions, excessive steps, out-of-range targets, and commands while stopped. It latches on communication failures and stops after 10 seconds without an action. Software **Stop** requests a hold at the measured position; it is not a hardware emergency stop or a collision detector. Disconnect and normal bridge shutdown retain holding torque; support the arm locally before removing power. Joint limits alone do not establish a collision-free workspace.

Software holds and camera-loss checks require a running bridge. If the laptop loses power, a servo can remain powered at its last target; keep the arm's power disconnect accessible.

The physical adapter is tested against a fake motor bus. Serial communication and a separate supervised, low-torque gripper test have been verified on one SO101; calibration and the calibrated web-control path remain unvalidated on that arm. The separate 2D sandbox needs no hardware.

`scripts/probe_arm.py --port /dev/cu.YOUR_ARM` reads the six expected SO101 motor IDs, raw positions, torque state, and stored calibration registers. Run it with the robot environment's Python. It does not configure the motors, change torque, or send position targets. Stored limits alone do not prove that the arm was properly calibrated or that the current workspace is clear.

An assistant can use the same authenticated APIs through `scripts/agent.py`. It reads the local token file without printing it. For example:

```bash
python3 scripts/agent.py start --port 8787 --experiment EXPERIMENT_ID
python3 scripts/agent.py observe --port 8787
python3 scripts/agent.py move --port 8787 --revision REVISION --dx 0.05 --note 'Approach the target'
python3 scripts/agent.py stop --port 8787
```

For the physical arm use `--mode so101`, `connect`, `start`, `observe`, and `jog --joint shoulder_pan --delta 0.5 --revision REVISION`. Review the camera and joint feedback before each step. Observations/actions/notes persist as session context; this does not modify an LLM's weights. Starting an agent session is not an autonomous background agent.

## Runner contract

The browser can select a registered recipe but cannot submit arbitrary shell commands. `bridge.local.json` is a trusted local operator file. Commands are executed as argv arrays, without a shell. Supported placeholders are experiment fields plus `{run_id}`, `{config_path}`, and `{output_dir}`. Do not wrap untrusted placeholder values in a shell command or naïvely forward them through SSH.

For charts, print one JSON object per line:

```text
ARMATURE_METRIC {"step": 10, "reward": 0.7, "success_rate": 0.75, "loss": 0.03}
```

Only supplied values are plotted. One local runner executes at a time. Stop terminates its local process group; normal bridge shutdown also stops it. On a hard crash, previous jobs are marked interrupted because their external state is unknown. Check GPU/scheduler processes before relaunching. A custom runner that submits detached external work needs its own remote cancellation implementation.

## Verification

```bash
npm test
python3 -B -m unittest discover -s tests -p 'test_*.py' -v
node --check web/app.js
```

Tests cover camera opt-in/lifecycle, config validation, export quoting, backup validation, SQLite persistence, real toy job completion and cancellation, HTTP authentication/CORS/pairing, episode review, structured grading failures, bounded control against a fake motor bus, and Slurm submission idempotence, cancellation, log offsets, and lease expiry. A real CPU Slurm smoke run also completed on Adroit. These checks do not establish GPU training performance, physical motor behavior, or live API grading quality.

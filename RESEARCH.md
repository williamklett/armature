# VLA selection for Armature

Research date: 2026-11-04. Target: SO-ARM101, Innomaker 1080p USB UVC camera, laptop control, Adroit training accessed from the laptop, GitHub Pages dashboard.

This is a model-selection memo. No model has been installed, trained, benchmarked, or connected to the physical arm in this project yet. Adroit access and GPU allocation have not been verified. Dashboard and local-bridge code exists, but its implementation status is separate from model or hardware validation.

## Recommendation

Use **π0.5 as the main VLA RL research candidate**, conditional on a suitable NVIDIA inference/training allocation. Keep **SmolVLA as the lightweight baseline** for getting the laptop/robot/data pipeline working. Neither currently gives this project a validated, ready-to-run SO101 + vision-language grader + online RL system.

The selection weighs evidence on the actual arm, available training code, inference cost, and the remaining integration work. It is not a claim that one model is universally best.

### Which one to start with

- **First working SO101 policy:** SmolVLA, fine-tuned on demonstrations from the actual arm and camera. Its small size and official low-cost-arm workflow reduce the cost of debugging data collection and inference.
- **Main VLA-weight RL experiment:** π0.5, after verifying the GPU allocation and reproducing an RLinf simulation recipe. Its documented RL path is the deciding advantage, not a claim of best performance on every benchmark.
- **If local inference is mandatory and no suitable GPU is available:** retain SmolVLA as the primary policy and treat its RL implementation as an additional research task. Laptop support alone does not establish usable control latency.
- **If two models are affordable:** compare both on exactly the same task, demonstrations, held-out starting conditions, and human-confirmed outcomes before choosing the final policy.

The June 2026 SO101 comparison is useful supporting evidence but is a small preprint, not sufficient by itself to establish a universal ranking. More recent models may outperform π0.5 elsewhere; released weights, an adaptable action interface, and a working RL trainer matter more for this project than a headline score.

## Candidates

| Model | Evidence relevant to this project | Proposed role |
| --- | --- | --- |
| π0.5 | Open weights and adaptation code; direct SO101 results; RLinf documents PPO/GRPO for the model in simulation. Hardware integration remains separate. | Main RL research candidate. |
| SmolVLA, 450M | Official SO100/SO101 demonstrations and consumer-hardware support. Standard LeRobot fine-tuning is imitation learning; experimental third-party GRPO code exists for LIBERO. | Lightweight baseline and laptop inference candidate. |
| GR00T N1.7, 3B | Current LeRobot documentation includes SO101 fine-tuning and deployment, using CUDA. A matching N1.7 RL recipe was not established in this review. | Later hardware comparison. |
| MolmoAct 2 | Open code/data and LeRobot training; strong reported bimanual results. The RLinf page reviewed is an evaluation guide. | Later capability comparison, after checking RL integration. |
| OpenVLA-OFT, 7B | Published LIBERO/ALOHA results and an established RL research ecosystem. Its OFT head uses continuous action regression; do not assume the original token-policy RL loss applies unchanged. | Reproduction baseline if the research question calls for it. |
| X-VLA / EVO1 | LeRobot integrations and smaller-model alternatives; X-VLA includes a bimanual SO101 recipe. A better supported SO101 RL path was not established. | Optional later ablations. |
| FLUX 3 Action, 7B | New world-action model with an SO101 checkpoint and a LeRobot LoRA recipe. The inspected integration explicitly says its current revision still needs GPU/robot validation. | Experimental later candidate. |
| π0.7 | Newer PI model with reported generalization gains; PI now offers hosted fine-tuning/inference. Downloadable π0.7 training weights were not verified in the official openpi repository. | Revisit if hosted-model access fits the project. |

Sources: [SmolVLA](https://huggingface.co/blog/smolvla), [openpi](https://github.com/Physical-Intelligence/openpi), [π0/π0.5 RL](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/pi0.html), [GR00T](https://huggingface.co/docs/lerobot/groot), [MolmoAct 2](https://allenai.org/blog/molmoact2), [MolmoAct2 evaluation](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/molmoact2.html), [OpenVLA-OFT](https://openvla-oft.github.io/), [X-VLA](https://huggingface.co/docs/lerobot/xvla), [EVO1](https://huggingface.co/docs/lerobot/evo1), [FLUX integration](https://huggingface.co/docs/lerobot/main/flux3), [π0.7](https://www.pi.website/blog/pi07), [PI API beta](https://www.pi.website/blog/api-beta).

## Most relevant comparative evidence

A June 2026 SO101 preprint evaluated four tasks, using 100 demonstrations per task and 20 hardware evaluation trials per model/task. Reported mean task success was 56.25% for π0.5, 51.25% for Wall-X, 33.75% for ACT, and 32.5% for SmolVLA. These are results from one setup with a limited evaluation budget, not performance predictions for this arm. Newer candidates were not tested. [Paper, Table 3](https://arxiv.org/html/2606.08881v1)

Ai2 reports a separate, commissioned third-party bimanual evaluation favoring MolmoAct 2 over π0.5 and OpenVLA-OFT. Different hardware/tasks prevent combining that result with the SO101 study into a universal ranking. [Ai2 report](https://allenai.org/blog/molmoact2)

## RL readiness

RLinf explicitly states that SO101 currently has hardware checks but no supported real-world task/training workflow. The project must supply task reset, observation/action contracts, reward, success criteria, and rollout collection before its model trainers become an on-arm RL system. [SO101 guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/so101.html)

For SmolVLA, FlowVLA-RL adds stochastic flow sampling and GRPO on LIBERO. Its authors report seed replication and describe negative results and selection caveats. This is useful reference code, not independently reproduced evidence for physical SO101 or LLM-graded learning. [Repository](https://github.com/BlackMirean/FlowVLA-RL)

LeRobot HIL-SERL is a useful separate SAC baseline. Its documented workflow should not be relabeled as fine-tuning a pretrained VLA. [HIL-SERL](https://huggingface.co/docs/lerobot/hilserl)

## Compute and deployment

The official openpi implementation lists NVIDIA GPU requirements of over 8 GB for inference, over 22.5 GB for LoRA fine-tuning, and over 70 GB for full fine-tuning. These are implementation-specific estimates, not RL memory estimates or guarantees for every configuration. It documents Ubuntu support. [Requirements](https://github.com/Physical-Intelligence/openpi#requirements)

SmolVLA is the better initial candidate when inference must remain on a consumer laptop. Measure action-chunk latency with the actual laptop, camera processing and checkpoint before committing to a control rate. [Official model description](https://huggingface.co/blog/smolvla)

The intended platform separates static dashboard hosting, the laptop robot bridge, and the GPU learner. GitHub Pages cannot run the Python control service. Cluster inference also needs a measured network/queueing path; the local controller must remain responsible for motor timing. [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)

## Proposed first comparison

1. Collect one consistent demonstration dataset with fixed camera placement, action units, and calibration identity. Reserve episodes and physical starting configurations for evaluation.
2. Fine-tune SmolVLA and π0.5 on that data. Compare held-out physical success, action-chunk latency, interventions, and task time before adding RL.
3. Reproduce a supported RL training recipe in simulation, then implement the SO101 task adapter and reward interface.
4. Score trajectories with a vision-language grader using a versioned rubric. Store grader score and human-confirmed success separately. Audit agreement on held-out successes and failures before using its scores for learning. TOPReward is a relevant existing VLM-reward reference, not proof the grader is accurate on this task. [TOPReward](https://huggingface.co/docs/lerobot/main/topreward)
5. Compare the starting VLA, RL-updated VLA, and interactive agent sessions on the same evaluation protocol. Record which weights change. Residual-policy training, VLA weight updates, and adaptation through context must have distinct experiment labels.

Interactive assistant control is a separate controller interface: observe, execute a bounded action or learned skill, observe again, and retain session history. That interface can use the same robot bridge without assuming a pretrained VLA learns from a conversation transcript.

## Details still needed before execution

- Exact SSH target and usable Adroit GPU allocation; inference location and measured latency.
- Laptop chip/memory, arm count and whether a leader arm is available.
- Arm calibration identity and serial port; discovered UVC camera identifier and supported capture modes.
- First task and its reset/success protocol; whether the priority is changing VLA weights or training a smaller controller around a frozen VLA.

These details affect implementation and feasibility; they do not change the current recommendation to retain both π0.5 and SmolVLA in the platform's model registry.

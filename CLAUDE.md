## Compute Environment
- Shared compute on the assigned docker server below; prefer frozen or lightly tuned open models.
- NEVER delete files, caches, checkpoints, datasets, logs, or directories to free disk space.

## Remote Server Access
- SSH entry command:
  ```bash
  ssh -t xuhu@202.120.12.172 -p 8022 "docker exec -it 29e8e3afb73f bash -c 'cd /home/lzy/AAAI_2026/ && exec bash'"
  ```
- SSH password: `xhqweQWE123!@#`
- Allowed remote workspace: `/home/lzy/AAAI_2026/` only. Work inside a per-direction subfolder (e.g. `/home/lzy/AAAI_2026/arch_pipeline/`) to avoid colliding with other directions.
- Do not access, inspect, modify, copy, move, or depend on any path outside `/home/lzy/AAAI_2026/`.
- Remote work is legal only after entering docker container `29e8e3afb73f` with the SSH entry command above.
- NEVER enter, inspect, list, modify, or operate any other docker container.
- NEVER operate on the remote host outside docker. Running project commands without first entering docker container `29e8e3afb73f` is illegal.
- NEVER run `docker ps`, `docker inspect`, or `docker exec` except as part of the exact SSH entry command above.
- NEVER `scp` or `rsync` files to or from paths outside `/home/lzy/AAAI_2026/`.
- Do not delete any files, caches, checkpoints, datasets, logs, directories, or temporary artifacts under any circumstance.
- If disk space is full or storage cleanup seems necessary, stop and report the situation to the user. Do not delete anything yourself.
- Always check `nvidia-smi` before using a GPU; never grab a fixed GPU blindly. Do not kill other users' processes; only stop processes clearly owned by this project/session.
- Env note: use `python3` (not `python`); 3× A800 80GB shared (often busy — check nvidia-smi, wait for a free GPU, never preempt). torch 2.4.1+cu121.

## Non-Negotiable Operating Rules
- NEVER: kill other processes, `pkill`, `kill -9`, or terminate others' jobs.
- NEVER: enter, inspect, list, modify, or operate any docker container except `29e8e3afb73f`.
- NEVER: run project commands on the remote host before entering docker container `29e8e3afb73f`; non-docker remote operation is illegal.
- NEVER: run `docker ps`, `docker inspect`, or `docker exec` except as part of the exact SSH entry command.
- NEVER: `scp` or `rsync` files to or from paths outside `/home/lzy/AAAI_2026/`.
- NEVER: delete any files or directories for cleanup, including your own outputs, caches, checkpoints, logs, lock files, or temporary files.
- NEVER: delete anything even if storage becomes full; instead stop and report the issue.
- NEVER: use a fixed GPU blindly — always check `nvidia-smi` first.
- ALWAYS: keep all existing files intact.
- Write honest negative results — do not fabricate.

## HuggingFace Mirror Guidance
- In China, Hugging Face downloads may be unstable, extremely slow, or inaccessible. Use the HF-Mirror endpoint to avoid changing project code.
- Core principle: replace `huggingface.co` with `hf-mirror.com` through `HF_ENDPOINT`.
- Temporary Linux/macOS setup for the current terminal:
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```
- Python scripts or notebooks must set the endpoint before importing Hugging Face libraries:
  ```python
  import os
  os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
  
  from transformers import AutoTokenizer, AutoModel
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  model = AutoModel.from_pretrained("bert-base-uncased")
  ```
- CLI download workflow:
  ```bash
  pip install huggingface-hub
  export HF_ENDPOINT=https://hf-mirror.com
  huggingface-cli download <model-id>
  ```
- ModelScope is an acceptable fallback when HF/HF-mirror is gated or failing; do not silently substitute unrelated models.

## Academic Network Access
- Prefer official sources first (paper pages, proceedings, arXiv/OpenReview, official benchmark pages, official GitHub).
- GitHub proxy fallback: https://ghproxy.link/
- HuggingFace mirror fallback: https://hf-mirror.com/

## ARIS Config
- AUTO_PROCEED: true
- human checkpoint: false
- effort: max
- venue: AAAI 2027 (main / spotlight target)
- difficulty: hard
- gpu: shared (assigned docker 29e8e3afb73f; check nvidia-smi)
- compute: assigned-remote-docker

- 

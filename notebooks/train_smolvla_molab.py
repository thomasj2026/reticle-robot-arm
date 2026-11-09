import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="SmolVLA on libero6dof")


@app.cell
def _():
    import json
    import os
    import subprocess
    import sys
    import time
    from datetime import datetime
    from pathlib import Path

    import marimo as mo

    return Path, datetime, json, mo, os, subprocess, sys, time


@app.cell
def _(mo):
    mo.md(
        r"""
        # Fine-tune SmolVLA on libero6dof UR5e demos

        Trains `lerobot/smolvla_base` on a LeRobot dataset made by `libero6dof.to_lerobot`.
        Checkpoints (weights **and** optimizer state) go to a Hugging Face model repo as
        they're saved, so a session that hits molab's 12-hour limit (or the 90-minute
        idle shutdown) can pick up where it left off with **Resume**.

        **Before you start**

        1. Attach the GPU: *notebook specs* button in the app header.
        2. Add a Hugging Face token with **write** access as a secret named `HF_TOKEN`
           (secrets panel), or paste it below.
        3. Push your dataset from your own machine:
           `python -m libero6dof.to_lerobot --src data/ur5e --repo-id <you>/<dataset> --push`

        Then fill in the form, run the **dry run** once, and start training.
        """
    )
    return


@app.cell
def _(mo, subprocess, sys):
    # LeRobot 0.7.0 is the version libero6dof's converter and serving adapter were tested against.
    LEROBOT_SPEC = "lerobot[smolvla,dataset,training]==0.7.0"
    try:
        import lerobot

        _status = f"LeRobot {lerobot.__version__} already installed."
    except ImportError:
        _r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", LEROBOT_SPEC], capture_output=True, text=True
        )
        if _r.returncode != 0:
            _r = subprocess.run(["uv", "pip", "install", "--python", sys.executable, "-q", LEROBOT_SPEC],
                                capture_output=True, text=True)
        _status = "Installed " + LEROBOT_SPEC if _r.returncode == 0 else "Install failed:\n" + _r.stderr[-3000:]
    mo.md(f"**Packages:** {_status}")
    return


@app.cell
def _(mo):
    import torch

    if torch.cuda.is_available():
        _p = torch.cuda.get_device_properties(0)
        gpu_ok = True
        _msg = f"GPU: **{_p.name}**, {_p.total_memory / 2**30:.0f} GB"
    else:
        gpu_ok = False
        _msg = "**No GPU attached.** Use the notebook specs button in the header, then re-run this cell."
    mo.callout(mo.md(_msg), kind="success" if gpu_ok else "danger")
    return (gpu_ok,)


@app.cell
def _(mo, os):
    token_box = mo.ui.text(kind="password", label="Hugging Face token (only if HF_TOKEN secret isn't set)", full_width=True)
    token_box if not os.getenv("HF_TOKEN") else mo.md("Using the `HF_TOKEN` secret.")
    return (token_box,)


@app.cell
def _(mo, os, token_box):
    from huggingface_hub import login, whoami

    hf_token = os.getenv("HF_TOKEN") or token_box.value
    mo.stop(not hf_token, mo.md("Waiting for a Hugging Face token."))
    os.environ["HF_TOKEN"] = hf_token  # picked up by lerobot-train
    login(token=hf_token, add_to_git_credential=False)
    hf_user = whoami(token=hf_token)["name"]
    mo.md(f"Logged in to Hugging Face as **{hf_user}**.")
    return (hf_user,)


@app.cell
def _(hf_user, mo):
    form = (
        mo.md(
            """
            **Run settings**

            {dataset} {model_repo}
            {base} {run_name}
            {steps} {batch} {save_freq}
            {mode}
            """
        )
        .batch(
            dataset=mo.ui.text(value=f"{hf_user}/libero6dof_ur5e_spatial", label="Dataset repo", full_width=True),
            model_repo=mo.ui.text(value=f"{hf_user}/smolvla_ur5e_spatial", label="Model / checkpoint repo", full_width=True),
            base=mo.ui.text(value="lerobot/smolvla_base", label="Base model"),
            run_name=mo.ui.text(value="smolvla_ur5e", label="Run name"),
            steps=mo.ui.number(start=10, stop=500_000, step=1000, value=20_000, label="Total steps"),
            batch=mo.ui.number(start=1, stop=512, value=64, label="Batch size"),
            save_freq=mo.ui.number(start=10, stop=100_000, step=500, value=2_000, label="Checkpoint every"),
            mode=mo.ui.radio(["Start new run", "Resume from checkpoint repo"], value="Start new run", label="Mode"),
        )
        .form(submit_button_label="Use these settings")
    )
    form
    return (form,)


@app.cell
def _(form, json, mo):
    from huggingface_hub import hf_hub_download

    mo.stop(form.value is None, mo.md("Submit the settings above."))
    cfg = form.value
    resume = cfg["mode"].startswith("Resume")

    def camera_keys(features: dict) -> list:
        return [k for k, v in features.items() if k.startswith("observation.images.")
                and (v.get("type") == "VISUAL" or v.get("dtype") in ("video", "image"))]

    # Fine-tuning keeps the base model's camera names, so map the dataset's
    # cameras onto them in order (agentview -> first, wrist -> second).
    rename_map = {}
    if not resume:
        _base_cfg = json.load(open(hf_hub_download(cfg["base"], "config.json")))
        _ds_info = json.load(open(hf_hub_download(cfg["dataset"], "meta/info.json", repo_type="dataset")))
        base_cams = camera_keys(_base_cfg["input_features"])
        ds_cams = camera_keys(_ds_info["features"])
        if len(ds_cams) > len(base_cams):
            raise ValueError(f"dataset has {len(ds_cams)} cameras but {cfg['base']} takes {len(base_cams)}")
        rename_map = {d: b for d, b in zip(ds_cams, base_cams) if d != b}
        _summary = mo.md(
            f"Dataset cameras `{ds_cams}` → base model cameras `{base_cams}`  \n"
            f"rename map: `{json.dumps(rename_map)}`  \n"
            f"dataset: {_ds_info.get('total_episodes')} episodes, {_ds_info.get('total_frames')} frames"
        )
    else:
        _summary = mo.md(f"Resuming from the latest checkpoint in `{cfg['model_repo']}`.")
    _summary
    return cfg, rename_map, resume


@app.cell
def _(Path, cfg, datetime, json, rename_map, resume):
    def build_command(dry_run: bool) -> list:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("outputs/train") / (f"{cfg['run_name']}_{'dryrun' if dry_run else stamp}")
        if resume and not dry_run:
            return [
                "lerobot-train",
                f"--config_path={cfg['model_repo']}",
                "--resume=true",
                f"--steps={int(cfg['steps'])}",
                f"--output_dir={out}",
            ]
        cmd = [
            "lerobot-train",
            f"--policy.path={cfg['base']}",
            f"--dataset.repo_id={cfg['dataset']}",
            f"--batch_size={int(cfg['batch'])}",
            "--policy.device=cuda",
            "--num_workers=4",
            f"--output_dir={out}",
            f"--job_name={cfg['run_name']}",
        ]
        if rename_map:
            cmd.append("--rename_map=" + json.dumps(rename_map))
        if dry_run:  # a few steps, nothing pushed: checks data, cameras and memory
            cmd += ["--steps=10", "--save_freq=10", "--log_freq=1", "--policy.push_to_hub=false"]
        else:
            cmd += [
                f"--steps={int(cfg['steps'])}",
                f"--save_freq={int(cfg['save_freq'])}",
                "--log_freq=100",
                f"--policy.repo_id={cfg['model_repo']}",
                "--policy.push_to_hub=true",
                "--save_checkpoint_to_hub=true",
            ]
        return cmd

    return (build_command,)


@app.cell
def _(mo, os, subprocess, time):
    def run_streaming(cmd: list, log_path: str, title: str):
        """Run a command, mirroring the last log lines into the cell output."""
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        lines, last = [], 0.0
        with open(log_path, "w") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for raw in proc.stdout:
                    log.write(raw)
                    for line in raw.replace("\r", "\n").splitlines():
                        if line.strip() and not line.lstrip().startswith("Training:"):
                            lines.append(line)
                    lines = lines[-40:]
                    if time.time() - last > 3:
                        last = time.time()
                        mo.output.replace(mo.vstack([mo.md(f"**{title}** (running)"), mo.plain_text("\n".join(lines))]))
                code = proc.wait()
            finally:
                if proc.poll() is None:  # cell interrupted: stop training cleanly
                    proc.terminate()
                    proc.wait(timeout=60)
        status = "finished" if code == 0 else f"failed (exit {code}); full log: {log_path}"
        mo.output.replace(mo.vstack([mo.md(f"**{title}** {status}"), mo.plain_text("\n".join(lines))]))
        return code

    return (run_streaming,)


@app.cell
def _(build_command, mo, resume):
    dry_button = mo.ui.run_button(label="1. Dry run (10 steps, nothing pushed)", disabled=resume)
    train_button = mo.ui.run_button(label="2. Resume training" if resume else "2. Start training", kind="success")
    mo.vstack([
        mo.md("**Command that will run:**"),
        mo.plain_text(" \\\n  ".join(build_command(dry_run=False))),
        mo.hstack([dry_button, train_button], justify="start"),
    ])
    return dry_button, train_button


@app.cell
def _(build_command, dry_button, gpu_ok, mo, run_streaming):
    mo.stop(not dry_button.value)
    mo.stop(not gpu_ok, mo.md("Attach a GPU first."))
    run_streaming(build_command(dry_run=True), "outputs/logs/dryrun.log", "Dry run")
    return


@app.cell
def _(build_command, cfg, gpu_ok, mo, run_streaming, train_button):
    mo.stop(not train_button.value)
    mo.stop(not gpu_ok, mo.md("Attach a GPU first."))
    _code = run_streaming(build_command(dry_run=False), f"outputs/logs/{cfg['run_name']}.log", "Training")
    mo.md(
        f"""
        Checkpoints are in **[{cfg['model_repo']}](https://huggingface.co/{cfg['model_repo']})**
        (`checkpoints/<step>/pretrained_model` plus optimizer state).

        If the session stopped early, reopen this notebook, choose **Resume**, and press
        the training button again.

        **Evaluate on your machine** (LeRobot env serves, simulator env evaluates):

        ```bash
        LEROBOT_CHECKPOINT={cfg['model_repo']} \\
          python -m libero6dof.server --policy libero6dof.lerobot_policy:make_policy --port 8000
        python -m libero6dof.evaluate --suite libero_spatial --policy remote --port 8000 --replan-steps 10
        ```
        """
    ) if _code == 0 else None
    return


if __name__ == "__main__":
    app.run()

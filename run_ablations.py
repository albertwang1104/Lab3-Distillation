#!/usr/bin/env python3
"""Run the Lab 3 ablations and original-loss control without overwriting outputs."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PROMPTS = [
    "A red bus driving on a desert road",
    "a boat in a river",
    "A cabin surrounded by forests",
    "A church beside a lake",
    "A villa close to the pool",
    "A castle next to a river",
    "A burger on the table",
    "A dog sitting on grass",
    "a cat sitting on a table",
    "A car on the road",
]
SWEEPS = {
    "sds-guidance": ("sds_guidance_scale", ("7.5", "15", "25", "50", "100")),
    "sdi-steps": ("sdi_inversion_n_steps", ("1", "3", "5", "10", "25")),
    "sdi-interval": ("sdi_update_interval", ("25", "75", "200", "500", "1000")),
    "diff-loss": ("original_diffusion_loss", ("baseline",)),
}
SEED = 2024


def parameters(sweep, value):
    if sweep == "sds-guidance":
        return {
            "loss_type": "sds",
            "guidance_scale": value,
            "steps": "500",
            "lr": "0.01",
        }
    if sweep == "diff-loss":
        return {
            "loss_type": "diff",
            "guidance_scale": "1.0",
            "steps": "500",
            "lr": "0.01",
        }
    return {
        "loss_type": "sdi",
        "guidance_scale": "7.5",
        "steps": "1000",
        "lr": "0.005",
        "inversion_n_steps": value if sweep == "sdi-steps" else "10",
        "inversion_guidance_scale": "-7.5",
        "inversion_eta": "0.3",
        "sdi_update_interval": value if sweep == "sdi-interval" else "25",
    }


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", choices=SWEEPS)
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Unique label grouping the experiments (default: current timestamp)",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--dry-run", action="store_true", help="Show settings and paths")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("--run-id must contain only letters, digits, underscores, or hyphens")
    if args.device < 0:
        parser.error("--device must be a nonnegative CUDA device index")

    folder, values = SWEEPS[args.sweep]
    sweep_dir = ROOT / "outputs" / "ablations" / args.run_id / folder
    destinations = [(value, sweep_dir / value) for value in values]
    for value, dest in destinations:
        print(f"{args.sweep}={value}: {dest}")
    if args.dry_run:
        return

    existing = [str(dest) for _, dest in destinations if dest.exists()]
    if existing:
        parser.error(
            "Refusing to overwrite existing result directories:\n" + "\n".join(existing)
            + "\nChoose a new --run-id."
        )

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch, sys; assert torch.cuda.is_available() and int(sys.argv[1]) < torch.cuda.device_count(), 'Requested CUDA GPU is unavailable'",
            str(args.device),
        ],
        cwd=ROOT,
        check=True,
    )

    summary = []
    for value, dest in destinations:
        dest.mkdir(parents=True, exist_ok=False)
        params = parameters(args.sweep, value)
        (dest / "sweep_config.json").write_text(
            json.dumps(
                {
                    "sweep": args.sweep,
                    "value": value,
                    "device": args.device,
                    "parameters": params,
                    "prompts_and_seeds": [
                        {"prompt": prompt, "seed": SEED + index}
                        for index, prompt in enumerate(PROMPTS)
                    ],
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        runtimes = []
        for index, prompt in enumerate(PROMPTS):
            prompt_key = prompt.replace(" ", "_")
            env = os.environ.copy()
            env["LAB3_SEED"] = str(SEED + index)
            env["PYTHONHASHSEED"] = env["LAB3_SEED"]
            command = [
                sys.executable,
                "-c",
                'import os, main; main.seed_everything(int(os.environ["LAB3_SEED"])); main.main()',
                "--prompt",
                prompt,
                "--negative_prompt",
                "low quality",
                "--device",
                str(args.device),
                "--save_dir",
                str(dest),
            ]
            for key, setting in params.items():
                command.extend((f"--{key}", setting))
            print(f"\n[{args.sweep}={value}] {index + 1}/{len(PROMPTS)}: {prompt}", flush=True)
            start = time.monotonic()
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            runtimes.append(
                {"prompt": prompt, "seed": SEED + index, "seconds": round(time.monotonic() - start, 2)}
            )
            shutil.copy2(dest / "run_config.json", dest / f"run_config_{prompt_key}.json")
            # eval.py scores every PNG in this directory; remove only this run's snapshots.
            for image in dest.glob("*.png"):
                if image.stem.isdigit():
                    image.unlink()
        (dest / "runtimes.json").write_text(
            json.dumps(runtimes, indent=2) + "\n", encoding="utf-8"
        )
        subprocess.run(
            [sys.executable, str(ROOT / "eval.py"), "--fdir1", str(dest), "--device", str(args.device)],
            cwd=ROOT,
            check=True,
        )
        score = json.loads((dest / "eval.json").read_text(encoding="utf-8"))["score"]
        summary.append({"value": value, "mean_clip_score": score})
        (sweep_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved images and CLIP scores: {dest}", flush=True)


if __name__ == "__main__":
    run()

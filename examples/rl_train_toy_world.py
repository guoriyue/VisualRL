"""Run a minimal RL training example on top of wm-infra's world-model env API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wm_infra.workloads.rl.training import ExperimentSpec, run_local_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the wm-infra toy RL training demo.")
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--temporal-root", type=str, default="")
    parser.add_argument("--replay-dir", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    result = run_local_experiment(
        ExperimentSpec(
            updates=args.updates,
            num_envs=args.num_envs,
            horizon=args.horizon,
            seed=args.seed,
            eval_episodes=args.eval_episodes,
            eval_interval=args.eval_interval,
            replay_dir=args.replay_dir,
            temporal_root=args.temporal_root,
        )
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(json.dumps(
        {
            "final_mean_return": result["final_mean_return"],
            "best_mean_return": result["best_mean_return"],
            "final_success_rate": result["final_success_rate"],
            "updates": result["config"]["updates"],
            "num_envs": result["config"]["num_envs"],
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()

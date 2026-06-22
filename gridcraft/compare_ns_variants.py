import argparse
import json
import os

from evaluate_world_model import evaluate_rnn_one_step, evaluate_rnn_horizon
from experiment_logging import add_wandb_args, logger_from_args
from experiment_metrics import flatten_variant_summary
from ns_symbolic import NS_VARIANTS
from wandb_schema import GENERAL, WORLD_MODEL_EVALUATION


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--variants", nargs="+", choices=NS_VARIANTS, default=list(NS_VARIANTS))
  parser.add_argument("--vae-json", default="vae/vae.json")
  parser.add_argument("--episodes", type=int, default=100)
  parser.add_argument("--max-steps", type=int, default=500)
  parser.add_argument("--horizon-steps", type=int, default=50)
  parser.add_argument("--horizons", nargs="+", type=int, default=None)
  parser.add_argument("--symbolic-coverage", type=float, default=1.0)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--imagination-mode", choices=["mean", "mode", "sample"], default="mean")
  parser.add_argument("--out", default="trainlog/ns_mawm_summary.json")
  add_wandb_args(parser)
  args = parser.parse_args()

  logger = logger_from_args(
    args,
    config=vars(args),
    default_group="ns_mawm_comparison",
    default_name=f"gridcraft-ns-compare-seed{args.seed}",
    tags=["gridcraft", "ns-mawm", "comparison"],
    info_sections=[GENERAL, WORLD_MODEL_EVALUATION],
    out_dir=os.path.dirname(args.out) or "trainlog",
  )

  summary = {}
  for variant in args.variants:
    eval_args = argparse.Namespace(
      vae_json=args.vae_json,
      rnn_json=None,
      ns_variant=variant,
      symbolic_coverage=args.symbolic_coverage,
      episodes=args.episodes,
      max_steps=args.max_steps,
      horizon_steps=args.horizon_steps,
      horizons=args.horizons,
      seed=args.seed,
      imagination_mode=args.imagination_mode,
      progress_every=0,
    )
    metrics = evaluate_rnn_one_step(eval_args)
    horizons = args.horizons if args.horizons is not None else ([args.horizon_steps] if args.horizon_steps > 0 else [])
    for horizon in horizons:
      eval_args.horizon_steps = int(horizon)
      metrics.update(evaluate_rnn_horizon(eval_args))
    summary[variant] = metrics
    out_path = f"trainlog/ns_eval_{variant}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
      json.dump(metrics, f, indent=2)
    logger.log({variant: metrics}, namespace="wm_evaluation")
    print(variant, json.dumps(metrics, indent=2))

  out_dir = os.path.dirname(args.out)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
  with open(args.out, "w") as f:
    json.dump(summary, f, indent=2)
  logger.log(flatten_variant_summary(summary), namespace="wm_evaluation")
  logger.save_json(args.out, summary)
  logger.finish()
  print("saved", args.out)


if __name__ == "__main__":
  main()

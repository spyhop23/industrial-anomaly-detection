"""
Train and evaluate an anomaly detection model on MVTec AD.

Usage:
    python scripts/train.py --category bottle --model patchcore
"""

import argparse
import time

import mlflow
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore, Padim, EfficientAd
from lightning.pytorch.callbacks import TQDMProgressBar

MODEL_REGISTRY = {
    "patchcore": Patchcore,
    "padim": Padim,
    "efficientad": EfficientAd,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train anomaly detection model on MVTec AD")
    parser.add_argument("--root", type=str, required=True,
                         help="Path to MVTec AD dataset root (contains category folders)")
    parser.add_argument("--category", type=str, default="bottle",
                         help="MVTec AD category, e.g. bottle, cable, capsule")
    parser.add_argument("--model", type=str, default="patchcore",
                         choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--mlflow_db", type=str, default="mlflow.db",
                         help="Path to local MLflow sqlite db")
    parser.add_argument("--experiment_name", type=str, default="mvtec-anomaly-detection")
    return parser.parse_args()


def train_and_evaluate(args):
    datamodule = MVTecAD(
        root=args.root,
        category=args.category,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls()

    engine = Engine(callbacks=[TQDMProgressBar()])

    start = time.time()
    engine.fit(datamodule=datamodule, model=model)
    train_time = time.time() - start

    test_results = engine.test(datamodule=datamodule, model=model)[0]
    return test_results, train_time


def log_to_mlflow(args, test_results, train_time):
    mlflow.set_tracking_uri(f"sqlite:///{args.mlflow_db}")
    mlflow.set_experiment(args.experiment_name)

    run_name = f"{args.model}-{args.category}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", args.model)
        mlflow.log_param("category", args.category)
        mlflow.log_param("train_batch_size", args.train_batch_size)
        mlflow.log_param("eval_batch_size", args.eval_batch_size)
        mlflow.log_metric("train_time_sec", train_time)
        for key, value in test_results.items():
            mlflow.log_metric(key, value)

    print(f"[{run_name}] image_AUROC={test_results['image_AUROC']:.4f}, "
          f"pixel_AUROC={test_results['pixel_AUROC']:.4f}, "
          f"train_time={train_time:.1f}s")


def main():
    args = parse_args()
    test_results, train_time = train_and_evaluate(args)
    log_to_mlflow(args, test_results, train_time)


if __name__ == "__main__":
    main()
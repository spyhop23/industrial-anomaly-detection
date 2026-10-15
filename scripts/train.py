"""
Train and evaluate an anomaly detection model on MVTec AD.

Usage:
    python scripts/train.py --root /path/to/mvtec_ad --category bottle --model patchcore
    python scripts/train.py --root /path/to/mvtec_ad --category bottle --model efficientad --max_steps 5000
"""
import argparse
import time

import torch
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

MODEL_DEFAULT_BATCH_SIZE = {
    "patchcore": 32,
    "padim": 32,
    "efficientad": 1,  # EfficientAD requires batch_size=1
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train anomaly detection model on MVTec AD")
    parser.add_argument("--root", type=str, required=True, help="Path to MVTec AD dataset root")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--model", type=str, default="patchcore", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--train_batch_size", type=int, default=None,
                         help="Overrides the model's default batch size if set")
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None,
                         help="Preferred over max_epochs for EfficientAD (small per-category datasets)")
    parser.add_argument("--mlflow_db", type=str, default="mlflow.db")
    parser.add_argument("--experiment_name", type=str, default="mvtec-anomaly-detection")
    return parser.parse_args()


def measure_inference_latency(model, datamodule, n_images=50):
    """Measure per-image inference latency on GPU if available.

    NOTE: Lightning may move the model back to CPU after engine.test(), so we
    explicitly force the device here instead of relying on
    next(model.parameters()).device, which can silently report CPU timings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader = datamodule.test_dataloader()
    batch = next(iter(loader))
    imgs = batch["image"][:1].to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(imgs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(n_images):
            model(imgs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start
    return (elapsed / n_images) * 1000  # ms/image


def train_and_evaluate(args):
    train_bs = args.train_batch_size or MODEL_DEFAULT_BATCH_SIZE[args.model]

    datamodule = MVTecAD(
        root=args.root,
        category=args.category,
        train_batch_size=train_bs,
        eval_batch_size=args.eval_batch_size,
    )

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls()

    engine_kwargs = {"callbacks": [TQDMProgressBar()]}
    if args.max_steps:
        engine_kwargs["max_steps"] = args.max_steps
    elif args.max_epochs:
        engine_kwargs["max_epochs"] = args.max_epochs
    elif args.model == "efficientad":
        # EfficientAD benefits from step-based training on MVTec's small per-category sets
        engine_kwargs["max_steps"] = 5000
    else:
        engine_kwargs["max_epochs"] = 1

    engine = Engine(**engine_kwargs)

    start = time.time()
    engine.fit(datamodule=datamodule, model=model)
    train_time = time.time() - start

    test_results = engine.test(datamodule=datamodule, model=model)[0]
    latency_ms = measure_inference_latency(model, datamodule)

    return test_results, train_time, latency_ms, train_bs


def log_to_mlflow(args, test_results, train_time, latency_ms, train_bs):
    mlflow.set_tracking_uri(f"sqlite:///{args.mlflow_db}")
    mlflow.set_experiment(args.experiment_name)

    run_name = f"{args.model}-{args.category}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", args.model)
        mlflow.log_param("category", args.category)
        mlflow.log_param("train_batch_size", train_bs)
        mlflow.log_param("max_epochs", args.max_epochs)
        mlflow.log_param("max_steps", args.max_steps)
        mlflow.log_metric("train_time_sec", train_time)
        mlflow.log_metric("inference_latency_ms", latency_ms)
        for key, value in test_results.items():
            mlflow.log_metric(key, value)

    print(
        f"[{run_name}] image_AUROC={test_results['image_AUROC']:.4f}, "
        f"pixel_AUROC={test_results['pixel_AUROC']:.4f}, "
        f"train_time={train_time:.1f}s, latency={latency_ms:.2f}ms/img"
    )


def main():
    args = parse_args()
    test_results, train_time, latency_ms, train_bs = train_and_evaluate(args)
    log_to_mlflow(args, test_results, train_time, latency_ms, train_bs)


if __name__ == "__main__":
    main()
"""
실제 서빙 엔드포인트(FastAPI /inspect)에서 수집한 anomaly_score를 바탕으로
클래스 불균형, 라벨 노이즈, 임계값 재보정 효과를 검증하는 스크립트.

핵심 발견:
- 재보정 효과는 모델이 충분히 학습됐을 때만 유효하다.
  capsule을 5000-step(under-trained, AUROC 0.68) vs 15000-step(재학습, AUROC 0.88)
  체크포인트로 교차검증한 결과, 동일한 재보정 기법이 전자에서는 무의미했고
  후자에서는 Precision을 0.832 -> 0.872까지 끌어올렸다 (Recall 1.0 유지).

입력: evaluation/results/{category}_inspect_results.csv
      (컬럼: filepath, filename, anomaly_score, is_anomaly, true_label, defect_type)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve,
)

N_SAMPLES = 500
N_TRIALS = 20
SEED = 42


def load(filename: str) -> pd.DataFrame:
    df = pd.read_csv(f"evaluation/results/{filename}")
    df["y_true"] = (df["true_label"] == "anomaly").astype(int)
    return df


def evaluate_at_ratio(df: pd.DataFrame, anomaly_ratio: float) -> dict:
    normal_df = df[df["y_true"] == 0]
    anomaly_df = df[df["y_true"] == 1]
    n_anomaly = int(N_SAMPLES * anomaly_ratio)
    n_normal = N_SAMPLES - n_anomaly

    metrics = []
    for trial in range(N_TRIALS):
        sampled_normal = normal_df.sample(n=n_normal, replace=True, random_state=SEED + trial)
        sampled_anomaly = anomaly_df.sample(n=n_anomaly, replace=True, random_state=SEED + trial + 1000)
        sub = pd.concat([sampled_normal, sampled_anomaly])
        y_true, y_score = sub["y_true"].values, sub["anomaly_score"].values
        y_pred = sub["is_anomaly"].astype(int).values
        metrics.append({
            "auroc": roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else np.nan,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })
    return pd.DataFrame(metrics).mean().to_dict()


def evaluate_with_label_noise(df: pd.DataFrame, noise_rate: float) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    aurocs = []
    for trial in range(N_TRIALS):
        y_true = df["y_true"].values.copy()
        n_flip = int(len(y_true) * noise_rate)
        flip_idx = rng.choice(len(y_true), size=n_flip, replace=False)
        y_true[flip_idx] = 1 - y_true[flip_idx]
        if len(set(y_true)) > 1:
            aurocs.append(roc_auc_score(y_true, df["anomaly_score"].values))
    return float(np.mean(aurocs)), float(np.std(aurocs))


def find_optimal_threshold(df: pd.DataFrame) -> tuple[float, dict]:
    y_true, y_score = df["y_true"].values, df["anomaly_score"].values
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = f1s[:-1].argmax()
    return thresholds[best_idx], {
        "precision": precisions[best_idx], "recall": recalls[best_idx], "f1": f1s[best_idx],
    }


def evaluate_threshold(df: pd.DataFrame, threshold: float) -> dict:
    y_true = df["y_true"].values
    y_pred = (df["anomaly_score"].values >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_category(label: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 50}\n{label}\n{'=' * 50}")

    print("\n[불균형 시뮬레이션]")
    for ratio in [0.5, 0.1, 0.05, 0.01]:
        m = evaluate_at_ratio(df, ratio)
        print(f"  anomaly_ratio={ratio:<5} auroc={m['auroc']:.3f} precision={m['precision']:.3f} "
              f"recall={m['recall']:.3f} f1={m['f1']:.3f}")

    print("\n[라벨 노이즈 강건성]")
    for rate in [0.0, 0.05, 0.1, 0.2]:
        mean, std = evaluate_with_label_noise(df, rate)
        print(f"  noise_rate={rate:<5} auroc={mean:.3f} (std={std:.3f})")

    print("\n[임계값 재보정 검증]")
    th, info = find_optimal_threshold(df)
    auroc = roc_auc_score(df["y_true"], df["anomaly_score"])
    print(f"  AUROC: {auroc:.3f}")
    print(f"  F1-최적 임계값: {th:.4f}  (precision={info['precision']:.3f}, recall={info['recall']:.3f})")
    print(f"  0.5 고정값:     {evaluate_threshold(df, 0.5)}")
    print(f"  재보정값:       {evaluate_threshold(df, th)}")


if __name__ == "__main__":
    run_category("bottle", load("bottle_inspect_results.csv"))
    run_category("capsule (5000 step, under-trained)", load("capsule_inspect_results.csv"))
    run_category("capsule (15000 step, 재학습)", load("capsule_v2_15000step_inspect_results.csv"))
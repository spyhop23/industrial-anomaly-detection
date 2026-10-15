"""
Week 4 - EfficientAD TensorRT FP16 optimization

PyTorch checkpoint -> ONNX -> FP16 ONNX -> TensorRT engine 변환 후,
원본 PyTorch(FP32) 모델과 TensorRT(FP16) 엔진의 레이턴시/정확도(AUROC)/모델 크기를 비교한다.

주의: 이 스크립트는 Kaggle 노트북에서 실행했던 코드를 정리한 것으로,
CHECKPOINTS / DATA_ROOT 경로는 실행 환경에 맞게 수정해야 한다.
(Kaggle 전용 경로이므로 로컬/다른 환경에서는 그대로 동작하지 않는다.)

필요 패키지:
    pip install anomalib mlflow lightning tensorrt onnx onnxscript onnxconverter-common
"""

import os
import time

import mlflow
import numpy as np
import onnx
import pandas as pd
import tensorrt as trt
import torch
import torch.nn.functional as F
from onnxconverter_common import float16
from sklearn.metrics import roc_auc_score

from anomalib.data import MVTecAD
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import EfficientAd

# ---------------------------------------------------------------------------
# 설정 (환경에 맞게 수정)
# ---------------------------------------------------------------------------
DATA_ROOT = "/kaggle/working/mvtec_ad"
EXPORT_DIR = "/kaggle/working/exported"
os.makedirs(EXPORT_DIR, exist_ok=True)

CHECKPOINTS = {
    "bottle": "/kaggle/input/notebooks/spyhop/03-multi-category-experiments/results/EfficientAd/MVTecAD/bottle/v0/weights/lightning/model.ckpt",
    "capsule": "/kaggle/input/notebooks/spyhop/03-efficientad-highstep-ablation/results/EfficientAd/MVTecAD/capsule/v0/weights/lightning/model.ckpt",
}

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

TRT_DTYPE_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}


# ---------------------------------------------------------------------------
# 1. ONNX export
# ---------------------------------------------------------------------------
def export_onnx(checkpoints: dict) -> dict:
    onnx_paths = {}
    for category, ckpt_path in checkpoints.items():
        print(f"--- ONNX export: {category} ---")
        model = EfficientAd.load_from_checkpoint(ckpt_path)
        engine = Engine()
        export_root = os.path.join(EXPORT_DIR, category)
        exported_path = engine.export(model=model, export_type=ExportType.ONNX, export_root=export_root)
        onnx_paths[category] = str(exported_path)
        print(f"  -> {exported_path}")
    return onnx_paths


# ---------------------------------------------------------------------------
# 2. ONNX FP16 변환
#    (Greater 노드는 FP32 유지 - pred_label/pred_mask 임계값 비교에서
#     FP16/FP32 타입 불일치 에러가 나기 때문)
# ---------------------------------------------------------------------------
def convert_to_fp16(onnx_paths: dict) -> dict:
    onnx_fp16_paths = {}
    for category, path in onnx_paths.items():
        model = onnx.load(path)
        model_fp16 = float16.convert_float_to_float16(
            model, keep_io_types=True, op_block_list=["Greater"],
        )
        new_path = path.replace(".onnx", "_fp16.onnx")
        onnx.save(model_fp16, new_path)
        onnx_fp16_paths[category] = new_path
        print(f"{category}: FP16 변환 완료 -> {new_path}")
    return onnx_fp16_paths


# ---------------------------------------------------------------------------
# 3. TensorRT 엔진 빌드 (Strongly Typed Network)
#    optimization profile은 반드시 "실제 입력 크기"로 고정해야 한다.
#    (-1 -> 1로 무조건 치환하면 H/W까지 1x1로 고정되어 엔진이 조용히 깨진다)
# ---------------------------------------------------------------------------
def build_tensorrt_engine(onnx_path: str, engine_path: str, fixed_shape: tuple,
                           max_workspace_size: int = 1 << 30) -> str:
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # parse_from_file 필수: parse(bytes)는 external data(.onnx.data)를 못 찾는다
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"ONNX 파싱 실패: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)

    input_tensor = network.get_input(0)
    print(f"ONNX 입력 shape: {input_tensor.shape}, 실제 사용할 shape: {fixed_shape}")

    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, fixed_shape, fixed_shape, fixed_shape)
    config.add_optimization_profile(profile)

    print(f"엔진 빌드 중... ({onnx_path})")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("엔진 빌드 실패")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"엔진 저장 완료: {engine_path}")
    return engine_path


# ---------------------------------------------------------------------------
# 4. TensorRT 추론 래퍼 (TensorRT 10+ API: execute_async_v3)
# ---------------------------------------------------------------------------
class TRTEngineWrapper:
    def __init__(self, engine_path: str):
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_name = None
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

    def infer(self, input_tensor: torch.Tensor) -> dict:
        input_tensor = input_tensor.contiguous().to("cuda", dtype=torch.float32)
        self.context.set_input_shape(self.input_name, tuple(input_tensor.shape))
        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())

        outputs = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            torch_dtype = TRT_DTYPE_TO_TORCH.get(self.engine.get_tensor_dtype(name), torch.float32)
            out = torch.empty(shape, dtype=torch_dtype, device="cuda")
            outputs[name] = out
            self.context.set_tensor_address(name, out.data_ptr())

        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return outputs


# ---------------------------------------------------------------------------
# 5. 벤치마크 유틸
# ---------------------------------------------------------------------------
def get_field(obj, name):
    return getattr(obj, name) if hasattr(obj, name) else obj[name]


def measure_latency_pytorch(model, dataloader, n_images: int = 50) -> float:
    device = torch.device("cuda")
    model = model.to(device).eval()
    batch = next(iter(dataloader))
    imgs = batch["image"][:1].to(device)
    with torch.no_grad():
        for _ in range(10):
            model(imgs)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_images):
            model(imgs)
    torch.cuda.synchronize()
    return ((time.time() - start) / n_images) * 1000


def measure_latency_trt(engine_wrapper: TRTEngineWrapper, dataloader, n_images: int = 50) -> float:
    batch = next(iter(dataloader))
    imgs = batch["image"][:1].cuda()
    for _ in range(10):
        engine_wrapper.infer(imgs)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_images):
        engine_wrapper.infer(imgs)
    torch.cuda.synchronize()
    return ((time.time() - start) / n_images) * 1000


def evaluate_model(predict_fn, dataloader) -> tuple:
    image_scores, image_labels = [], []
    pixel_scores, pixel_labels = [], []

    for batch in dataloader:
        imgs = batch["image"].cuda()
        gt_label = batch["gt_label"]
        gt_mask = batch["gt_mask"]

        for i in range(imgs.shape[0]):
            out = predict_fn(imgs[i:i + 1])
            score = get_field(out, "pred_score")
            amap = get_field(out, "anomaly_map")

            score_val = score.detach().flatten()[0].item() if hasattr(score, "detach") else float(score)
            image_scores.append(score_val)
            image_labels.append(int(gt_label[i].item()))

            mask_i = gt_mask[i]
            target_h, target_w = mask_i.shape[-2], mask_i.shape[-1]

            amap_t = amap.detach().float()
            if amap_t.dim() == 3:
                amap_t = amap_t.unsqueeze(1)
            if amap_t.shape[-2:] != (target_h, target_w):
                amap_t = F.interpolate(amap_t, size=(target_h, target_w), mode="bilinear", align_corners=False)

            pixel_scores.append(amap_t.cpu().numpy().flatten())
            pixel_labels.append(mask_i.cpu().numpy().flatten().astype(int))

    image_auroc = roc_auc_score(image_labels, image_scores)
    pixel_auroc = roc_auc_score(np.concatenate(pixel_labels), np.concatenate(pixel_scores))
    return image_auroc, pixel_auroc


def pytorch_predict(model):
    model.eval()

    def _predict(img):
        with torch.no_grad():
            return model(img)

    return _predict


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    categories = list(CHECKPOINTS.keys())

    datamodules = {}
    for category in categories:
        dm = MVTecAD(root=DATA_ROOT, category=category, eval_batch_size=1)
        dm.setup()
        datamodules[category] = dm

    onnx_paths = export_onnx(CHECKPOINTS)
    onnx_fp16_paths = convert_to_fp16(onnx_paths)

    trt_paths = {}
    for category, onnx_path in onnx_fp16_paths.items():
        sample_shape = tuple(next(iter(datamodules[category].test_dataloader()))["image"].shape)
        engine_path = os.path.join(EXPORT_DIR, category, "model_fp16.engine")
        trt_paths[category] = build_tensorrt_engine(onnx_path, engine_path, fixed_shape=sample_shape)

    trt_engines = {category: TRTEngineWrapper(trt_paths[category]) for category in categories}

    pt_models = {category: EfficientAd.load_from_checkpoint(ckpt) for category, ckpt in CHECKPOINTS.items()}

    rows = []
    for category in categories:
        loader = datamodules[category].test_dataloader()

        pt_latency = measure_latency_pytorch(pt_models[category], loader)
        trt_latency = measure_latency_trt(trt_engines[category], loader)

        pt_img_auroc, pt_pix_auroc = evaluate_model(pytorch_predict(pt_models[category]), loader)
        trt_img_auroc, trt_pix_auroc = evaluate_model(trt_engines[category].infer, loader)

        ckpt_size = os.path.getsize(CHECKPOINTS[category]) / (1024 ** 2)
        onnx_fp32_path = onnx_paths[category]
        onnx_fp32_size = os.path.getsize(onnx_fp32_path) / (1024 ** 2)
        data_file = onnx_fp32_path + ".data"
        if os.path.exists(data_file):
            onnx_fp32_size += os.path.getsize(data_file) / (1024 ** 2)
        engine_size = os.path.getsize(trt_paths[category]) / (1024 ** 2)

        row = {
            "category": category,
            "pytorch_ms": pt_latency,
            "tensorrt_fp16_ms": trt_latency,
            "speedup": pt_latency / trt_latency,
            "pytorch_image_auroc": pt_img_auroc,
            "pytorch_pixel_auroc": pt_pix_auroc,
            "tensorrt_image_auroc": trt_img_auroc,
            "tensorrt_pixel_auroc": trt_pix_auroc,
            "checkpoint_mb": ckpt_size,
            "onnx_fp32_mb": onnx_fp32_size,
            "tensorrt_fp16_engine_mb": engine_size,
        }
        rows.append(row)
        print(row)

        with mlflow.start_run(run_name=f"EfficientAD-{category}-tensorrt-fp16"):
            mlflow.log_params({"category": category, "precision": "fp16"})
            for k, v in row.items():
                if k != "category":
                    mlflow.log_metric(k, v)

    summary_df = pd.DataFrame(rows)
    print(summary_df)
    return summary_df


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:////kaggle/working/week4_mlflow.db")
    mlflow.set_experiment("week4-tensorrt-optimization")
    main()
"""
Week 4(확장) - EfficientAD TensorRT FP16 + INT8 양자화 비교

기존 export_tensorrt.py(FP32 -> ONNX -> FP16 TensorRT)에
INT8 PTQ(Post-Training Quantization, Entropy Calibration) 경로를 추가한 버전.

PyTorch(FP32) / TensorRT(FP16) / TensorRT(INT8) 세 갈래로
레이턴시 · 정확도(AUROC) · 모델 크기를 비교한다.

중요: FP16 엔진은 STRONGLY_TYPED 네트워크로 빌드하지만,
      INT8은 그래프에 박힌 타입이 아니라 "레이어별 동적 범위를 보고
      TensorRT가 스스로 INT8 적용 여부를 판단"하는 캘리브레이션 방식이라
      STRONGLY_TYPED와 함께 쓸 수 없다. 그래서 INT8 엔진은 일반(규칙적) 네트워크로
      별도로 빌드한다.

캘리브레이션 데이터는 MVTec의 "train split(정상 이미지만)"을 사용한다.
EfficientAD를 포함한 이상탐지 모델은 정상 분포를 학습하는 구조이므로,
양자화 범위도 같은 정상 분포 기준으로 잡는 것이 자연스럽다.

주의: 이 스크립트는 Kaggle에서 쓰던 경로(CHECKPOINTS / DATA_ROOT)를
로컬 환경에 맞게 수정해야 한다 (기존 export_tensorrt.py와 동일한 전제).

필요 패키지:
    pip install anomalib mlflow lightning tensorrt onnx onnxscript onnxconverter-common
    (INT8 경로 추가 의존성 없음 - 캘리브레이터 버퍼도 기존 코드처럼 torch CUDA 텐서로 처리)
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
DATA_ROOT = "/path/to/local/mvtec_ad"          # TODO: 로컬 경로로 수정
EXPORT_DIR = "/path/to/local/exported"          # TODO: 로컬 경로로 수정
os.makedirs(EXPORT_DIR, exist_ok=True)

CHECKPOINTS = {
    "bottle": "/path/to/local/checkpoints/bottle/model.ckpt",    # TODO
    "capsule": "/path/to/local/checkpoints/capsule/model.ckpt",  # TODO
}

# 카테고리별 캘리브레이션에 쓸 정상 이미지 배치 수.
# 너무 적으면 동적 범위가 불안정하고, 너무 많으면 시간이 오래 걸린다.
# 보통 100~500장 사이에서 시작해 정확도 손실을 보며 조정한다.
N_CALIBRATION_BATCHES = 100

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

TRT_DTYPE_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
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
# 3. TensorRT 엔진 빌드 (FP16 - Strongly Typed Network)
#    optimization profile은 반드시 "실제 입력 크기"로 고정해야 한다.
#    (-1 -> 1로 무조건 치환하면 H/W까지 1x1로 고정되어 엔진이 조용히 깨진다)
# ---------------------------------------------------------------------------
def build_tensorrt_engine_fp16(onnx_path: str, engine_path: str, fixed_shape: tuple,
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

    print(f"FP16 엔진 빌드 중... ({onnx_path})")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("FP16 엔진 빌드 실패")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"엔진 저장 완료: {engine_path}")
    return engine_path


# ---------------------------------------------------------------------------
# 3b. INT8 캘리브레이터
#     - MVTec train split(정상 이미지만)에서 배치를 뽑아
#       레이어별 활성값 동적 범위를 측정한다.
#     - 디바이스 버퍼는 기존 TRTEngineWrapper와 동일하게 torch CUDA 텐서로 처리
#       (pycuda 등 추가 의존성 불필요).
# ---------------------------------------------------------------------------
class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataloader, cache_file: str, n_batches: int = N_CALIBRATION_BATCHES):
        super().__init__()
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.cache_file = cache_file
        self.n_batches = n_batches
        self.count = 0
        self.device_input = None  # 첫 배치에서 shape 보고 1회 할당, 이후 재사용

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names):
        if self.count >= self.n_batches:
            return None
        try:
            batch = next(self.iterator)
        except StopIteration:
            return None

        img = batch["image"][:1].contiguous().to("cuda", dtype=torch.float32)
        if self.device_input is None:
            self.device_input = img.clone()
        else:
            self.device_input.copy_(img)

        self.count += 1
        if self.count % 20 == 0:
            print(f"  캘리브레이션 진행: {self.count}/{self.n_batches}")
        return [int(self.device_input.data_ptr())]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"  캘리브레이션 캐시 재사용: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"  캘리브레이션 캐시 저장: {self.cache_file}")


# ---------------------------------------------------------------------------
# 3c. TensorRT 엔진 빌드 (INT8 - 일반 네트워크, STRONGLY_TYPED 사용 불가)
#     원본 FP32 ONNX(onnx_paths)를 입력으로 사용한다.
#     (FP16 변환본이 아니라 FP32본을 써야 캘리브레이터가 보는 동적 범위가
#      실제 추론 시 값과 어긋나지 않는다)
# ---------------------------------------------------------------------------
def build_tensorrt_engine_int8(onnx_path: str, engine_path: str, fixed_shape: tuple,
                                calibrator: EntropyCalibrator,
                                max_workspace_size: int = 1 << 30) -> str:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)  # STRONGLY_TYPED 아님: 캘리브레이션 기반 PTQ용
    parser = trt.OnnxParser(network, TRT_LOGGER)

    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"ONNX 파싱 실패: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator

    input_tensor = network.get_input(0)
    print(f"ONNX 입력 shape: {input_tensor.shape}, 실제 사용할 shape: {fixed_shape}")

    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, fixed_shape, fixed_shape, fixed_shape)
    config.add_optimization_profile(profile)
    config.set_calibration_profile(profile)  # 캘리브레이션도 같은 고정 shape로 수행

    print(f"INT8 엔진 빌드 중 (캘리브레이션 포함, 시간이 걸릴 수 있음)... ({onnx_path})")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("INT8 엔진 빌드 실패")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"엔진 저장 완료: {engine_path}")
    return engine_path


# ---------------------------------------------------------------------------
# 4. TensorRT 추론 래퍼 (TensorRT 10+ API: execute_async_v3)
#    FP16/INT8 엔진 공용 - 엔진 내부 정밀도와 무관하게 동일하게 호출한다.
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


def file_size_mb(path: str) -> float:
    size = os.path.getsize(path)
    data_file = path + ".data"
    if os.path.exists(data_file):
        size += os.path.getsize(data_file)
    return size / (1024 ** 2)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    categories = list(CHECKPOINTS.keys())

    datamodules = {}
    calib_datamodules = {}
    for category in categories:
        dm = MVTecAD(root=DATA_ROOT, category=category, eval_batch_size=1)
        dm.setup()
        datamodules[category] = dm

        # 캘리브레이션 전용: train split(정상 이미지)에서 1장씩 순회
        calib_dm = MVTecAD(root=DATA_ROOT, category=category, train_batch_size=1, eval_batch_size=1)
        calib_dm.setup()
        calib_datamodules[category] = calib_dm

    onnx_paths = export_onnx(CHECKPOINTS)
    onnx_fp16_paths = convert_to_fp16(onnx_paths)

    trt_fp16_paths = {}
    trt_int8_paths = {}
    for category in categories:
        sample_shape = tuple(next(iter(datamodules[category].test_dataloader()))["image"].shape)

        fp16_engine_path = os.path.join(EXPORT_DIR, category, "model_fp16.engine")
        trt_fp16_paths[category] = build_tensorrt_engine_fp16(
            onnx_fp16_paths[category], fp16_engine_path, fixed_shape=sample_shape,
        )

        print(f"--- INT8 캘리브레이션: {category} ---")
        cache_file = os.path.join(EXPORT_DIR, category, "int8_calibration.cache")
        calibrator = EntropyCalibrator(
            calib_datamodules[category].train_dataloader(),
            cache_file=cache_file,
            n_batches=N_CALIBRATION_BATCHES,
        )
        int8_engine_path = os.path.join(EXPORT_DIR, category, "model_int8.engine")
        trt_int8_paths[category] = build_tensorrt_engine_int8(
            onnx_paths[category],  # FP32 원본 사용 (FP16 변환본 아님)
            int8_engine_path, fixed_shape=sample_shape, calibrator=calibrator,
        )

    trt_fp16_engines = {c: TRTEngineWrapper(trt_fp16_paths[c]) for c in categories}
    trt_int8_engines = {c: TRTEngineWrapper(trt_int8_paths[c]) for c in categories}
    pt_models = {c: EfficientAd.load_from_checkpoint(ckpt) for c, ckpt in CHECKPOINTS.items()}

    rows = []
    for category in categories:
        loader = datamodules[category].test_dataloader()

        pt_latency = measure_latency_pytorch(pt_models[category], loader)
        fp16_latency = measure_latency_trt(trt_fp16_engines[category], loader)
        int8_latency = measure_latency_trt(trt_int8_engines[category], loader)

        pt_img_auroc, pt_pix_auroc = evaluate_model(pytorch_predict(pt_models[category]), loader)
        fp16_img_auroc, fp16_pix_auroc = evaluate_model(trt_fp16_engines[category].infer, loader)
        int8_img_auroc, int8_pix_auroc = evaluate_model(trt_int8_engines[category].infer, loader)

        row = {
            "category": category,
            "pytorch_ms": pt_latency,
            "tensorrt_fp16_ms": fp16_latency,
            "tensorrt_int8_ms": int8_latency,
            "speedup_fp16": pt_latency / fp16_latency,
            "speedup_int8": pt_latency / int8_latency,
            "pytorch_image_auroc": pt_img_auroc,
            "pytorch_pixel_auroc": pt_pix_auroc,
            "tensorrt_fp16_image_auroc": fp16_img_auroc,
            "tensorrt_fp16_pixel_auroc": fp16_pix_auroc,
            "tensorrt_int8_image_auroc": int8_img_auroc,
            "tensorrt_int8_pixel_auroc": int8_pix_auroc,
            "auroc_drop_fp16": pt_img_auroc - fp16_img_auroc,
            "auroc_drop_int8": pt_img_auroc - int8_img_auroc,
            "checkpoint_mb": os.path.getsize(CHECKPOINTS[category]) / (1024 ** 2),
            "onnx_fp32_mb": file_size_mb(onnx_paths[category]),
            "tensorrt_fp16_engine_mb": os.path.getsize(trt_fp16_paths[category]) / (1024 ** 2),
            "tensorrt_int8_engine_mb": os.path.getsize(trt_int8_paths[category]) / (1024 ** 2),
        }
        rows.append(row)
        print(row)

        with mlflow.start_run(run_name=f"EfficientAD-{category}-tensorrt-int8"):
            mlflow.log_params({
                "category": category,
                "precision": "int8_vs_fp16_vs_fp32",
                "n_calibration_batches": N_CALIBRATION_BATCHES,
            })
            for k, v in row.items():
                if k != "category":
                    mlflow.log_metric(k, v)

    summary_df = pd.DataFrame(rows)
    print(summary_df)
    return summary_df


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///week4_mlflow.db")  # 로컬 상대 경로로 변경
    mlflow.set_experiment("week4-tensorrt-int8-optimization")
    main()

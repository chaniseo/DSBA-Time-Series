"""
4단계 검증 — 모델 계약 준수 / 차원 변화표(요구사항 2-2) / 파라미터 분석

[TEST 1] cell-19와 동일한 경로로 생성되는가
         globals() 조회 → 위치인자 1개 → nn.Module
[TEST 2] 실제 배치로 forward가 통과하고 차원 변화가 설계와 일치하는가  ← 2-2 산출물
[TEST 3] 파라미터 구성 및 baseline(18,624) 대비
[TEST 4] RevIN 가역성 / 그래디언트 흐름 / 수치 안정성
"""

import numpy as np
import torch
from omegaconf import OmegaConf

import models  # noqa: F401  (globals() 조회 대상)
from data_provider import create_dataloader_default
from models import DRLinear, RevIN  # noqa: F401

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 준비 (3단계 산출물 재사용)
# ─────────────────────────────────────────────────────────────────────────────
info, trn, val, tst = create_dataloader_default(
    task_name="long_term_forecast",
    data_name="custom",
    sub_data_name=None,
    data_info={
        "datadir": "dataset/ETT-small/ETTh1_missing.csv",
        "handle_hidden_missing": True,
    },
    train_setting={"batch_size": 8, "test_batch_size": 8, "num_workers": 0,
                   "pin_memory": False, "shuffle": False},
    scaler="standard", window_size=96, label_len=0, pred_len=96,
    model_type="forecasting", split_rate=[0.7, 0.1, 0.2], timeenc=0, freq="h",
)

MODELSETTING = OmegaConf.create({
    "window_size": 96, "label_len": 0, "pred_len": 96,
    "taskname": "long_term_forecast", "pretrain": False,
    "timeenc": 0, "freq": "h", "embed_type": "learned", "batch_size": 8,
    "enc_in": info["enc_in"], "c_out": info["c_out"],
    "moving_avg": 25, "individual": True, "d_model": 16, "dropout": 0.1,
    "use_revin": True, "use_time_bias": True,
})


def create_model(modelname, params):
    """cell-19의 create_model을 그대로 재현 — 계약 준수 확인용."""
    model_classes = globals()
    if modelname not in model_classes:
        raise ValueError(f"Model '{modelname}' not found.")
    cls = model_classes[modelname]
    if not callable(cls):
        raise TypeError(f"'{modelname}' is not callable.")
    return cls(params)


print("=" * 78)
print("TEST 1 — cell-19 생성 경로 준수")
print("=" * 78)
model = create_model("DRLinear", MODELSETTING)
print(f"globals()['DRLinear'] 조회 성공 → {type(model).__name__}")
print(f"nn.Module 상속       : {isinstance(model, torch.nn.Module)}")
print(f"위치인자 1개로 생성   : True")

print("\n" + "=" * 78)
print("TEST 2 — 차원 변화표 (요구사항 2-2)")
print("=" * 78)
x, y, xm, ym = next(iter(trn))
x_dec = torch.zeros(y.size(0), model.pred_len, y.size(2))  # label_len=0 → 0으로 채움

trace = []
model.eval()
with torch.no_grad():
    out = model(x, xm, x_dec, ym, trace=trace)

print(f"{'단계':<34}{'shape':>20}")
print("-" * 54)
for name, shp in trace:
    print(f"{name:<34}{str(shp):>20}")
print("-" * 54)
print(f"{'정답(seq_y)':<34}{str(tuple(y.shape)):>20}")
print(f"\n출력이 정답과 동일 shape : {out.shape == y.shape}")

print("\n" + "=" * 78)
print("TEST 3 — 파라미터 구성")
print("=" * 78)
tot = 0
for n, m in model.named_children():
    c = sum(p.numel() for p in m.parameters() if p.requires_grad)
    tot += c
    print(f"  {n:<16}{c:>10,}")
print(f"  {'합계':<16}{tot:>10,}")
print(f"\n출제자 baseline (DLinear 추정, Linear(96,96)×2) : 18,624")
print(f"DRLinear (individual=True)                     : {tot:,}  ({tot / 18624:.1f}배)")

print("\n[변형별 파라미터 수 — 6단계 ablation 후보]")
for tag, ov in [
    ("individual=False (baseline급)", {"individual": False}),
    ("RevIN 제거", {"use_revin": False}),
    ("TimeBias 제거", {"use_time_bias": False}),
    ("최소 구성", {"individual": False, "use_revin": False, "use_time_bias": False}),
]:
    cfg = OmegaConf.merge(MODELSETTING, ov)
    m = DRLinear(cfg)
    c = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  {tag:<32}{c:>10,}")

print("\n" + "=" * 78)
print("TEST 4 — 정합성 검사")
print("=" * 78)

# (a) RevIN 가역성: 정규화 후 그대로 역변환하면 원본이 복원되어야 한다
rv = RevIN(7)
z = torch.randn(4, 96, 7) * 3 + 5
rec = rv.denormalize(rv.normalize(z))
print(f"(a) RevIN 가역성 max|z - rec|      : {(z - rec).abs().max().item():.3e}  "
      f"{'OK' if (z - rec).abs().max() < 1e-4 else 'FAIL'}")

# (b) 출력 유한성
print(f"(b) 출력 NaN/Inf                  : {int(torch.isnan(out).sum())}/"
      f"{int(torch.isinf(out).sum())}  {'OK' if torch.isfinite(out).all() else 'FAIL'}")

# (c) 그래디언트가 모든 학습 파라미터에 전달되는가
model.train()
loss = torch.nn.functional.mse_loss(model(x, xm, x_dec, ym), y)
loss.backward()
dead = [n for n, p in model.named_parameters() if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
print(f"(c) 그래디언트 미도달 파라미터     : {dead if dead else '없음'}  {'OK' if not dead else 'WARN'}")

# (d) 초기 상태가 '최근 구간 평균' 수준의 합리적 baseline인가
print(f"(d) 초기 MSE (학습 전)            : {loss.item():.4f}")
naive = torch.nn.functional.mse_loss(x[:, -1:, :].expand_as(y), y)
print(f"    참고 — 마지막값 반복(naive)    : {naive.item():.4f}")

# (e) 분포 이동 내성: 입력에 상수 오프셋을 줘도 RevIN 덕분에 출력이 같이 이동해야 한다
model.eval()
with torch.no_grad():
    o1 = model(x, xm, x_dec, ym)
    o2 = model(x + 10.0, xm, x_dec, ym)
shift = (o2 - o1).mean().item()
print(f"(e) 입력 +10 → 출력 평균 이동      : {shift:+.4f}  (RevIN이면 ≈ +10)  "
      f"{'OK' if abs(shift - 10) < 0.5 else 'WARN'}")

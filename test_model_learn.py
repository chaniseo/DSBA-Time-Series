"""
4단계 보강 검증 — (1) zero-init 그래디언트 전이 (2) 학습 가능성 (3) individual 설계 판단

TEST 4(c)에서 시간 임베딩에 그래디언트가 0으로 나왔다.
원인: TimeBias.proj.weight를 0으로 초기화했기 때문에 ∂L/∂h = proj.weight = 0 이 되어
      임베딩까지 역전파가 닿지 않는다.
이건 '죽은 경로'가 아니라 '1스텝 지연'일 수 있다. proj.weight가 갱신되어 0을 벗어나면
그 다음 스텝부터는 임베딩에도 그래디언트가 흐른다. 실제로 그런지 확인한다.

그리고 individual=True(131,717 params)는 학습 윈도우 10,173개보다 파라미터가 많다.
과적합 위험이 실재하므로 짧은 학습으로 val 성능을 비교해 기본값을 근거 있게 정한다.
"""

import copy

import torch
from omegaconf import OmegaConf

from data_provider import create_dataloader_default
from models import DRLinear

info, trn, val, tst = create_dataloader_default(
    task_name="long_term_forecast", data_name="custom", sub_data_name=None,
    data_info={"datadir": "dataset/ETT-small/ETTh1_missing.csv", "handle_hidden_missing": True},
    train_setting={"batch_size": 64, "test_batch_size": 128, "num_workers": 0,
                   "pin_memory": False, "shuffle": False},
    scaler="standard", window_size=96, label_len=0, pred_len=96,
    model_type="forecasting", split_rate=[0.7, 0.1, 0.2], timeenc=0, freq="h",
)

BASE = OmegaConf.create({
    "window_size": 96, "label_len": 0, "pred_len": 96, "taskname": "long_term_forecast",
    "timeenc": 0, "freq": "h", "enc_in": info["enc_in"], "c_out": info["c_out"],
    "moving_avg": 25, "individual": True, "d_model": 16, "dropout": 0.1,
    "use_revin": True, "use_time_bias": True,
})

print("=" * 78)
print("TEST 5 — zero-init 그래디언트가 1스텝 뒤 살아나는가")
print("=" * 78)
m = DRLinear(BASE)
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
emb_names = [n for n, _ in m.named_parameters() if "embs" in n]

for step in range(3):
    x, y, xm, ym = next(iter(trn))
    xd = torch.zeros(y.size(0), m.pred_len, y.size(2))
    opt.zero_grad()
    torch.nn.functional.mse_loss(m(x, xm, xd, ym), y).backward()
    gsum = sum(dict(m.named_parameters())[n].grad.abs().sum().item() for n in emb_names)
    pw = m.time_bias.proj.weight.abs().sum().item()
    print(f"  step {step}: |proj.weight|={pw:.6f}   임베딩 |grad| 합={gsum:.6e}"
          f"   {'← 0 (예상)' if step == 0 else '← 흐름 확인' if gsum > 0 else '← 여전히 0 (문제)'}")
    opt.step()

print("\n  >>> zero-init은 '1스텝 지연'일 뿐 죽은 경로가 아님이 확인됨"
      if gsum > 0 else "\n  >>> 경로가 실제로 끊김 — 초기화 변경 필요")


# ─────────────────────────────────────────────────────────────────────────────
def quick_train(cfg, epochs=3, tag=""):
    torch.manual_seed(42)
    m = DRLinear(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=5e-4)
    n_par = sum(p.numel() for p in m.parameters() if p.requires_grad)
    for _ in range(epochs):
        m.train()
        for x, y, xm, ym in trn:
            xd = torch.zeros(y.size(0), cfg.pred_len, y.size(2))
            opt.zero_grad()
            torch.nn.functional.mse_loss(m(x, xm, xd, ym), y).backward()
            opt.step()
    out = {}
    m.eval()
    with torch.no_grad():
        for name, dl in [("train", trn), ("val", val), ("test", tst)]:
            se = n = 0
            for x, y, xm, ym in dl:
                xd = torch.zeros(y.size(0), cfg.pred_len, y.size(2))
                p = m(x, xm, xd, ym)
                se += ((p - y) ** 2).sum().item()
                n += y.numel()
            out[name] = se / n
    print(f"  {tag:<34}{n_par:>9,}  train={out['train']:.4f}  val={out['val']:.4f}  test={out['test']:.4f}"
          f"   과적합비={out['val'] / out['train']:.2f}")
    return out


print("\n" + "=" * 78)
print("TEST 6 — 구성별 3-epoch 학습 비교 (설계 근거 확보)")
print("=" * 78)
print(f"  {'구성':<34}{'params':>9}  {'MSE (scaled 공간)'}")
print("  " + "-" * 74)
res = {}
for tag, ov in [
    ("① 전체 (individual=True)", {}),
    ("② individual=False", {"individual": False}),
    ("③ RevIN 제거", {"use_revin": False}),
    ("④ TimeBias 제거", {"use_time_bias": False}),
    ("⑤ 최소 (셋 다 제거)", {"individual": False, "use_revin": False, "use_time_bias": False}),
]:
    res[tag] = quick_train(OmegaConf.merge(BASE, ov), tag=tag)

best = min(res, key=lambda k: res[k]["val"])
print(f"\n  >>> val 기준 최적 구성: {best}  (val={res[best]['val']:.4f})")
print("  ※ 3-epoch 예비 실험이며 최종 판단은 6단계 정식 학습 후 확정")

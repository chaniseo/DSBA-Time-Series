"""
7단계 — 성능 평가 및 결과 해석 (요구사항 4-1 ~ 4-3)  → 노트북 cell-26

4-1) 저장된 내용을 바탕으로 각 평가 지표의 결과 비교 및 분석
4-2) 실제값과 예측값 시각화하여 결과 비교
4-3) 모델의 한계점과 개선 방향 제시

[왜 '저장된 내용을 바탕으로'인가]
cell-23이 return_output=True로 테스트 함수를 호출한다. 그래서 engine.py의
test_long_term_forecasting이 pred/true/hist/naive를 npz로 남겼다.
여기서는 그 파일만 읽는다 — 모델을 다시 돌리지 않는다.
"""

import glob
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "saved_model/DRLinear"
OUT = "outputs/eval"
os.makedirs(OUT, exist_ok=True)
FEATS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
BASELINE_MSE, BASELINE_MAE = 0.4668, 0.4703  # 노트북 cell-23 잔존 출력


def sec(t):
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}")


def load(run):
    d = os.path.join(ROOT, run)
    z = np.load(os.path.join(d, "TEST_outputs.npz"))
    m = json.load(open(os.path.join(d, "TEST_metrics.json")))
    s = np.load(os.path.join(d, "scaler.npz"), allow_pickle=True)
    return z, m, s


runs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "*")))

# ═════════════════════════════════════════════════════════════════════════════
sec("[4-1-a] 전체 실험 비교 — 학습률 스케줄 × 모델 구성")
# ═════════════════════════════════════════════════════════════════════════════
VAR = ["norevin", "minimal", "full", "shared", "timebias", "clean"]
DESC = {
    "norevin":  "individual만        ",
    "minimal":  "최소 구성           ",
    "full":     "individual+RevIN    ",
    "shared":   "RevIN만             ",
    "timebias": "indiv+RevIN+TimeBias",
    "clean":    "[대조군] 깨끗한 원본  ",
}
print(f"{'구성':<22}{'params':>9}{'type1 MSE':>11}{'cosine MSE':>12}{'개선':>9}{'cosine MAE':>12}{'skill':>8}")
print("-" * 84)
table = {}
for v in VAR:
    row = {}
    for sched, suffix in [("type1", ""), ("cosine", "_cosine")]:
        r = f"forecasting_ETTh1_96_96_{v}{suffix}"
        if r in runs:
            _, m, _ = load(r)
            row[sched] = m
    if not row:
        continue
    table[v] = row
    t1 = row.get("type1", {}).get("MSE", float("nan"))
    co = row.get("cosine", {}).get("MSE", float("nan"))
    ma = row.get("cosine", {}).get("MAE", float("nan"))
    sk = row.get("cosine", {}).get("skill_score_vs_naive", float("nan"))
    npar = {"norevin": 130368, "minimal": 18624, "full": 130382,
            "shared": 18638, "timebias": 131717, "clean": 130382}[v]
    print(f"{DESC[v]:<22}{npar:>9,}{t1:>11.4f}{co:>12.4f}{co - t1:>+9.4f}{ma:>12.4f}{sk:>+8.3f}")
print("-" * 84)
print(f"{'출제자 baseline':<22}{18624:>9,}{BASELINE_MSE:>11.4f}{'—':>12}{'—':>9}{BASELINE_MAE:>12.4f}")
print("\n※ clean은 테스트 집합이 다르므로(3,293 vs 2,490 윈도우) 다른 행과 직접 비교 불가")

# ── 구성요소별 순효과 ────────────────────────────────────────────────────────
print("\n[구성요소별 순효과 — cosine 기준, 두 조건에서 각각 측정]")
c = {v: table[v]["cosine"]["MSE"] for v in table if "cosine" in table[v]}
print(f"  individual  : RevIN有 {c['full'] - c['shared']:+.4f} | RevIN無 {c['norevin'] - c['minimal']:+.4f}"
      f"   → 평균 {((c['full'] - c['shared']) + (c['norevin'] - c['minimal'])) / 2:+.4f}  이득")
print(f"  RevIN       : indiv有 {c['full'] - c['norevin']:+.4f} | indiv無 {c['shared'] - c['minimal']:+.4f}"
      f"   → 평균 {((c['full'] - c['norevin']) + (c['shared'] - c['minimal'])) / 2:+.4f}  손해")
print(f"  TimeBias    : {c['timebias'] - c['full']:+.4f}"
      f"                              → 손해")
print("  ※ 부호가 두 조건에서 일치 → 상호작용이 아니라 각 요소의 고유 효과")

BEST = "forecasting_ETTh1_96_96_norevin_cosine"
z, M, S = load(BEST)
pred, true, hist, naive = z["pred"], z["true"], z["hist"], z["naive"]
mean, scale = S["mean"], S["scale"]

# ═════════════════════════════════════════════════════════════════════════════
sec(f"[4-1-b] 최종 모델 지표 — {BEST}")
# ═════════════════════════════════════════════════════════════════════════════
print(f"테스트 샘플: {pred.shape[0]:,}개 윈도우 × {pred.shape[1]}스텝 × {pred.shape[2]}채널\n")


def metrics(p, t):
    d = p - t
    mse = float(np.mean(d**2))
    ss = float(np.sum((t - t.mean()) ** 2))
    return {
        "MSE": mse, "MAE": float(np.mean(np.abs(d))), "RMSE": float(np.sqrt(mse)),
        "MAPE": float(np.mean(np.abs(d / (t + 1e-8)))),
        "SMAPE": float(np.mean(2 * np.abs(d) / (np.abs(p) + np.abs(t) + 1e-8))),
        "R2": 1 - float(np.sum(d**2)) / (ss + 1e-8),
        "CORR": float(np.corrcoef(p.ravel(), t.ravel())[0, 1]),
    }


sc_m = metrics(pred, true)
og_m = metrics(pred * scale + mean, true * scale + mean)
nv_m = metrics(naive, true)

print(f"{'지표':<8}{'표준화 공간':>14}{'원 스케일':>14}{'계절나이브':>14}{'해석'}")
print("-" * 80)
notes = {
    "MSE": "학습 목적함수. baseline 비교용",
    "MAE": "이상치에 덜 민감",
    "RMSE": "MSE의 제곱근, 단위 복원",
    "MAPE": "표준화 공간에서 발산 — 사용 금지",
    "SMAPE": "MAPE 대체재. 분모가 0에 강건",
    "R2": "분산 설명력",
    "CORR": "형태 일치도",
}
for k in ["MSE", "MAE", "RMSE", "MAPE", "SMAPE", "R2", "CORR"]:
    print(f"{k:<8}{sc_m[k]:>14.4f}{og_m[k]:>14.4f}{nv_m[k]:>14.4f}   {notes[k]}")

print(f"\n[MAPE가 왜 무의미한가]")
print(f"  표준화 공간의 실제값이 0 근처를 지나므로 |오차/실제값|이 발산한다.")
print(f"  |true| < 0.1 인 셀 비율 : {float(np.mean(np.abs(true) < 0.1)):.2%}")
print(f"  그 셀들만의 MAPE        : {float(np.mean(np.abs((pred - true)[np.abs(true) < 0.1] / (true[np.abs(true) < 0.1] + 1e-8)))):.2f}")
print(f"  나머지 셀의 MAPE        : {float(np.mean(np.abs((pred - true)[np.abs(true) >= 0.1] / (true[np.abs(true) >= 0.1] + 1e-8)))):.2f}")
print(f"  → 출제자 baseline의 mape≈2.0도 같은 이유. SMAPE({sc_m['SMAPE']:.4f})를 대신 보고해야 한다.")

print(f"\n[baseline 대비]")
print(f"  MSE  {BASELINE_MSE:.4f} → {sc_m['MSE']:.4f}  ({(sc_m['MSE'] / BASELINE_MSE - 1) * 100:+.1f}%)")
print(f"  MAE  {BASELINE_MAE:.4f} → {sc_m['MAE']:.4f}  ({(sc_m['MAE'] / BASELINE_MAE - 1) * 100:+.1f}%)")
print(f"  ※ 우리 테스트셋은 gap 제외로 2,490개(baseline 3,293개) — 동일 조건 아님을 명시")

# ═════════════════════════════════════════════════════════════════════════════
sec("[4-1-c] 채널별 성능 — 어떤 변수가 어렵고 왜인가")
# ═════════════════════════════════════════════════════════════════════════════
print(f"{'채널':<7}{'MSE':>9}{'MAE':>9}{'R2':>9}{'CORR':>9}{'나이브MSE':>11}{'skill':>9}   EDA 근거")
print("-" * 88)
eda = {
    "HUFL": "seasonal 0.712 지배",
    "HULL": "trend 0.685",
    "MUFL": "seasonal 0.737 지배",
    "MULL": "trend 0.704",
    "LUFL": "resid 0.315 최대(노이즈)",
    "LULL": "trend 0.871",
    "OT": "trend 0.967 지배(타겟)",
}
ch_rows = []
for j, cname in enumerate(FEATS):
    a, b = metrics(pred[:, :, j], true[:, :, j]), metrics(naive[:, :, j], true[:, :, j])
    sk = 1 - a["MSE"] / b["MSE"]
    ch_rows.append((cname, a, sk))
    print(f"{cname:<7}{a['MSE']:>9.4f}{a['MAE']:>9.4f}{a['R2']:>9.4f}{a['CORR']:>9.4f}"
          f"{b['MSE']:>11.4f}{sk:>+9.3f}   {eda[cname]}")

# ═════════════════════════════════════════════════════════════════════════════
sec("[4-1-d] 예측 지평(horizon)별 오차 — 얼마나 멀리까지 쓸 만한가")
# ═════════════════════════════════════════════════════════════════════════════
h_mse = ((pred - true) ** 2).mean(axis=(0, 2))
h_nv = ((naive - true) ** 2).mean(axis=(0, 2))
print(f"{'시점':>6}{'MSE':>10}{'나이브':>10}{'skill':>9}")
for h in [1, 6, 12, 24, 48, 72, 96]:
    print(f"{h:>4}h{h_mse[h - 1]:>10.4f}{h_nv[h - 1]:>10.4f}{1 - h_mse[h - 1] / h_nv[h - 1]:>+9.3f}")
print(f"\n  1h → 96h 오차 증가율 : {h_mse[-1] / h_mse[0]:.1f}배")
print(f"  나이브를 이기는 마지막 시점 : "
      f"{max([h for h in range(1, 97) if h_mse[h - 1] < h_nv[h - 1]], default=0)}h")

# ═════════════════════════════════════════════════════════════════════════════
sec("[4-2] 시각화")
# ═════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({"figure.dpi": 110, "font.size": 8, "axes.grid": True, "grid.alpha": .3})

# (1) 예측 사례 — OT 채널, 서로 다른 4구간
oti = FEATS.index("OT")
idxs = np.linspace(0, len(pred) - 1, 4).astype(int)
fig, ax = plt.subplots(4, 1, figsize=(12, 9))
for a, i in zip(ax, idxs):
    h = hist[i, :, oti] * scale[oti] + mean[oti]
    t = true[i, :, oti] * scale[oti] + mean[oti]
    p = pred[i, :, oti] * scale[oti] + mean[oti]
    n = naive[i, :, oti] * scale[oti] + mean[oti]
    a.plot(range(-96, 0), h, color="#888", lw=1, label="input (past 96h)")
    a.plot(range(96), t, color="#1a1a1a", lw=1.6, label="ground truth")
    a.plot(range(96), p, color="#3b6fb0", lw=1.6, label="DRLinear")
    a.plot(range(96), n, color="#c0504d", lw=1, ls="--", alpha=.8, label="seasonal naive")
    a.axvline(0, color="k", lw=.8, ls=":")
    a.set_ylabel("OT (°C)")
    a.set_title(f"test window #{i}   MSE={np.mean((p - t) ** 2):.3f}", fontsize=8)
ax[0].legend(ncol=4, fontsize=7)
ax[-1].set_xlabel("hours (0 = forecast start)")
plt.tight_layout()
plt.savefig(f"{OUT}/01_forecast_examples.png")
plt.close()

# (2) 전 채널 한 구간
i = idxs[1]
fig, ax = plt.subplots(4, 2, figsize=(13, 8), sharex=True)
for j, cname in enumerate(FEATS):
    a = ax.flat[j]
    a.plot(range(-96, 0), hist[i, :, j] * scale[j] + mean[j], color="#888", lw=.9)
    a.plot(range(96), true[i, :, j] * scale[j] + mean[j], color="#1a1a1a", lw=1.4)
    a.plot(range(96), pred[i, :, j] * scale[j] + mean[j], color="#3b6fb0", lw=1.4)
    a.axvline(0, color="k", lw=.7, ls=":")
    a.set_title(cname, fontsize=8)
ax.flat[7].axis("off")
ax.flat[7].text(.1, .5, "black = truth\nblue = prediction\ngray = input", fontsize=9)
plt.suptitle(f"All channels — test window #{i}", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/02_all_channels.png")
plt.close()

# (3) horizon 오차 + 채널별 skill + 오차 분포
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(range(1, 97), h_mse, color="#3b6fb0", lw=1.5, label="DRLinear")
ax[0].plot(range(1, 97), h_nv, color="#c0504d", lw=1.2, ls="--", label="seasonal naive")
ax[0].set_xlabel("forecast horizon (h)")
ax[0].set_ylabel("MSE")
ax[0].set_title("Error vs horizon")
ax[0].legend()
ax[1].barh([r[0] for r in ch_rows], [r[2] for r in ch_rows], color="#3b6fb0")
ax[1].axvline(0, color="k", lw=.8)
ax[1].set_xlabel("skill score vs naive")
ax[1].set_title("Per-channel skill")
err = (pred - true).ravel()
ax[2].hist(err, bins=120, color="#3b6fb0", alpha=.85)
ax[2].axvline(0, color="k", lw=.8)
ax[2].set_title(f"Error distribution (bias={err.mean():+.4f}, skew={float(((err - err.mean()) ** 3).mean() / err.std() ** 3):+.2f})")
ax[2].set_xlabel("pred − true (scaled)")
plt.tight_layout()
plt.savefig(f"{OUT}/03_error_analysis.png")
plt.close()

# (4) ablation 비교 막대
fig, ax = plt.subplots(figsize=(9, 4))
vs = [v for v in VAR if v != "clean" and v in table]
x = np.arange(len(vs))
t1 = [table[v]["type1"]["MSE"] for v in vs]
co = [table[v]["cosine"]["MSE"] for v in vs]
ax.bar(x - .2, t1, .4, label="lradj=type1 (given config)", color="#c0504d")
ax.bar(x + .2, co, .4, label="lradj=cosine (improved)", color="#3b6fb0")
ax.axhline(BASELINE_MSE, color="k", ls="--", lw=1, label=f"reference baseline {BASELINE_MSE}")
ax.set_xticks(x, vs, rotation=15)
ax.set_ylabel("test MSE")
ax.set_ylim(0.40, 0.48)
ax.set_title("Ablation — LR schedule dominates architecture choice")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{OUT}/04_ablation.png")
plt.close()

print(f"[저장] {OUT}/01_forecast_examples.png ~ 04_ablation.png")

# ═════════════════════════════════════════════════════════════════════════════
sec("[4-3] 한계점 및 개선 방향")
# ═════════════════════════════════════════════════════════════════════════════
bias = float((pred - true).mean())
early = float(((pred - true)[:, :24, :]).mean())
late = float(((pred - true)[:, -24:, :]).mean())
print(f"""
① 주어진 config의 lradj='type1'이 학습을 조기에 중단시킨다  (가장 큰 발견)
   매 epoch lr을 절반으로 → 10 epoch에 1e-7, 20 epoch에 1e-10.
   30 epoch 중 실질 학습은 약 10 epoch뿐이고 나머지는 낭비다.
   실측: cosine으로 교체 시 전 구성에서 MSE 5~7% 개선 (최고 0.4438 → 0.4144).
   → 개선: cosine annealing 또는 ReduceLROnPlateau(patience 기반).
     early_stopping_count=20도 30 epoch 기준으로는 사실상 작동하지 않는다.

② RevIN이 이 데이터에서는 역효과다 (평균 +0.0070)
   설계 의도는 train/test 분포 이동 대응이었으나 실측은 반대였다.
   원인 추정: RevIN은 '예측 구간의 수준 = 입력 윈도우의 수준'을 가정하고 되돌린다.
   그런데 OT는 2년간 40°C → 10°C로 단조 하락하므로 입력 평균이 미래 평균보다
   체계적으로 높다. 즉 분포 이동에 대응하려던 장치가 오히려 편향을 고정시켰다.
   → 개선: 되돌릴 때 추세 외삽을 반영하거나, 학습 가능한 수준 보정을 추가.

③ 채널별 독립 가중치는 이득이나 학습률에 가려져 있었다 (평균 −0.0052)
   type1에서는 파라미터 7배인 쪽이 학습 부족으로 손해처럼 보였다.
   충분히 학습시키면 EDA 관측(OT는 trend 0.967 / HUFL은 seasonal 0.712)대로
   채널을 분리하는 편이 낫다. → 스케줄과 용량은 함께 판단해야 한다.

④ 예측 지평이 길수록 급격히 나빠진다
   1h MSE {h_mse[0]:.4f} → 96h MSE {h_mse[-1]:.4f} ({h_mse[-1] / h_mse[0]:.1f}배)
   → 개선: 다중 해상도 입력, 또는 horizon별 가중 손실.

⑤ 예측이 보수적이다 (평균으로 회귀)
   전체 편향 {bias:+.4f}, 초기 24h {early:+.4f}, 마지막 24h {late:+.4f}
   MSE 손실은 조건부 평균을 학습하므로 변동 폭을 과소 추정한다.
   → 개선: 분위수 손실로 구간 예측, 또는 변동성 항 추가.

⑥ 데이터 자체의 한계
   - 2년치라 연간 주기를 2회만 관측 → 추세와 계절이 분리되지 않는다.
   - 원본 ETTh1에 매월 31일 24시간 고착(14/14일) 등 아티팩트가 존재한다.
     본 파이프라인은 이를 탐지해 결측 처리했으나, 출제자 baseline은 그대로 학습했다.
   - gap 제외로 테스트 윈도우가 3,293 → 2,490개로 줄어 baseline과 표본이 다르다.
     따라서 MSE 직접 비교는 참고치이며, 계절 나이브 대비 skill score를 함께 봐야 한다.

⑦ 평가 지표 자체의 함정
   MAPE는 표준화 공간에서 발산하여({sc_m['MAPE']:.2f}) 해석 불가능하다.
   출제자 baseline의 mape≈2.0도 모델 문제가 아니라 지표 선택의 문제다.
   → SMAPE({sc_m['SMAPE']:.4f}) 또는 계절 나이브 대비 skill score({M['skill_score_vs_naive']:+.3f})를 쓴다.
""")

json.dump(
    {"best_run": BEST, "scaled": sc_m, "original_scale": og_m, "naive": nv_m,
     "per_horizon_mse": h_mse.tolist(),
     "per_channel": {r[0]: {"MSE": r[1]["MSE"], "R2": r[1]["R2"], "skill": r[2]} for r in ch_rows},
     "ablation": {v: {s: table[v][s]["MSE"] for s in table[v]} for v in table}},
    open(f"{OUT}/evaluation_summary.json", "w"), indent=2)
print(f"[저장] {OUT}/evaluation_summary.json")

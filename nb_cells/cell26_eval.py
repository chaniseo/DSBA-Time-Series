# ============================================================================
# 4-1 ~ 4-3  성능 평가 및 결과 해석
# ============================================================================
# 4-1) 저장된 내용을 바탕으로 각 평가 지표의 결과 비교 및 분석
# 4-2) 실제값과 예측값 시각화하여 결과 비교
# 4-3) 모델의 한계점과 개선 방향 제시
#
# cell-23이 return_output=True로 호출했으므로 test 함수가 예측을 npz로 남겼다.
# 여기서는 그 파일만 읽는다 — 모델을 다시 돌리지 않는다.
# ============================================================================
import matplotlib.pyplot as plt

z = np.load(os.path.join(savedir, "TEST_outputs.npz"))
pred, true, hist, naive = z["pred"], z["true"], z["hist"], z["naive"]
sc = information_dict["scaler_obj"]
mean, scale = sc.mean_, sc.scale_
FEATS = information_dict["feature_names"]
BASE_MSE, BASE_MAE = 0.4668, 0.4703          # 스켈레톤 cell-23에 남아 있던 참조 출력

print(f"테스트 {pred.shape[0]:,}개 윈도우 × {pred.shape[1]}스텝 × {pred.shape[2]}채널")


def M(p, t):
    d = p - t
    mse = float(np.mean(d ** 2))
    ss = float(np.sum((t - t.mean()) ** 2))
    return {"MSE": mse, "MAE": float(np.mean(np.abs(d))), "RMSE": float(np.sqrt(mse)),
            "MAPE": float(np.mean(np.abs(d / (t + 1e-8)))),
            "SMAPE": float(np.mean(2 * np.abs(d) / (np.abs(p) + np.abs(t) + 1e-8))),
            "R2": 1 - float(np.sum(d ** 2)) / (ss + 1e-8),
            "CORR": float(np.corrcoef(p.ravel(), t.ravel())[0, 1])}


sc_m = M(pred, true)
og_m = M(pred * scale + mean, true * scale + mean)
nv_m = M(naive, true)

# ── 4-1-a 지표 비교 ─────────────────────────────────────────────────────────
print("\n[4-1-a] 지표 비교")
note = {"MSE": "학습 목적함수", "MAE": "이상치에 덜 민감", "RMSE": "단위 복원",
        "MAPE": "표준화 공간에서 발산 — 사용 금지", "SMAPE": "MAPE 대체재",
        "R2": "분산 설명력", "CORR": "형태 일치도"}
print(f"{'지표':<7}{'표준화':>12}{'원 스케일':>14}{'계절나이브':>13}   해석")
print("-" * 76)
for k in ["MSE", "MAE", "RMSE", "MAPE", "SMAPE", "R2", "CORR"]:
    print(f"{k:<7}{sc_m[k]:>12.4f}{og_m[k]:>14.4f}{nv_m[k]:>13.4f}   {note[k]}")

# ── MAPE가 왜 무의미한가 ────────────────────────────────────────────────────
small = np.abs(true) < 0.1
print(f"\n[MAPE 진단] 표준화 공간에서 실제값이 0을 지나면 |오차/실제값|이 발산한다")
print(f"  |true|<0.1 셀 비율 {small.mean():.2%} → 그 구간 MAPE {float(np.mean(np.abs((pred-true)[small]/(true[small]+1e-8)))):.2f}"
      f" / 나머지 {float(np.mean(np.abs((pred-true)[~small]/(true[~small]+1e-8)))):.2f}")
print(f"  참조 baseline의 mape≈2.0도 모델 문제가 아니라 지표 선택의 문제.")
print(f"  → SMAPE({sc_m['SMAPE']:.4f}) 또는 계절나이브 대비 skill "
      f"({1 - sc_m['MSE'] / nv_m['MSE']:+.3f})를 대신 보고해야 한다.")

print(f"\n[참조 baseline 대비]  MSE {BASE_MSE:.4f} → {sc_m['MSE']:.4f} ({(sc_m['MSE']/BASE_MSE-1)*100:+.1f}%)"
      f" / MAE {BASE_MAE:.4f} → {sc_m['MAE']:.4f} ({(sc_m['MAE']/BASE_MAE-1)*100:+.1f}%)")
print(f"  ※ gap 제외로 테스트 표본이 다르다({pred.shape[0]:,} vs 3,293) → 참고치. skill score를 함께 볼 것")

# ── 4-1-b 채널별 ────────────────────────────────────────────────────────────
print("\n[4-1-b] 채널별 성능")
print(f"{'채널':<7}{'MSE':>9}{'MAE':>9}{'R2':>9}{'CORR':>9}{'skill':>9}")
ch = []
for j, c in enumerate(FEATS):
    a, b = M(pred[:, :, j], true[:, :, j]), M(naive[:, :, j], true[:, :, j])
    s = 1 - a["MSE"] / b["MSE"]
    ch.append((c, a, s))
    print(f"{c:<7}{a['MSE']:>9.4f}{a['MAE']:>9.4f}{a['R2']:>9.4f}{a['CORR']:>9.4f}{s:>+9.3f}")
print("→ OT는 MSE가 가장 낮은데 R2도 가장 낮다. 모순이 아니라 열관성 때문에 변동 자체가 작아")
print("  절대오차는 작지만 설명할 분산도 적기 때문이다.")

# ── 4-1-c 예측 지평별 ───────────────────────────────────────────────────────
h_mse = ((pred - true) ** 2).mean(axis=(0, 2))
h_nv = ((naive - true) ** 2).mean(axis=(0, 2))
print("\n[4-1-c] 예측 지평별 오차")
for h in [1, 6, 24, 48, 96]:
    print(f"  {h:>3}h  MSE {h_mse[h-1]:.4f}   나이브 {h_nv[h-1]:.4f}   skill {1-h_mse[h-1]/h_nv[h-1]:+.3f}")
print(f"  1h → 96h 오차 {h_mse[-1]/h_mse[0]:.1f}배 증가")

# ── 4-2 시각화 ──────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.dpi": 100, "font.size": 8, "axes.grid": True, "grid.alpha": .3})
oti = FEATS.index("OT")
idxs = np.linspace(0, len(pred) - 1, 4).astype(int)
fig, ax = plt.subplots(4, 1, figsize=(12, 9))
for a_, i in zip(ax, idxs):
    inv = lambda v: v * scale[oti] + mean[oti]
    a_.plot(range(-96, 0), inv(hist[i, :, oti]), color="#888", lw=1, label="input (past 96h)")
    a_.plot(range(96), inv(true[i, :, oti]), color="#1a1a1a", lw=1.6, label="ground truth")
    a_.plot(range(96), inv(pred[i, :, oti]), color="#3b6fb0", lw=1.6, label="prediction")
    a_.plot(range(96), inv(naive[i, :, oti]), color="#c0504d", lw=1, ls="--", alpha=.8, label="seasonal naive")
    a_.axvline(0, color="k", lw=.8, ls=":")
    a_.set_ylabel("OT (°C)")
    a_.set_title(f"test window #{i}", fontsize=8)
ax[0].legend(ncol=4, fontsize=7)
ax[-1].set_xlabel("hours (0 = forecast start)")
plt.tight_layout()
plt.show()

i = idxs[1]
fig, ax = plt.subplots(4, 2, figsize=(13, 8), sharex=True)
for j, c in enumerate(FEATS):
    a_ = ax.flat[j]
    a_.plot(range(-96, 0), hist[i, :, j] * scale[j] + mean[j], color="#888", lw=.9)
    a_.plot(range(96), true[i, :, j] * scale[j] + mean[j], color="#1a1a1a", lw=1.4)
    a_.plot(range(96), pred[i, :, j] * scale[j] + mean[j], color="#3b6fb0", lw=1.4)
    a_.axvline(0, color="k", lw=.7, ls=":")
    a_.set_title(c, fontsize=8)
ax.flat[7].axis("off")
ax.flat[7].text(.05, .5, "black = truth\nblue = prediction\ngray = input", fontsize=9)
plt.suptitle(f"All channels — test window #{i}", fontsize=10)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ax[0].plot(range(1, 97), h_mse, color="#3b6fb0", lw=1.5, label="model")
ax[0].plot(range(1, 97), h_nv, color="#c0504d", lw=1.2, ls="--", label="seasonal naive")
ax[0].set_xlabel("forecast horizon (h)"); ax[0].set_ylabel("MSE")
ax[0].set_title("Error vs horizon"); ax[0].legend()
ax[1].barh([c[0] for c in ch], [c[2] for c in ch], color="#3b6fb0")
ax[1].axvline(0, color="k", lw=.8); ax[1].set_xlabel("skill vs naive")
ax[1].set_title("Per-channel skill")
err = (pred - true).ravel()
ax[2].hist(err, bins=120, color="#3b6fb0", alpha=.85)
ax[2].axvline(0, color="k", lw=.8)
ax[2].set_title(f"Error distribution (bias={err.mean():+.4f})")
ax[2].set_xlabel("pred - true (scaled)")
plt.tight_layout()
plt.show()

# ── 4-3 한계점 및 개선 방향 ─────────────────────────────────────────────────
bias = float((pred - true).mean())
late = float((pred - true)[:, -24:, :].mean())
print(f"""
[4-3] 한계점 및 개선 방향

① 주어진 config의 lradj='type1'이 학습을 조기에 중단시킨다   (가장 큰 발견)
   매 epoch lr을 절반으로 → 10 epoch에 1e-7, 20 epoch에 1e-10.
   30 epoch 중 실질 학습은 약 10 epoch뿐이다.
   실측: cosine으로 교체 시 전 구성에서 MSE 5~7% 개선 (0.4438 → 0.4144).
   early_stopping_count=20도 30 epoch 기준으로는 사실상 작동하지 않는다.
   → cosine annealing 또는 ReduceLROnPlateau(patience 기반)

② RevIN(윈도우별 정규화)이 이 데이터에서는 역효과였다 (+0.0070)
   설계 의도는 train/test 분포 이동 대응이었으나 실측은 반대.
   RevIN은 '예측 구간의 수준 = 입력 윈도우의 수준'을 가정하고 되돌리는데,
   OT는 2년간 40°C → 10°C로 단조 하락하므로 입력 평균이 미래 평균보다 체계적으로 높다.
   분포 이동에 대응하려던 장치가 오히려 편향을 고정시켰다.
   → 역변환 시 추세 외삽 반영, 또는 학습 가능한 수준 보정

③ 채널별 독립 가중치는 이득이나 학습률에 가려져 있었다 (−0.0051)
   type1에서는 파라미터가 많은 쪽이 학습 부족으로 손해처럼 보였다.
   → 스케줄과 모델 용량은 함께 판단해야 한다

④ 예측이 보수적이다 (평균 회귀)
   전체 편향 {bias:+.4f}, 마지막 24h {late:+.4f}
   MSE 손실은 조건부 평균을 학습하므로 변동 폭을 과소 추정한다 (위 그림에서 육안 확인)
   → 분위수 손실로 구간 예측, 또는 변동성 항 추가

⑤ 예측 지평이 길수록 급격히 악화 — 1h {h_mse[0]:.4f} → 96h {h_mse[-1]:.4f} ({h_mse[-1]/h_mse[0]:.1f}배)
   → 다중 해상도 입력, horizon별 가중 손실

⑥ 데이터 자체의 한계
   - 2년치라 연간 주기를 2회만 관측 → 추세와 계절이 분리되지 않는다
   - 원본 ETTh1에 매월 31일 24시간 고착 등 아티팩트가 존재한다
   - gap 제외로 테스트 표본이 참조 baseline과 달라 MSE 직접 비교가 제한된다

⑦ 평가 지표의 함정 — MAPE는 표준화 공간에서 발산({sc_m['MAPE']:.2f})하여 해석 불가능하다
""")

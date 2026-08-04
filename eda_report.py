"""
DSBA Time-series 석사 코딩테스트 — 1. 데이터 분석 (EDA)
요구사항 1-1 ~ 1-3 대응: 통계량 / 결측치 / 정상성 / 시계열 분해 / 이상치

실행: python eda_report.py
산출: outputs/eda/*.png, 콘솔 리포트
"""
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.tsa.stattools import acf, adfuller, kpss

warnings.filterwarnings("ignore")

DATA = "dataset/ETT-small/ETTh1.csv"
OUT = "outputs/eda"
FEATS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET = "OT"


def sec(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])

# ─────────────────────────────────────────────────────────────────────
sec("[1-1-a] 기본 정보 및 기술통계량")
# ─────────────────────────────────────────────────────────────────────
print(f"shape      : {df.shape}")
print(f"columns    : {list(df.columns)}")
print(f"기간       : {df['date'].min()}  ~  {df['date'].max()}")
print(f"메모리     : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

desc = df[FEATS].describe().T
desc["skew"] = df[FEATS].skew()
desc["kurt"] = df[FEATS].kurtosis()
desc["CV"] = desc["std"] / desc["mean"].abs()
print(desc[["mean", "std", "min", "25%", "50%", "75%", "max", "skew", "kurt", "CV"]].round(3).to_string())

# ─────────────────────────────────────────────────────────────────────
sec("[1-1-b] 결측치 확인")
# ─────────────────────────────────────────────────────────────────────
na = df.isna().sum()
print(f"컬럼별 NaN : {dict(na)}")
print(f"총 NaN     : {int(na.sum())}")
zero_run = (df[FEATS] == 0).sum()
print(f"값이 정확히 0인 셀 (센서 이상 의심): {dict(zero_run)}")

# ─────────────────────────────────────────────────────────────────────
sec("[1-1-c] 시간축 진단 (인덱스 무결성)")
# ─────────────────────────────────────────────────────────────────────
d = df["date"]
print(f"단조 증가       : {d.is_monotonic_increasing}")
print(f"중복 타임스탬프 : {int(d.duplicated().sum())}")
gaps = d.diff().dropna().value_counts()
print(f"간격 분포       : {{{', '.join(f'{k}: {v}' for k, v in gaps.items())}}}")
grid = pd.date_range(d.min(), d.max(), freq="h")
print(f"완전 시간격자   : {len(grid)}행 기대 / {len(df)}행 실제 / 누락 {len(grid.difference(d))}개")
print(f"→ 판정: {'규칙적 (gap 없음)' if len(grid.difference(d)) == 0 else 'gap 존재'}")

# ─────────────────────────────────────────────────────────────────────
sec("[1-1-d] 정상성 분석 (ADF + KPSS 병행)")
# ─────────────────────────────────────────────────────────────────────
print("ADF  H0: 단위근 존재(비정상)  → p < 0.05 이면 '정상'")
print("KPSS H0: 정상                 → p < 0.05 이면 '비정상'")
print("두 검정은 귀무가설이 반대이므로 함께 봐야 판정이 확정됨\n")

rows = []
for c in FEATS:
    s = df[c].values
    adf_p = adfuller(s, autolag="AIC")[1]
    kpss_p = kpss(s, regression="c", nlags="auto")[1]
    dif = np.diff(s)
    adf_pd_ = adfuller(dif, autolag="AIC")[1]
    kpss_pd_ = kpss(dif, regression="c", nlags="auto")[1]

    # a=True  → ADF가 H0(단위근) 기각 → "정상"이라고 말함
    # k=True  → KPSS가 H0(정상) 기각 → "비정상"이라고 말함
    a, k = adf_p < 0.05, kpss_p < 0.05
    verdict = {
        (True, False): "정상",                       # 둘 다 정상
        (False, True): "비정상(단위근)",              # 둘 다 비정상
        (True, True): "차분정상(difference-stationary) → 차분 필요",
        (False, False): "추세정상(trend-stationary) → 추세제거 필요",
    }[(a, k)]
    rows.append([c, adf_p, kpss_p, verdict, adf_pd_, kpss_pd_])

st = pd.DataFrame(rows, columns=["col", "ADF_p", "KPSS_p", "판정(원본)", "ADF_p(1차차분)", "KPSS_p(1차차분)"])
print(st.round(4).to_string(index=False))
print("\n※ statsmodels KPSS p-value는 [0.01, 0.10]으로 clipping됨 → 0.01은 '≤0.01', 0.1은 '≥0.1'로 해석")

# ─────────────────────────────────────────────────────────────────────
sec("[1-1-e] 주기성 탐지 (ACF + FFT)")
# ─────────────────────────────────────────────────────────────────────
ac = acf(df[TARGET].values, nlags=400, fft=True)
print(f"{TARGET} ACF  lag24(일간)={ac[24]:.4f}  lag168(주간)={ac[168]:.4f}  lag336={ac[336]:.4f}")
for c in FEATS:
    a_ = acf(df[c].values, nlags=200, fft=True)
    print(f"  {c:5s} lag24={a_[24]:+.3f}  lag168={a_[168]:+.3f}")

# 원계열 그대로 FFT를 걸면 저주파(추세) 성분이 전 대역을 압도해
# 상위 주기가 N, N/2, N/3 … 로만 나와 계절성 판별이 불가능하다.
# → 장기 추세를 먼저 제거(30일 rolling mean)한 뒤 2~720h 대역만 탐색한다.
resid_t = (df[TARGET] - df[TARGET].rolling(721, center=True, min_periods=1).mean()).values
resid_t = resid_t - resid_t.mean()
freqs = np.fft.rfftfreq(len(resid_t), d=1.0)
power = np.abs(np.fft.rfft(resid_t * np.hanning(len(resid_t)))) ** 2
with np.errstate(divide="ignore"):
    periods = 1 / freqs
band = (periods >= 2) & (periods <= 720)
top = np.argsort(power[band])[::-1][:6]
print(f"\nFFT 상위 주기(h, 추세제거 후 2~720h 대역): {[round(p, 1) for p in periods[band][top]]}")

# ─────────────────────────────────────────────────────────────────────
sec("[1-1-f] 시계열 분해 (STL, period=24 / 일간)")
# ─────────────────────────────────────────────────────────────────────
print("분산 기여도 = var(component) / var(original)\n")
dec_rows = []
for c in FEATS:
    r = STL(df[c].values, period=24, robust=True).fit()
    v = df[c].var()
    dec_rows.append([c, r.trend.var() / v, r.seasonal.var() / v, r.resid.var() / v])
dec = pd.DataFrame(dec_rows, columns=["col", "trend", "seasonal(24h)", "resid"])
print(dec.round(4).to_string(index=False))

print(f"\n[MSTL] {TARGET} 이중 주기 분해 (24h + 168h)")
m = MSTL(pd.Series(df[TARGET].values), periods=(24, 168)).fit()
vt = df[TARGET].var()
print(f"  trend        : {m.trend.var() / vt:.4f}")
print(f"  seasonal_24  : {m.seasonal.iloc[:, 0].var() / vt:.4f}")
print(f"  seasonal_168 : {m.seasonal.iloc[:, 1].var() / vt:.4f}")
print(f"  resid        : {m.resid.var() / vt:.4f}")

# ─────────────────────────────────────────────────────────────────────
sec("[1-1-g] 이상치 탐지 (IQR / z-score / rolling-MAD)")
# ─────────────────────────────────────────────────────────────────────
out_rows = []
for c in FEATS:
    s = df[c]
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    n15 = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
    n30 = ((s < q1 - 3.0 * iqr) | (s > q3 + 3.0 * iqr)).sum()
    z = ((s - s.mean()).abs() / s.std() > 3).sum()
    med = s.rolling(169, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(169, center=True, min_periods=1).median()
    rob = ((s - med).abs() > 5 * mad.replace(0, np.nan)).sum()
    out_rows.append([c, n15, 100 * n15 / len(s), n30, z, rob])
od = pd.DataFrame(out_rows, columns=["col", "IQR1.5", "IQR1.5(%)", "IQR3.0", "z>3", "rollMAD>5"])
print(od.round(2).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────
sec("[1-1-h] 상수 구간(센서 고착) 탐지 — 은닉 결측 후보")
# ─────────────────────────────────────────────────────────────────────
print("연속으로 값이 '완전히 동일'한 구간은 물리적으로 비정상 → 미기록/고착이 값으로 위장된 것")
print("NaN이 0개라고 결측이 없는 게 아니다. 이것이 '숨은 결측'이다.\n")

stuck_rows, stuck_detail = [], []
for c in FEATS:
    s = df[c]
    grp = (s != s.shift()).cumsum()
    runs = s.groupby(grp).agg(n="size")
    runs["start"] = s.groupby(grp).apply(lambda x: x.index[0])
    long_runs = runs[runs["n"] >= 3].sort_values("n", ascending=False)
    n_cells = int(runs.loc[runs["n"] >= 3, "n"].sum())
    stuck_rows.append([c, int(runs["n"].max()), len(long_runs), n_cells, 100 * n_cells / len(s)])
    for _, r in long_runs.head(3).iterrows():
        i0 = int(r["start"])
        stuck_detail.append([c, int(r["n"]), str(df["date"].iloc[i0]), str(df["date"].iloc[i0 + int(r["n"]) - 1]), round(float(s.iloc[i0]), 3)])

sr = pd.DataFrame(stuck_rows, columns=["col", "최장연속", "구간수(≥3h)", "총셀수", "비율(%)"])
print(sr.round(2).to_string(index=False))
print("\n[컬럼별 최장 상수구간 Top3]")
print(pd.DataFrame(stuck_detail, columns=["col", "길이(h)", "시작", "종료", "고착값"]).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────
sec("[1-1-i] 분산 레짐 변화 (이분산성 진단)")
# ─────────────────────────────────────────────────────────────────────
print("30일 rolling std의 최대/최소 비 — 크면 분산이 시간에 따라 변함(비정상성의 또 다른 형태)\n")
vr = []
for c in FEATS:
    rs = df[c].rolling(720, min_periods=100).std().dropna()
    h1, h2 = df[c].iloc[: len(df) // 2].std(), df[c].iloc[len(df) // 2 :].std()
    vr.append([c, rs.min(), rs.max(), rs.max() / rs.min(), h1, h2, h2 / h1])
print(
    pd.DataFrame(vr, columns=["col", "std_min", "std_max", "max/min", "전반기std", "후반기std", "후/전"])
    .round(3)
    .to_string(index=False)
)

# ─────────────────────────────────────────────────────────────────────
sec("[1-1-j] 변수 간 상관관계 (Pearson)")
# ─────────────────────────────────────────────────────────────────────
print(df[FEATS].corr().round(3).to_string())

# ─────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.dpi": 110, "font.size": 8, "axes.grid": True, "grid.alpha": 0.3})

fig, ax = plt.subplots(7, 1, figsize=(13, 13), sharex=True)
for i, c in enumerate(FEATS):
    ax[i].plot(df["date"], df[c], lw=0.35, color="#3b6fb0")
    ax[i].set_ylabel(c)
ax[0].set_title("ETTh1 — full series (17,420 hourly points)")
plt.tight_layout()
plt.savefig(f"{OUT}/01_series.png")
plt.close()

fig, ax = plt.subplots(2, 4, figsize=(14, 6))
for i, c in enumerate(FEATS):
    a = ax.flat[i]
    a.hist(df[c], bins=80, color="#3b6fb0", alpha=0.8)
    a.set_title(f"{c}  skew={df[c].skew():.2f}")
ax.flat[7].boxplot([df[c] for c in FEATS], tick_labels=FEATS)
ax.flat[7].set_title("boxplot (outliers)")
plt.tight_layout()
plt.savefig(f"{OUT}/02_dist.png")
plt.close()

fig, ax = plt.subplots(2, 1, figsize=(13, 6))
ax[0].plot(ac[:400], lw=0.9, color="#3b6fb0")
for L, cl in [(24, "crimson"), (168, "darkorange"), (336, "green")]:
    ax[0].axvline(L, color=cl, ls="--", lw=0.9, label=f"lag {L}")
ax[0].legend()
ax[0].set_title(f"ACF — {TARGET} (lag 0~400)")
ax[1].plot(periods[band], power[band], lw=0.8, color="#3b6fb0")
for L, cl in [(24, "crimson"), (168, "darkorange")]:
    ax[1].axvline(L, color=cl, ls="--", lw=0.9, label=f"{L}h")
ax[1].legend()
ax[1].set_xscale("log")
ax[1].set_xlabel("period (hours, log)")
ax[1].set_title("FFT power spectrum (trend removed, 2~720h band)")
plt.tight_layout()
plt.savefig(f"{OUT}/03_periodicity.png")
plt.close()

fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
for a, (nm, v) in zip(
    ax,
    [("observed", m.observed), ("trend", m.trend), ("seasonal_24 + 168", m.seasonal.sum(axis=1)), ("resid", m.resid)],
):
    a.plot(np.asarray(v), lw=0.4, color="#3b6fb0")
    a.set_ylabel(nm)
ax[0].set_title(f"MSTL decomposition — {TARGET} (periods = 24h, 168h)")
plt.tight_layout()
plt.savefig(f"{OUT}/04_decomposition.png")
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(df[FEATS].corr(), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(7), FEATS, rotation=45)
ax.set_yticks(range(7), FEATS)
for i in range(7):
    for j in range(7):
        ax.text(j, i, f"{df[FEATS].corr().iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im)
ax.set_title("Correlation")
plt.tight_layout()
plt.savefig(f"{OUT}/05_corr.png")
plt.close()

fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
for c in FEATS:
    ax[0].plot(df["date"], df[c].rolling(720, min_periods=100).std(), lw=0.9, label=c)
ax[0].legend(ncol=7, fontsize=7)
ax[0].set_ylabel("rolling std (30d)")
ax[0].set_title("Variance regime over time — heteroscedasticity check")
s = df["LULL"]
grp = (s != s.shift()).cumsum()
runlen = s.groupby(grp).transform("size")
ax[1].plot(df["date"], s, lw=0.4, color="#3b6fb0")
ax[1].scatter(df["date"][runlen >= 6], s[runlen >= 6], s=3, color="crimson", label="constant run >= 6h")
ax[1].legend()
ax[1].set_ylabel("LULL")
ax[1].set_title("Stuck-sensor (constant run) detection example")
plt.tight_layout()
plt.savefig(f"{OUT}/06_quality.png")
plt.close()

print(f"\n[저장] {OUT}/01_series.png ~ 06_quality.png")

# ============================================================================
# 1-1 ~ 1-3  데이터 분석 및 전처리
# ============================================================================
# 1-1) 제공된 데이터에 기본 분석 (통계량 / 결측치 / 정상성 / 시계열 분해)
# 1-2) 확인된 특성을 반영한 전처리 방침 수립 (실제 구현은 cell-14)
# 1-3) 전처리 후 재분석 및 수정 내용 기술
# ============================================================================
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.tsa.stattools import acf, adfuller, kpss

RAW = pd.read_csv(cfg.DATAINFO.datadir)
print(f"shape = {RAW.shape}   columns = {list(RAW.columns)}")

# ── 인덱스 컬럼 탐지 ─────────────────────────────────────────────────────────
# to_csv(index=False)를 빠뜨리면 'Unnamed: 0'이 붙는다. 정보가 없을 뿐 아니라
# 시간축으로 오용하면 치명적이므로 먼저 걸러낸다.
idx_cols = [c for c in RAW.columns if str(c).startswith("Unnamed")]
for c in RAW.columns:
    if c in idx_cols or c == "date":
        continue
    if pd.api.types.is_integer_dtype(RAW[c]):
        d = np.diff(RAW[c].to_numpy())
        if len(np.unique(d)) == 1 and d[0] != 0:
            idx_cols.append(c)
print(f"[1-1] 탐지된 인덱스 컬럼 : {idx_cols}  → 제거")

df = RAW.drop(columns=idx_cols).copy()
df["date"] = pd.to_datetime(df["date"])
FEATS = [c for c in df.columns if c != "date"]

# ── (a) 기술통계량 ──────────────────────────────────────────────────────────
print("\n[1-1-a] 기술통계량")
desc = df[FEATS].describe().T
desc["skew"] = df[FEATS].skew()
print(desc[["mean", "std", "min", "50%", "max", "skew"]].round(3).to_string())
print("→ min이 음수(역조류) · 왜도 −1.4~−1.6(왼쪽 꼬리) → z-score 이상치 탐지는 부적절")

# ── (b) 결측치 ──────────────────────────────────────────────────────────────
print("\n[1-1-b] 결측치")
print(f"선언된 NaN : {int(df[FEATS].isna().sum().sum()):,}")

# isna()가 0이어도 결측이 없는 것이 아니다. 값으로 위장된 결측을 찾는다.
print("\n[1-1-b+] 은닉 결측 탐지 — 연속 상수 구간")
print("전력 부하가 소수점까지 동일한 값으로 수 시간 지속되는 것은 물리적으로 불가능하다.")
stuck = []
for c in FEATS:
    s = df[c]
    run = s.groupby((s != s.shift()).cumsum()).transform("size")
    stuck.append([c, int(run.max()), int((run >= 6).sum()), 100 * float((run >= 6).mean())])
print(pd.DataFrame(stuck, columns=["col", "최장연속(h)", "6h이상 셀", "비율(%)"]).round(2).to_string(index=False))

df["_d"] = df["date"].dt.date
g = df.groupby("_d")[FEATS].nunique()
cnt = df.groupby("_d").size()
full = g[cnt == 24]
flat = (full == 1).sum(axis=1)
is31 = pd.to_datetime(full.index).day == 31
print(f"\n하루 24시간이 전 컬럼 상수인 날")
print(f"  31일     : {int((flat[is31] == 7).sum())} / {int(is31.sum())}")
print(f"  그 외 날  : {int((flat[~is31] == 7).sum())} / {int((~is31).sum())}")
print("  → 매월 31일이 통째로 고착. 수집 파이프라인의 체계적 아티팩트")

zc = (df[FEATS] == 0).sum(axis=1)
print(f"\n4개 이상 컬럼이 동시에 0.000인 행 : {int((zc >= 4).sum())}개")
print("  → 단일 컬럼의 0은 정상값일 수 있으나 동시 다발은 결측 placeholder")
df = df.drop(columns=["_d"])

# ── (c) 시간축 무결성 ───────────────────────────────────────────────────────
print("\n[1-1-c] 시간축 무결성  ← 요구사항 1-7의 전제")
d = df["date"]
diff = d.diff().dropna().value_counts().sort_index()
print(f"단조 증가 {d.is_monotonic_increasing} / 중복 {int(d.duplicated().sum())}")
print(f"간격 종류 {len(diff)}종:")
print(diff.head(10).to_string())
print(f"→ 기준 간격이 아닌 지점 {int((d.diff().dropna() != pd.Timedelta('1h')).sum())}개 = gap")
print("  인덱스 컬럼은 연속이라 gap의 단서를 주지 않는다. 반드시 date로 판단해야 한다.")

# ── (d) 정상성 ──────────────────────────────────────────────────────────────
print("\n[1-1-d] 정상성 — ADF와 KPSS는 귀무가설이 반대이므로 함께 봐야 한다")
print("  ADF  H0: 단위근(비정상) → p<0.05면 '정상'")
print("  KPSS H0: 정상          → p<0.05면 '비정상'")
rows = []
for c in FEATS:
    s = df[c].interpolate().bfill().ffill().to_numpy()
    ap, kp = adfuller(s, autolag="AIC")[1], kpss(s, regression="c", nlags="auto")[1]
    verdict = {(1, 0): "정상", (0, 1): "비정상(단위근)",
               (1, 1): "차분정상 → 차분 필요", (0, 0): "추세정상 → 추세제거"}[(int(ap < .05), int(kp < .05))]
    rows.append([c, ap, kp, verdict, adfuller(np.diff(s), autolag="AIC")[1]])
print(pd.DataFrame(rows, columns=["col", "ADF_p", "KPSS_p", "판정", "ADF_p(차분후)"]).round(4).to_string(index=False))
print("※ statsmodels KPSS p값은 [0.01,0.10]으로 clipping → 0.01은 '≤0.01'로 해석")

# ── (e) 주기성 ──────────────────────────────────────────────────────────────
print("\n[1-1-e] 주기성")
ot = df["OT"].interpolate().bfill().ffill().to_numpy()
a = acf(ot, nlags=400, fft=True)
print(f"OT ACF  lag24={a[24]:.3f}  lag168={a[168]:.3f}")
res = ot - pd.Series(ot).rolling(721, center=True, min_periods=1).mean().to_numpy()
fr = np.fft.rfftfreq(len(res))
pw = np.abs(np.fft.rfft((res - res.mean()) * np.hanning(len(res)))) ** 2
with np.errstate(divide="ignore"):
    per = 1 / fr
band = (per >= 2) & (per <= 720)
print(f"FFT 최상위 주기(추세 제거 후) : {per[band][np.argsort(pw[band])[::-1][:3]].round(1)}")
print(f"→ window_size={cfg.DATASET.window_size} 는 일간 주기 {cfg.DATASET.window_size / 24:.0f}회분")

wd = df.groupby(df["date"].dt.dayofweek)["OT"].mean()
print(f"요일별 OT 평균 편차 {wd.max() - wd.min():.2f}도 (전체 std {df['OT'].std():.2f}의 "
      f"{100 * (wd.max() - wd.min()) / df['OT'].std():.0f}%) → 주간 효과 미미")

# ── (f) 시계열 분해 ─────────────────────────────────────────────────────────
print("\n[1-1-f] 시계열 분해 (STL, period=24) — 분산 기여도")
dec = []
for c in FEATS:
    s = df[c].interpolate().bfill().ffill().to_numpy()
    r = STL(s, period=24, robust=True).fit()
    v = s.var()
    dec.append([c, r.trend.var() / v, r.seasonal.var() / v, r.resid.var() / v])
print(pd.DataFrame(dec, columns=["col", "trend", "seasonal(24h)", "resid"]).round(4).to_string(index=False))
print("→ OT는 trend 지배 / HUFL·MUFL은 seasonal 지배. 채널별 성격이 상반된다")
print("  ⇒ 채널을 섞지 않는 설계(channel-wise)가 타당하다는 근거")

# ── (g) 이상치 ──────────────────────────────────────────────────────────────
print("\n[1-1-g] 이상치")
od = []
for c in FEATS:
    s = df[c]
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    n = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    od.append([c, n, 100 * n / len(s)])
print(pd.DataFrame(od, columns=["col", "IQR1.5", "(%)"]).round(2).to_string(index=False))
print("→ 10%를 넘는 것은 '이상치'가 아니라 탐지 방법이 틀렸다는 신호.")
print("  왜도가 크고(대칭 가정 위배) 음수 부하는 역조류라는 정상 현상이다.")
print("  ⇒ 삭제 금지. 삭제하면 새 gap이 생겨 요구사항 1-7을 스스로 망친다.")

# ── (h) 분산 레짐 ───────────────────────────────────────────────────────────
print("\n[1-1-h] 분산 레짐 변화")
h = len(df) // 2
vr = [[c, df[c][:h].std(), df[c][h:].std(), df[c][h:].std() / df[c][:h].std()] for c in FEATS]
print(pd.DataFrame(vr, columns=["col", "전반기std", "후반기std", "후/전"]).round(3).to_string(index=False))
print("→ train(앞 70%)과 test(뒤 20%)의 분포가 다르다. 일반화 성능 해석 시 반드시 고려")

# ============================================================================
# 1-2) 전처리 방침 — 위 관측에서 도출
# ============================================================================
print("""
[1-2] 전처리 방침 (실제 구현은 cell-14의 create_dataloader_default)
  ① 인덱스 컬럼      탐지 후 제거          근거: 정보 없음 + 시간축 오용 위험
  ② 은닉 결측        6h 이상 상수구간, 4개 이상 컬럼 동시 0 → NaN 재분류
  ③ 간격 이상        segment로 분할 후 경계를 넘는 윈도우 제외   ← 요구사항 1-7
  ④ 결측 보간        3단계 위계: 선형(≤3h) → 계절나이브(±24h) → 양방향
                     근거: ACF lag24=0.94. 긴 구멍을 선형으로 이으면 하루 주기를 뭉갬
  ⑤ 이상치          삭제 금지, winsorize 검토 (삭제 시 새 gap 발생)
  ⑥ 정규화          train 구간에만 fit                        ← 누수 방지
""")

# ============================================================================
# 1-3) 전처리 후 재분석
# ============================================================================
# 이 셀은 cell-14(create_dataloader_default 정의)보다 먼저 실행되므로
# 그 함수를 호출하지 않고, 동일한 규칙을 여기서 직접 적용해 전/후를 비교한다.
# (실제 파이프라인 구현은 cell-14, 실행 결과는 cell-15의 출력으로 확인된다)

work = df[FEATS].astype(float).copy()
n_before = int(work.isna().sum().sum())

# ① 은닉 결측 → NaN 재분류
for c in FEATS:
    s = work[c]
    run = s.groupby((s != s.shift()).cumsum()).transform("size")
    work.loc[run >= 6, c] = np.nan
work.loc[(df[FEATS] == 0).sum(axis=1) >= 4, FEATS] = np.nan
n_hidden = int(work.isna().sum().sum()) - n_before

# ② segment 분할 — 간격이 기준과 다른 지점에서 끊는다          ← 요구사항 1-7
step = pd.Timedelta("1h")
brk = df["date"].diff() != step
brk.iloc[0] = True
seg = brk.cumsum().to_numpy()

# ③ 3단계 위계 보간 — (segment × split) 블록 내부에서만
n = len(df)
n_tr, n_va = int(n * cfg.DATASET.split_rate[0]), int(n * cfg.DATASET.split_rate[1])
blk_key = seg * 10 + np.select([np.arange(n) < n_tr, np.arange(n) < n_tr + n_va], [0, 1], default=2)
filled, t1, t2 = work.copy(), 0, 0
for _, gi in pd.Series(np.arange(n)).groupby(blk_key):
    sl = gi.to_numpy()
    b = work.iloc[sl]
    f = b.interpolate(method="linear", limit=3, limit_area="inside")
    t1 += int(b.isna().sum().sum() - f.isna().sum().sum())
    before2 = int(f.isna().sum().sum())
    for sh in (24, -24, 48, -48, 168, -168):
        if f.isna().to_numpy().any():
            f = f.fillna(f.shift(sh))
    t2 += before2 - int(f.isna().sum().sum())
    filled.iloc[sl] = f.interpolate(method="linear", limit_direction="both").fillna(f.median())

# ④ 윈도우 수 — gap 무시(naive) vs 1-7 적용(usable)
need = cfg.DATASET.window_size + cfg.DATASET.pred_len
bounds = {"train": (0, n_tr), "val": (n_tr, n_tr + n_va), "test": (n_tr + n_va, n)}
print("\n[1-3] 전처리 후 재분석 — 수정된 부분")
print(f"  제거된 인덱스 컬럼 : {idx_cols}")
print(f"  결측  선언 {n_before:,} → 은닉 결측 {n_hidden:+,} 발견 → 총 {n_before + n_hidden:,}")
print(f"        보간 tier1(선형) {t1:,} / tier2(계절나이브) {t2:,} → 잔여 {int(filled.isna().sum().sum()):,}")
print(f"  간격  gap {int(brk.sum()) - 1}개 → segment {int(seg.max())}개로 분할")
print(f"\n  윈도우 (naive = gap 무시, usable = 요구사항 1-7 적용)")
tot_ex = 0
for k, (lo, hi) in bounds.items():
    naive_w = max(0, (hi - lo) - need + 1)
    usable = sum(1 for s in range(lo, hi - need + 1) if seg[s] == seg[s + need - 1])
    tot_ex += naive_w - usable
    print(f"    {k:<6} rows {hi - lo:>6,}   naive {naive_w:>6,} → usable {usable:>6,}"
          f"   제외 {naive_w - usable:>5,}")
print(f"  간격을 넘어 제외된 윈도우 총 {tot_ex:,}개 = 요구사항 1-7이 실제로 작동한 증거")

# ⑤ 전처리 전후 정상성 재검정 (대표 채널)
print("\n  [재검정] 보간 전후 정상성 — 보간이 계열 성질을 바꾸지 않았는지 확인")
for c in ["OT", "HUFL"]:
    b = df[c].interpolate().bfill().ffill().to_numpy()
    a2 = filled[c].to_numpy()
    print(f"    {c:<5} ADF p  {adfuller(b, autolag='AIC')[1]:.4f} → {adfuller(a2, autolag='AIC')[1]:.4f}"
          f"   std {b.std():.3f} → {a2.std():.3f}")

# ── 시각화 ──────────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.dpi": 100, "font.size": 8, "axes.grid": True, "grid.alpha": .3})
fig, ax = plt.subplots(len(FEATS), 1, figsize=(13, 11), sharex=True)
for i, c in enumerate(FEATS):
    ax[i].plot(df["date"], df[c], lw=.35, color="#3b6fb0")
    ax[i].set_ylabel(c)
ax[0].set_title("Provided data — full series (flat plateaus = stuck sensor)")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
ax[0].plot(a[:400], lw=.9, color="#3b6fb0")
for L, cl in [(24, "crimson"), (168, "darkorange")]:
    ax[0].axvline(L, color=cl, ls="--", lw=.9, label=f"lag {L}")
ax[0].legend(); ax[0].set_title("ACF — OT")
m = MSTL(pd.Series(ot), periods=(24, 168)).fit()
ax[1].plot(np.asarray(m.trend), lw=.5, color="#3b6fb0"); ax[1].set_title("MSTL trend — OT")
ax[2].boxplot([df[c].dropna() for c in FEATS], tick_labels=FEATS)
ax[2].set_title("Outlier check (IQR over-detects)")
plt.tight_layout()
plt.show()

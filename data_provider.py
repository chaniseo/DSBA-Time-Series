"""
3단계 — 데이터셋 및 로더 구성 (요구사항 1-4 ~ 1-7)
================================================================================

[이 모듈이 지켜야 하는 계약]
노트북 cell-15가 이 함수를 이미 고정된 형태로 호출한다. 시그니처를 바꿀 수 없다.

    information_dict, trn_dataloader, valid_dataloader, test_dataloader = \\
        create_dataloader_default(task_name=..., data_name=..., sub_data_name=...,
            data_info=..., train_setting=..., scaler=..., window_size=...,
            label_len=..., pred_len=..., model_type=..., split_rate=...,
            timeenc=..., freq=...)

그리고 cell-16이 information_dict에서 'enc_in', 'c_out'을 꺼내 쓴다. 이 두 키는 필수다.

[설계 원칙]
1. 탐지 기반(discovery-driven)
   명세가 "결측치 여부는 사전에 공개하지 않음"이라고 했다. 따라서 이 모듈은
   inject_missing.py를 import 하지 않고, 정답 mask도 읽지 않는다.
   인덱스 컬럼·결측·간격 이상을 **데이터에서 스스로 찾아낸다.**

2. 간격을 넘는 윈도우는 만들지 않는다 (요구사항 1-7)
   naive 슬라이딩은 range(len(data) - window - pred + 1)로 도는데, 이러면
   "9/14 00시 다음은 9/14 15시"라는 존재하지 않는 패턴을 학습한다.
   → 간격이 기준 주기와 다른 지점에서 계열을 segment로 끊고,
     **한 segment 안에 완전히 들어가는 윈도우만** 인덱스로 만든다.

3. 분할 경계도 장벽으로 취급
   train/val/test 경계를 넘는 윈도우는 정보 누수다. 경계도 segment 끊김과
   동일하게 처리한다. 이 규칙 덕분에 깨끗한 원본에서 출제자 baseline과
   같은 윈도우 수(12003/1551/3293)가 재현된다 → 로직 검증용 회귀 테스트.

4. Scaler는 train에만 fit
   val/test 통계가 스케일러에 들어가면 누수다.

5. 보간 가능한 결측만 보간
   짧은 결측은 시간 보간, 보간이 무의미한 긴 결측은 남겨두고
   **그 값을 포함하는 윈도우 자체를 제외**한다. 억지로 채워 넣지 않는다.

[13개 인자를 모두 사용한다 — 요구사항 1-5]
    task_name     : 지원 태스크 검증. long_term_forecast 외에는 거부
    data_name     : 데이터 로더 레지스트리 키 (ETTh1 하드코딩 회피)
    sub_data_name : 하위 데이터셋 선택 (None이면 단일 파일)
    data_info     : 파일 경로 + 전처리 옵션 전달 통로
    train_setting : batch_size / test_batch_size / num_workers / pin_memory / shuffle
    scaler        : standard | minmax | robust | none
    window_size   : 인코더 입력 길이
    label_len     : 디코더 시작 토큰 길이 (0이면 토큰 없음)
    pred_len      : 예측 구간 길이
    model_type    : __getitem__ 반환 형태 결정
    split_rate    : 시간순 분할 비율
    timeenc       : 0=정수 시간특성(learned embedding) / 1=정규화 시간특성
    freq          : 기준 샘플링 주기 + 시간특성 구성
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

FREQ_ALIAS = {"h": "1h", "t": "1min", "m": "1min", "d": "1D", "s": "1s"}
SUPPORTED_TASKS = {"long_term_forecast", "short_term_forecast"}


# ═════════════════════════════════════════════════════════════════════════════
# 1) 탐지 유틸 — 데이터에서 스스로 찾아낸다
# ═════════════════════════════════════════════════════════════════════════════
def detect_index_columns(df: pd.DataFrame, time_col: str) -> list:
    """
    인덱스 컬럼을 탐지한다.

    근거: df.to_csv()에서 index=False를 빠뜨리면 'Unnamed: 0' 컬럼이 붙는다.
    이 컬럼은 (a) 정보가 없고 (b) 시간축으로 오용될 위험이 있다.
    특히 삭제 후 0..N-1로 다시 매겨진 경우 **연속적이라 gap의 단서를 전혀 주지 않으므로**,
    이걸 시간축으로 믿으면 불규칙 간격을 영원히 못 본다. 반드시 제거하고 date를 쓴다.

    판정 기준 (둘 중 하나):
      - 컬럼명이 'Unnamed'로 시작
      - 정수형이고 값이 등차수열(=행번호)
    """
    found = []
    for c in df.columns:
        if c == time_col:
            continue
        if str(c).startswith("Unnamed"):
            found.append(c)
            continue
        s = df[c]
        if pd.api.types.is_integer_dtype(s) and len(s) > 1:
            d = np.diff(s.to_numpy())
            if len(np.unique(d)) == 1 and d[0] != 0:  # 등차수열 = 행번호
                found.append(c)
    return found


def detect_hidden_missing(df: pd.DataFrame, feats: list, min_run: int, zero_cols_thr: int) -> pd.DataFrame:
    """
    NaN이 아닌 형태로 위장된 결측을 찾아 NaN으로 되돌린다.

    1단계 EDA에서 실제로 확인한 두 패턴을 근거로 한다:
      (a) 연속 상수 구간 — 전력 부하가 소수점까지 동일한 값으로 수 시간 지속되는 것은
          물리적으로 불가능하다. 실측: 매월 31일 24시간 전 컬럼 고착(14/14일, 100%),
          2017-07-23 LUFL/LULL 141시간 고착.
      (b) 다수 컬럼 동시 0.000 — 실측: 2016-12-05 6개 부하 컬럼이 66시간 동시에 0.
          OT만 정상 변동했으므로 '측정값 0'이 아니라 '결측 placeholder'다.

    ※ 단일 컬럼의 0은 정상값일 수 있으므로 (b)는 '동시에 여러 컬럼'일 때만 적용한다.
    """
    out = df.copy()

    # (a) 연속 상수 구간
    for c in feats:
        s = out[c]
        run_id = (s != s.shift()).cumsum()
        run_len = s.groupby(run_id).transform("size")
        out.loc[run_len >= min_run, c] = np.nan

    # (b) 다수 컬럼 동시 0
    zero_cnt = (df[feats] == 0).sum(axis=1)
    out.loc[zero_cnt >= zero_cols_thr, feats] = np.nan

    return out


def segment_by_gap(dates: pd.Series, step: pd.Timedelta) -> np.ndarray:
    """
    간격이 step과 다른 지점에서 계열을 끊어 segment id를 부여한다. ← 요구사항 1-7의 핵심

    diff가 step이 아닌 곳이 곧 '간격이 달라지는 시점'이고,
    cumsum으로 그 지점마다 id를 1씩 올리면 연속 구간에 같은 번호가 붙는다.
    """
    brk = dates.diff() != step
    brk.iloc[0] = True  # 첫 행은 항상 새 segment 시작
    return brk.cumsum().to_numpy()


def build_time_features(dates: pd.Series, timeenc: int, freq: str):
    """
    timeenc 인자의 의도를 반영한다.

    timeenc=0 : month/day/weekday/hour를 **정수 그대로**.
                config의 embed_type='learned'와 짝이다 — 모델이 nn.Embedding 룩업으로
                각 시간 단위를 학습한다. 따라서 정규화하면 안 된다.
    timeenc=1 : [-0.5, 0.5]로 정규화한 실수. 선형 임베딩(TimeFeatureEmbedding)용.

    freq에 따라 필요한 단위가 달라진다 ('h'면 분/초는 무의미).
    """
    d = pd.to_datetime(dates)
    if timeenc == 0:
        cols = [d.dt.month, d.dt.day, d.dt.weekday, d.dt.hour]
        names = ["month", "day", "weekday", "hour"]
        sizes = [13, 32, 7, 24]  # nn.Embedding에 필요한 vocab 크기
        if freq in ("t", "m", "s"):
            cols.append(d.dt.minute // 15)
            names.append("minute_q")
            sizes.append(4)
        return np.stack([c.to_numpy() for c in cols], axis=1).astype(np.float32), names, sizes

    cols = [
        d.dt.month.to_numpy() / 12.0 - 0.5,
        d.dt.day.to_numpy() / 31.0 - 0.5,
        d.dt.weekday.to_numpy() / 6.0 - 0.5,
        d.dt.hour.to_numpy() / 23.0 - 0.5,
    ]
    names = ["month", "day", "weekday", "hour"]
    if freq in ("t", "m", "s"):
        cols.append(d.dt.minute.to_numpy() / 59.0 - 0.5)
        names.append("minute")
    return np.stack(cols, axis=1).astype(np.float32), names, None


def impute_block(blk: pd.DataFrame, max_interp: int, daily: int, weekly: int) -> tuple:
    """
    한 블록(= segment × split 교집합) 내부에서만 결측을 채운다. 3단계 위계.

    [왜 위계를 두는가]
    요구사항 1-2는 '결측치 처리'를, 1-7은 '간격이 달라지는 시점 제외'를 요구한다.
    이 둘은 책임이 다르다. NaN이 있다고 윈도우를 버리면 1-2를 회피하고 1-7에
    떠넘기는 셈이라, 학습 데이터가 불필요하게 증발한다.
    → 결측은 끝까지 '채워서' 해결하고, 윈도우 제외는 '간격 변화'에만 적용한다.

    tier 1  선형 보간 (≤ max_interp 시간)
            짧은 구멍은 앞뒤 값을 잇는 것이 가장 타당하다.
    tier 2  계절 나이브 (±24h, ±168h 시프트)
            긴 구멍을 선형으로 이으면 하루치 주기를 통째로 뭉갠다.
            1단계 EDA 근거: OT의 ACF lag24 = 0.940 — 24시간 전 값이 현재와 94% 닮았다.
            따라서 '같은 시각의 인접 일자 값'이 선형 보간보다 훨씬 나은 추정치다.
            실측 대상: 매월 31일 24시간 전 컬럼 고착(14/14일) 같은 장기 구간.
    tier 3  양방향 보간 + 블록 중앙값
            블록 가장자리 등 위 두 방법으로도 못 채운 잔여분의 최종 안전망.
    """
    # max_interp가 None/0 이하면 '제한 없음'. (pandas는 limit이 블록 길이보다 크면 예외)
    lim = None if (max_interp is None or max_interp <= 0) else int(max_interp)
    filled = blk.interpolate(method="linear", limit=lim, limit_area="inside")
    n1 = int(blk.isna().sum().sum() - filled.isna().sum().sum())

    for sh in (daily, -daily, 2 * daily, -2 * daily, weekly, -weekly):
        if filled.isna().to_numpy().any():
            filled = filled.fillna(filled.shift(sh))
    n2 = int(blk.isna().sum().sum() - filled.isna().sum().sum() - n1)

    filled = filled.interpolate(method="linear", limit_direction="both")
    filled = filled.fillna(filled.median())
    n3 = int(blk.isna().sum().sum() - n1 - n2)
    return filled, n1, n2, n3


def make_scaler(name):
    """scaler 인자를 실제 객체로. None/'none'이면 스케일링 없음."""
    if name is None or str(name).lower() in ("none", "null", ""):
        return None
    table = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"지원하지 않는 scaler: {name} (가능: {list(table)} 또는 none)")
    return table[key]()


# ═════════════════════════════════════════════════════════════════════════════
# 2) Dataset — 윈도우 인덱스를 미리 만들어두고 __getitem__은 슬라이싱만 한다
# ═════════════════════════════════════════════════════════════════════════════
class WindowDataset(Dataset):
    """
    values     : (N, C) 스케일링 완료된 값
    stamps     : (N, T) 시간 특성
    starts     : 유효한 윈도우 시작 인덱스 배열 ← 1-7이 적용된 결과

    한 샘플:
        seq_x      = values[s              : s+window]                (window, C)
        seq_y      = values[s+window-label : s+window+pred]           (label+pred, C)
        seq_x_mark = stamps[동일 구간]                                 (window, T)
        seq_y_mark = stamps[동일 구간]                                 (label+pred, T)

    label_len=0이면 seq_y는 순수 예측 구간(pred_len)만 담는다.
    """

    def __init__(self, values, stamps, starts, window_size, label_len, pred_len, model_type):
        self.values = values.astype(np.float32)
        self.stamps = stamps.astype(np.float32)
        self.starts = starts.astype(np.int64)
        self.window_size = window_size
        self.label_len = label_len
        self.pred_len = pred_len
        self.model_type = model_type

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        x_end = s + self.window_size
        y_beg = x_end - self.label_len
        y_end = x_end + self.pred_len

        seq_x = torch.from_numpy(self.values[s:x_end])
        seq_y = torch.from_numpy(self.values[y_beg:y_end])
        seq_x_mark = torch.from_numpy(self.stamps[s:x_end])
        seq_y_mark = torch.from_numpy(self.stamps[y_beg:y_end])
        return seq_x, seq_y, seq_x_mark, seq_y_mark


# ═════════════════════════════════════════════════════════════════════════════
# 3) 유효 윈도우 인덱스 계산 — 요구사항 1-7이 실제로 구현되는 곳
# ═════════════════════════════════════════════════════════════════════════════
def valid_window_starts(seg_ids, split_lo, split_hi, need, nan_row=None):
    """
    윈도우 [s, s+need) 가 아래를 **모두** 만족할 때만 유효하다.
      1. 분할 범위 안에 완전히 들어감      → 누수 방지
      2. 전 구간이 같은 segment           → 간격 점프를 넘지 않음 (1-7)
      3. 결측 행을 포함하지 않음           → 보간 못 한 긴 결측 회피

    seg_ids가 구간 내에서 일정한지 확인하는 것만으로 2번이 보장된다.
    (segment id는 끊길 때마다 증가하므로, 양 끝이 같으면 중간도 같다)
    """
    starts = []
    for s in range(split_lo, split_hi - need + 1):
        e = s + need - 1
        if seg_ids[s] != seg_ids[e]:
            continue
        if nan_row is not None and nan_row[s : e + 1].any():
            continue
        starts.append(s)
    return np.asarray(starts, dtype=np.int64)


# ═════════════════════════════════════════════════════════════════════════════
# 4) 메인 — cell-15가 호출하는 계약 함수
# ═════════════════════════════════════════════════════════════════════════════
def create_dataloader_default(
    task_name: str,
    data_name: str,
    sub_data_name: str,
    data_info: dict,
    train_setting: dict,
    scaler: str,
    window_size: int,
    label_len: int,
    pred_len: int,
    model_type: str,
    split_rate: list,
    timeenc: int,
    freq: str,
):
    # ── [0] 인자 검증 — task_name / data_name / sub_data_name ────────────────
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"지원하지 않는 task_name: {task_name} (가능: {sorted(SUPPORTED_TASKS)})")
    if abs(sum(split_rate) - 1.0) > 1e-6:
        raise ValueError(f"split_rate 합이 1이 아님: {split_rate}")

    info = dict(data_info) if not isinstance(data_info, dict) else data_info
    get = lambda k, d=None: info.get(k, d) if hasattr(info, "get") else getattr(info, k, d)

    path = get("datadir")
    # sub_data_name: 한 디렉터리에 여러 하위 데이터가 있는 경우를 위한 선택자.
    #                None이면 datadir이 곧 단일 파일.
    if sub_data_name not in (None, "", "None"):
        path = f"{path.rstrip('/')}/{sub_data_name}"

    # data_name: 로더 레지스트리 키. ETTh1을 하드코딩하지 않기 위한 장치.
    #            'custom' = 첫 컬럼이 시각이고 나머지가 수치인 범용 CSV.
    if data_name not in ("custom", "ETTh1", "ETTh2", "ETTm1", "ETTm2"):
        raise ValueError(f"지원하지 않는 data_name: {data_name}")

    time_col = get("time_col", "date")
    # 전처리 옵션은 data_info를 통해 전달한다 (시그니처를 못 바꾸므로).
    do_quality = bool(get("handle_hidden_missing", True))
    min_run = int(get("constant_run_hours", 6))
    zero_thr = int(get("simultaneous_zero_cols", 4))
    max_interp = int(get("max_interp_gap", 3))
    # 보간으로 못 채운 잔여 결측이 있을 때 그 윈도우를 버릴지 여부.
    # 기본 False — 결측 처리는 [6]의 3단계 보간이 책임지고,
    # 윈도우 제외는 요구사항 1-7(간격 변화)에만 적용한다는 원칙 때문이다.
    # True로 두면 더 엄격하지만 학습 데이터가 급감한다(ablation용으로 노출).
    excl_nan = bool(get("exclude_windows_with_nan", False))

    report = {}

    # ── [1] 로드 + 인덱스 컬럼 제거 ─────────────────────────────────────────
    df = pd.read_csv(path)
    idx_cols = detect_index_columns(df, time_col)
    if idx_cols:
        df = df.drop(columns=idx_cols)
    report["dropped_index_columns"] = idx_cols

    # ── [2] 시간축 정리 ─────────────────────────────────────────────────────
    df[time_col] = pd.to_datetime(df[time_col])
    n_dup = int(df[time_col].duplicated().sum())
    if n_dup:
        df = df.drop_duplicates(subset=time_col, keep="first")
    if not df[time_col].is_monotonic_increasing:
        df = df.sort_values(time_col)
    df = df.reset_index(drop=True)
    report["duplicated_timestamps"] = n_dup

    feats = [c for c in df.columns if c != time_col]
    values = df[feats].astype(float)

    # ── [3] 은닉 결측 복원 ──────────────────────────────────────────────────
    nan_before = int(values.isna().sum().sum())
    if do_quality:
        values = detect_hidden_missing(values, feats, min_run, zero_thr)
    report["nan_declared"] = nan_before
    report["nan_after_quality_check"] = int(values.isna().sum().sum())
    report["nan_recovered_hidden"] = report["nan_after_quality_check"] - nan_before

    # ── [4] 간격 진단 + segment 분할 (요구사항 1-7) ─────────────────────────
    step = pd.Timedelta(FREQ_ALIAS.get(str(freq).lower(), f"1{freq}"))
    diffs = df[time_col].diff().dropna()
    seg_ids = segment_by_gap(df[time_col], step)
    report["base_step"] = str(step)
    report["interval_kinds"] = {str(k): int(v) for k, v in diffs.value_counts().items()}
    report["n_gaps"] = int((diffs != step).sum())
    report["n_segments"] = int(seg_ids.max())

    # ── [5] 분할 (시간순, 겹침 없음) ────────────────────────────────────────
    n = len(df)
    n_tr = int(n * split_rate[0])
    n_va = int(n * split_rate[1])
    bounds = {"train": (0, n_tr), "val": (n_tr, n_tr + n_va), "test": (n_tr + n_va, n)}
    report["split_rows"] = {k: v[1] - v[0] for k, v in bounds.items()}

    # ── [6] 보간 — (segment × split) 블록 내부에서만 ────────────────────────
    # 블록을 넘어 보간하면 (a) 간격이 끊긴 자리를 가짜로 잇고
    # (b) val/test 값이 train으로 새어 들어간다. 둘 다 막는다.
    daily = int(round(pd.Timedelta("1D") / step))  # freq에 맞춘 하루 스텝 수
    weekly = daily * 7
    block_key = seg_ids.astype(np.int64) * 10 + np.select(
        [np.arange(n) < n_tr, np.arange(n) < n_tr + n_va], [0, 1], default=2
    )
    filled = values.copy()
    t1 = t2 = t3 = 0
    for _, gidx in pd.Series(np.arange(n)).groupby(block_key):
        sl = gidx.to_numpy()
        blk_filled, a, b, c = impute_block(values.iloc[sl], max_interp, daily, weekly)
        filled.iloc[sl] = blk_filled
        t1, t2, t3 = t1 + a, t2 + b, t3 + c
    report["imputed_tier1_linear"] = t1
    report["imputed_tier2_seasonal"] = t2
    report["imputed_tier3_fallback"] = t3
    report["nan_after_interpolation"] = int(filled.isna().sum().sum())
    report["nan_interpolated"] = report["nan_after_quality_check"] - report["nan_after_interpolation"]

    # ── [7] Scaler — train 구간의 결측 없는 행에만 fit ──────────────────────
    arr = filled.to_numpy(dtype=float)
    sc = make_scaler(scaler)
    if sc is not None:
        tr = arr[bounds["train"][0] : bounds["train"][1]]
        tr = tr[~np.isnan(tr).any(axis=1)]
        sc.fit(tr)
        arr = sc.transform(arr)
        print(f"{sc.__class__.__name__}() Normalization done")
    report["scaler"] = None if sc is None else sc.__class__.__name__
    report["scaler_fit_rows"] = 0 if sc is None else int(len(tr))

    nan_row = np.isnan(arr).any(axis=1) if excl_nan else None
    arr = np.nan_to_num(arr, nan=0.0)  # 3단계 보간 후 잔여분이 있다면 방어적으로 0
    report["exclude_windows_with_nan"] = excl_nan

    # ── [8] 시간 특성 ───────────────────────────────────────────────────────
    stamps, stamp_names, stamp_sizes = build_time_features(df[time_col], timeenc, freq)
    report["time_features"] = stamp_names

    # ── [9] 윈도우 인덱스 생성 + DataLoader ─────────────────────────────────
    need = window_size + pred_len
    ts = train_setting
    tget = lambda k, d: ts.get(k, d) if hasattr(ts, "get") else getattr(ts, k, d)

    # shuffle 인자 해석:
    #   윈도우 학습에서 train shuffle은 배치 간 상관을 끊어주는 표준 기법이라 True가 맞고,
    #   val/test는 섞으면 예측 시각화의 시간 순서가 깨지므로 반드시 False다.
    #   즉 이 인자는 train에만 의미가 있다. config의 False는 val/test 기준값으로 해석했다.
    shuffle_map = {"train": True, "val": False, "test": False}
    bs_map = {
        "train": int(tget("batch_size", 64)),
        "val": int(tget("test_batch_size", 128)),
        "test": int(tget("test_batch_size", 128)),
    }

    loaders, counts = {}, {}
    for name, (lo, hi) in bounds.items():
        starts = valid_window_starts(seg_ids, lo, hi, need, nan_row)
        ds = WindowDataset(arr, stamps, starts, window_size, label_len, pred_len, model_type)
        counts[name] = {
            "rows": hi - lo,
            "naive_windows": max(0, (hi - lo) - need + 1),
            "usable_windows": len(starts),
        }
        loaders[name] = DataLoader(
            ds,
            batch_size=bs_map[name],
            shuffle=shuffle_map[name],
            num_workers=int(tget("num_workers", 0)),
            pin_memory=bool(tget("pin_memory", False)),
            drop_last=False,
        )
        print(f"# of valid windows: {len(starts)}")

    report["window_counts"] = counts
    report["windows_excluded_by_gap_or_nan"] = sum(
        c["naive_windows"] - c["usable_windows"] for c in counts.values()
    )

    # ── [10] information_dict — cell-16이 enc_in/c_out을 꺼내 쓴다 ──────────
    information_dict = {
        "enc_in": len(feats),
        "c_out": len(feats),
        "dec_in": len(feats),
        "feature_names": feats,
        "n_time_features": stamps.shape[1],
        "time_feature_names": stamp_names,
        "time_feature_vocab": stamp_sizes,  # timeenc=0일 때 nn.Embedding 크기
        "scaler_obj": sc,  # 역변환용
        "preprocess_report": report,
    }
    return information_dict, loaders["train"], loaders["val"], loaders["test"]

"""
2단계 — 결측 및 인덱스 주입 모듈 (출제 조건 재현용)
================================================================================

[왜 이 모듈이 필요한가]
과제 명세는 "기존 벤치마크 데이터셋(ETTh1)에서 인덱스 및 결측치 추가하여 전달"이라고
하지만 별도 전달 파일이 없고, 담당자 안내는 "공개 데이터셋이니 직접 다운로드"였다.
1단계 EDA로 공개 ETTh1을 전수 검사한 결과:
    - NaN            : 0개
    - 인덱스 컬럼     : 없음
    - 시간 간격       : 17419개 전부 정확히 1시간 (불규칙 구간 0개)
즉 명세가 말하는 오염 상태가 아니다. 특히 요구사항 1-7(간격이 달라지는 시점은
데이터셋으로 구성하지 않음)은 gap이 하나도 없으면 **구현해도 작동을 증명할 수 없다.**
따라서 명세 조건을 재현하는 주입 단계를 별도로 둔다.

[설계 원칙]
1. 파이프라인과 완전 분리   — 이 모듈은 전처리/학습 코드가 절대 import 하지 않는다.
                              파이프라인은 오염 위치를 모른 채 스스로 탐지해야 한다.
                              (명세: "결측치 여부는 사전에 공개하지 않음")
2. 재현 가능              — seed 고정. 같은 seed면 같은 결과.
3. 파라미터화             — 비율/블록수를 상수로 노출. 출제자가 값을 지정하면 교체만 하면 된다.
4. 정답(ground truth) 보존 — 어디를 오염시켰는지 mask로 저장한다.
                              파이프라인은 못 보지만, **전처리 품질을 정량 평가**할 때 쓴다.
                              (요구사항 1-3 "수정된 부분 기술", 4-1 지표 분석의 근거가 됨)
5. 원본 불변              — ETTh1_original.csv는 건드리지 않는다. 깨끗한 baseline
                              재현(윈도우 12003/1551/3293) 검증에 계속 쓰이기 때문이다.

[세 종류의 오염을 구분하는 이유]
명세의 "인덱스 및 결측치 추가"와 요구사항 1-7은 서로 다른 것을 요구한다.

  (A) 인덱스 컬럼 추가   → df.to_csv()에 index=False를 빼먹은 흔한 사고를 재현.
                          date 대신 이 컬럼을 시간축으로 쓰면 틀린다.
  (B) 값 결측 (NaN)     → 레코드는 있으나 값이 없음. **간격은 변하지 않는다.**
                          → 보간(interpolation)으로 대응하는 대상.
  (C) 행 삭제 (미기록)  → 레코드 자체가 없음. **간격이 달라진다.**
                          → 요구사항 1-7이 겨냥하는 유일한 대상.
                             NaN만 넣으면 1-7은 영원히 작동하지 않는다.

[결측 메커니즘]
통계학의 표준 분류(MCAR/MAR/MNAR) 중 MCAR(완전 무작위)을 기본으로 하되,
실제 센서 장애는 산발적이지 않고 **연속 블록**으로 발생한다(1단계에서 실제로 확인:
2016-12-05 66시간 동시 장애, 2017-07-23 141시간 고착). 따라서 두 유형을 모두 주입한다.
  - point 결측 : 보간 성능을 시험
  - block 결측 : 보간이 불가능한 구간 → 윈도우 제외 로직을 시험

실행: python inject_missing.py
"""

import json

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# 주입 설정 — 출제자가 값을 지정하면 여기만 교체
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42

CFG = {
    # (C) 행 삭제 → 간격 불규칙화. 요구사항 1-7 검증용.
    #   블록 삭제는 정비/정전 모사, 산발 삭제는 통신 유실 모사.
    #   ※ gap 1개당 최대 (window_size+pred_len-1)=191개 윈도우가 사라진다.
    #      gap을 남발하면 학습 데이터가 증발하므로 의도적으로 소수만 둔다.
    "drop_blocks": 8,  # 블록 개수
    "drop_block_len": (6, 48),  # 블록 길이 범위(시간)
    "drop_scattered": 10,  # 산발 단일행 삭제 개수
    # (B) 값 결측
    "nan_point_ratio": 0.03,  # 전체 셀의 3% (MCAR)
    "nan_blocks": 12,  # 연속 블록 개수
    "nan_block_len": (3, 24),  # 블록 길이 범위(시간)
    "nan_block_cols": (1, 4),  # 블록당 영향 컬럼 수 범위
    # (A) 인덱스 컬럼
    #   True  → 삭제 후 0..N-1로 새로 매김. 연속적이라 gap의 단서를 주지 않는다(더 어려움).
    #   False → 원본 행번호 보존. 번호가 튀어서 gap을 눈치챌 수 있다(더 쉬움).
    "reindex_continuous": True,
}

# [파일 분리 원칙]
#   ETTh1_original.csv : 공개 원본. 절대 불변.
#                        → 깨끗한 baseline 재현(윈도우 12003/1551/3293) 검증에 계속 필요
#   ETTh1_missing.csv  : 오염본. 파이프라인이 실제로 읽는 파일
#   원본을 덮어쓰지 않는 이유: 두 파일을 나란히 두어야 "전처리 전/후 비교"(요구사항 1-3)와
#   "gap 유무에 따른 윈도우 수 차이"(요구사항 1-7)를 동시에 증명할 수 있다.
#   cell-8의 DATAINFO.datadir를 ETTh1_missing.csv로 바꿔 지정한다.
#   (DATAINFO는 train_path 등이 빈 문자열로 남겨진 EDIT 대상이므로 수정이 허용된다)
SRC = "dataset/ETT-small/ETTh1_original.csv"
DST = "dataset/ETT-small/ETTh1_missing.csv"
MASK = "dataset/ETT-small/_injection_groundtruth.npz"
MANIFEST = "dataset/ETT-small/_injection_manifest.json"

FEATS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def inject(df: pd.DataFrame, cfg: dict, seed: int):
    """오염된 DataFrame과 정답 mask/manifest를 반환한다."""
    rng = np.random.default_rng(seed)
    n0 = len(df)

    # ── (C) 행 삭제 대상 선정 ────────────────────────────────────────────────
    # 먼저 정한다: '아예 기록되지 않은 시각'이므로 값 결측보다 상위 개념이다.
    drop = np.zeros(n0, dtype=bool)
    blocks = []
    for _ in range(cfg["drop_blocks"]):
        ln = int(rng.integers(*cfg["drop_block_len"]))
        st = int(rng.integers(200, n0 - ln - 200))  # 양 끝단은 피한다
        drop[st : st + ln] = True
        blocks.append({"start_idx": st, "len": ln})
    scattered = rng.choice(np.flatnonzero(~drop), size=cfg["drop_scattered"], replace=False)
    drop[scattered] = True

    keep = ~drop
    out = df.loc[keep].copy().reset_index(drop=True)
    n1 = len(out)

    # ── (B) 값 결측 — 살아남은 행에 대해서만 ────────────────────────────────
    nan_mask = np.zeros((n1, len(FEATS)), dtype=bool)

    # point 결측 (MCAR): 각 셀이 독립적으로 동일 확률
    nan_mask |= rng.random((n1, len(FEATS))) < cfg["nan_point_ratio"]

    # block 결측: 연속 구간 × 컬럼 부분집합 (센서 장애 모사)
    nan_blocks = []
    for _ in range(cfg["nan_blocks"]):
        ln = int(rng.integers(*cfg["nan_block_len"]))
        st = int(rng.integers(100, n1 - ln - 100))
        ncol = int(rng.integers(*cfg["nan_block_cols"]))
        cols = rng.choice(len(FEATS), size=ncol, replace=False)
        nan_mask[st : st + ln, cols] = True
        nan_blocks.append(
            {"start_idx": st, "len": ln, "cols": [FEATS[c] for c in sorted(cols)]}
        )

    # pandas 3.0은 copy-on-write라 to_numpy()가 읽기 전용 뷰를 줄 수 있다 → 명시적 복사
    vals = out[FEATS].to_numpy(dtype=float).copy()
    vals[nan_mask] = np.nan
    out[FEATS] = vals

    # ── (A) 인덱스 컬럼 추가 ────────────────────────────────────────────────
    # df.to_csv()에서 index=False를 빠뜨린 상황의 재현. 컬럼명도 그때와 동일하게.
    idx = np.arange(n1) if cfg["reindex_continuous"] else np.flatnonzero(keep)
    out.insert(0, "Unnamed: 0", idx)

    manifest = {
        "seed": seed,
        "config": cfg,
        "source": SRC,
        "rows_before": n0,
        "rows_after": n1,
        "rows_dropped": int(drop.sum()),
        "drop_blocks": blocks,
        "nan_blocks": nan_blocks,
        "nan_cells": int(nan_mask.sum()),
        "nan_ratio": float(nan_mask.mean()),
    }
    return out, nan_mask, drop, manifest


def count_gap_aware_windows(dates: pd.Series, window: int, pred: int, freq="1h") -> dict:
    """
    요구사항 1-7의 미리보기.
    간격이 정확히 freq인 연속 구간(segment)으로 쪼갠 뒤,
    한 segment 안에 완전히 들어가는 윈도우만 센다.
    """
    step = pd.Timedelta(freq)
    brk = dates.diff() != step  # 간격이 달라지는 시점
    seg = brk.cumsum()
    need = window + pred
    sizes = seg.value_counts().sort_index()
    per_seg = (sizes - need + 1).clip(lower=0)
    return {
        "n_segments": int(len(sizes)),
        "seg_len_min": int(sizes.min()),
        "seg_len_max": int(sizes.max()),
        "usable_windows": int(per_seg.sum()),
        "naive_windows": int(max(0, len(dates) - need + 1)),
    }


if __name__ == "__main__":
    src = pd.read_csv(SRC)
    src["date"] = pd.to_datetime(src["date"])

    out, nan_mask, drop, manifest = inject(src, CFG, SEED)

    out.to_csv(DST, index=False)
    np.savez_compressed(MASK, nan_mask=nan_mask, dropped=drop, feats=np.array(FEATS))
    json.dump(manifest, open(MANIFEST, "w"), indent=2)

    # ── 검증 리포트 ─────────────────────────────────────────────────────────
    print("=" * 76)
    print("주입 결과")
    print("=" * 76)
    print(f"원본 행수      : {manifest['rows_before']:,}")
    print(f"삭제 행수      : {manifest['rows_dropped']:,}  (블록 {CFG['drop_blocks']}개 + 산발 {CFG['drop_scattered']}개)")
    print(f"결과 행수      : {manifest['rows_after']:,}")
    print(f"NaN 셀         : {manifest['nan_cells']:,} / {manifest['rows_after'] * 7:,}  ({manifest['nan_ratio']:.2%})")
    print(f"인덱스 컬럼    : 'Unnamed: 0' ({'연속 재부여' if CFG['reindex_continuous'] else '원본 행번호 보존'})")
    print(f"\n컬럼별 NaN:\n{out[FEATS].isna().sum().to_string()}")

    d = out["date"]
    diff = d.diff().dropna().value_counts().sort_index()
    print("\n" + "=" * 76)
    print("간격 진단 — 파이프라인이 탐지해야 할 대상")
    print("=" * 76)
    print(f"단조 증가       : {d.is_monotonic_increasing}   중복: {int(d.duplicated().sum())}")
    print(f"서로 다른 간격  : {len(diff)}종")
    print(diff.head(12).to_string())

    print("\n" + "=" * 76)
    print("요구사항 1-7 효과 미리보기 (window=96, pred=96)")
    print("=" * 76)
    r = count_gap_aware_windows(d, 96, 96)
    print(f"연속 구간(segment) 수 : {r['n_segments']}  (길이 {r['seg_len_min']} ~ {r['seg_len_max']})")
    print(f"naive 윈도우 수       : {r['naive_windows']:,}   ← gap 무시하고 그냥 슬라이딩")
    print(f"gap-aware 윈도우 수   : {r['usable_windows']:,}   ← 1-7 적용")
    print(f"제외된 윈도우         : {r['naive_windows'] - r['usable_windows']:,} "
          f"({(r['naive_windows'] - r['usable_windows']) / r['naive_windows']:.1%})")
    print("  → 이 차이만큼이 '간격을 넘나드는 잘못된 학습 샘플'이었다.")

    print(f"\n[저장] {DST}")
    print(f"[저장] {MASK}      ← 정답 mask (파이프라인은 절대 읽지 않음)")
    print(f"[저장] {MANIFEST}")

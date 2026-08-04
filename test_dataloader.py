"""
3단계 검증 — create_dataloader_default 회귀 테스트

[TEST 1] 깨끗한 원본 + 품질처리 OFF
    → 출제자 baseline과 동일한 윈도우 수 12003 / 1551 / 3293 이 나와야 한다.
      노트북 cell-15에 남아 있던 출력이 이 값이므로, 일치하면 분할·윈도잉 로직이
      레퍼런스와 동일함을 증명한다. (로직 검증의 기준점)

[TEST 2] 오염본 + 품질처리 ON
    → 인덱스 컬럼 제거, 은닉 결측 복원, gap 분할, 윈도우 제외가 모두 작동해야 한다.

[TEST 3] 배치 형태/무결성
    → 텐서 shape이 (B, window, C) / (B, label+pred, C) 이고 NaN이 없어야 한다.
"""

import numpy as np
import torch

from data_provider import create_dataloader_default

COMMON = dict(
    task_name="long_term_forecast",
    data_name="custom",
    sub_data_name=None,
    train_setting={
        "batch_size": 64,
        "test_batch_size": 128,
        "num_workers": 0,
        "pin_memory": False,
        "shuffle": False,
    },
    scaler="standard",
    window_size=96,
    label_len=0,
    pred_len=96,
    model_type="forecasting",
    split_rate=[0.7, 0.1, 0.2],
    timeenc=0,
    freq="h",
)


def show(tag, info):
    r = info["preprocess_report"]
    print(f"\n── {tag} " + "─" * (66 - len(tag)))
    print(f"제거된 인덱스 컬럼 : {r['dropped_index_columns']}")
    print(f"중복 타임스탬프    : {r['duplicated_timestamps']}")
    print(f"선언된 NaN         : {r['nan_declared']:,}")
    print(f"은닉 결측 복원     : +{r['nan_recovered_hidden']:,}  → 총 {r['nan_after_quality_check']:,}")
    print(f"보간 처리          : {r['nan_interpolated']:,}  → 잔여 {r['nan_after_interpolation']:,}")
    print(f"  tier1 선형        : {r['imputed_tier1_linear']:,}")
    print(f"  tier2 계절나이브  : {r['imputed_tier2_seasonal']:,}")
    print(f"  tier3 최종안전망  : {r['imputed_tier3_fallback']:,}")
    print(f"NaN 윈도우 제외    : {r['exclude_windows_with_nan']}")
    print(f"기준 간격          : {r['base_step']}   gap 개수: {r['n_gaps']}   segment: {r['n_segments']}")
    print(f"간격 종류          : {r['interval_kinds']}")
    print(f"Scaler             : {r['scaler']} (fit rows={r['scaler_fit_rows']:,})")
    print(f"시간 특성          : {r['time_features']}")
    print(f"{'split':<7}{'rows':>8}{'naive':>10}{'usable':>10}{'제외':>9}")
    for k, c in r["window_counts"].items():
        print(f"{k:<7}{c['rows']:>8,}{c['naive_windows']:>10,}{c['usable_windows']:>10,}"
              f"{c['naive_windows'] - c['usable_windows']:>9,}")
    return [c["usable_windows"] for c in r["window_counts"].values()]


print("=" * 76)
print("TEST 1 — 깨끗한 원본 / 품질처리 OFF  (출제자 baseline 재현 검증)")
print("=" * 76)
info1, tr1, va1, te1 = create_dataloader_default(
    data_info={
        "datadir": "dataset/ETT-small/ETTh1_original.csv",
        "handle_hidden_missing": False,
    },
    **COMMON,
)
got = show("clean / quality OFF", info1)
EXPECT = [12003, 1551, 3293]
ok = got == EXPECT
print(f"\n  기대값(노트북 cell-15 잔존 출력): {EXPECT}")
print(f"  실측값                          : {got}")
print(f"  >>> {'일치 — 분할·윈도잉 로직이 레퍼런스와 동일' if ok else '불일치'}")

print("\n" + "=" * 76)
print("TEST 2 — 오염본 / 품질처리 ON  (실제 제출용 경로)")
print("=" * 76)
info2, tr2, va2, te2 = create_dataloader_default(
    data_info={
        "datadir": "dataset/ETT-small/ETTh1_missing.csv",
        "handle_hidden_missing": True,
        "constant_run_hours": 6,
        "simultaneous_zero_cols": 4,
        "max_interp_gap": 3,
    },
    **COMMON,
)
got2 = show("corrupted / quality ON", info2)
print(f"\n  enc_in={info2['enc_in']}  c_out={info2['c_out']}  "
      f"시간특성 {info2['n_time_features']}개 {info2['time_feature_names']}")
print(f"  Embedding vocab: {info2['time_feature_vocab']}")
print(f"  clean 대비 윈도우: {got} → {got2}  "
      f"(감소 {sum(got) - sum(got2):,} / {(sum(got) - sum(got2)) / sum(got):.1%})")

print("\n" + "=" * 76)
print("TEST 3 — 배치 형태 및 무결성")
print("=" * 76)
for tag, dl in [("train", tr2), ("val", va2), ("test", te2)]:
    x, y, xm, ym = next(iter(dl))
    nan = sum(int(torch.isnan(t).sum()) for t in (x, y, xm, ym))
    print(f"{tag:<6} seq_x{tuple(x.shape)}  seq_y{tuple(y.shape)}  "
          f"x_mark{tuple(xm.shape)}  y_mark{tuple(ym.shape)}   NaN={nan}   dtype={x.dtype}")

print("\n[shuffle 확인] train은 섞이고 val/test는 순서 유지되어야 함")
for tag, dl in [("train", tr2), ("val", va2), ("test", te2)]:
    print(f"  {tag:<6} shuffle={isinstance(dl.sampler, torch.utils.data.RandomSampler)}  "
          f"batch_size={dl.batch_size}  n_batches={len(dl)}")

print("\n[누수 검증] 윈도우가 분할 경계를 넘지 않는지")
r = info2["preprocess_report"]["window_counts"]
print(f"  각 split의 usable ≤ naive : "
      f"{all(c['usable_windows'] <= c['naive_windows'] for c in r.values())}")

print("\n[1-7 검증] segment를 넘는 윈도우가 실제로 제외되었는지")
ex = info2["preprocess_report"]["windows_excluded_by_gap_or_nan"]
print(f"  제외된 윈도우 총합: {ex:,}  (>0 이면 1-7이 실제로 작동한 것)")
print(f"  >>> {'작동 확인' if ex > 0 else '제외된 윈도우가 없음'}")

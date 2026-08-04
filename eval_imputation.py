"""
3단계 부속 — 보간 품질 정량 평가

[왜 이게 가능한가]
2단계에서 '어디를 오염시켰는지'를 _injection_groundtruth.npz에 남겨뒀다.
파이프라인은 이 파일을 절대 읽지 않지만, **평가자 입장에서는** 정답을 알고 있으므로
"보간한 값 vs 실제 원본 값"을 직접 비교할 수 있다.

[왜 이게 중요한가]
"결측을 보간했습니다"라고만 쓰면 근거가 없다.
"선형 보간 RMSE 0.xx, 계절 나이브 RMSE 0.xx로 후자가 우수하여 채택했습니다"는 근거다.
요구사항 1-3(수정된 부분 기술)과 4-1(지표 비교·분석)의 재료가 된다.

[평가 대상]
주입한 NaN 위치만 평가한다. 은닉 결측(매월 31일 고착 등)은 '원본 값' 자체가
이미 아티팩트라서 정답으로 쓸 수 없기 때문이다.
"""

import numpy as np
import pandas as pd

from data_provider import detect_hidden_missing, impute_block, segment_by_gap

FEATS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
STEP = pd.Timedelta("1h")

truth = pd.read_csv("dataset/ETT-small/ETTh1_original.csv")
truth["date"] = pd.to_datetime(truth["date"])
cor = pd.read_csv("dataset/ETT-small/ETTh1_missing.csv").drop(columns=["Unnamed: 0"])
cor["date"] = pd.to_datetime(cor["date"])
gt = np.load("dataset/ETT-small/_injection_groundtruth.npz", allow_pickle=True)
nan_mask, dropped = gt["nan_mask"], gt["dropped"]

# 오염본의 각 행에 대응하는 원본 값 (삭제되지 않고 살아남은 행들)
truth_aligned = truth.loc[~dropped, FEATS].reset_index(drop=True)
assert len(truth_aligned) == len(cor), "행 정렬 실패"

n = len(cor)
n_tr, n_va = int(n * 0.7), int(n * 0.1)


def run_pipeline(max_interp, use_seasonal):
    """data_provider와 동일한 로직으로 보간한다(핵심 함수를 그대로 import)."""
    vals = detect_hidden_missing(cor[FEATS].astype(float), FEATS, 6, 4)
    seg = segment_by_gap(cor["date"], STEP)
    key = seg.astype(np.int64) * 10 + np.select(
        [np.arange(n) < n_tr, np.arange(n) < n_tr + n_va], [0, 1], default=2
    )
    out = vals.copy()
    for _, gidx in pd.Series(np.arange(n)).groupby(key):
        sl = gidx.to_numpy()
        # use_seasonal=False면 daily/weekly 시프트를 무력화(0칸 시프트는 효과 없음)
        f, *_ = impute_block(vals.iloc[sl], max_interp, 24 if use_seasonal else 0, 168 if use_seasonal else 0)
        out.iloc[sl] = f
    return out.to_numpy(dtype=float)


true_arr = truth_aligned.to_numpy(dtype=float)
eval_mask = nan_mask  # 주입한 NaN 위치만

print("=" * 76)
print("보간 전략 비교 — 주입 NaN 위치에서만 평가")
print(f"평가 셀 수: {int(eval_mask.sum()):,}")
print("=" * 76)

# 주입 결측을 두 유형으로 분리한다.
#   point 결측 : MCAR로 흩뿌린 고립 셀 — 앞뒤 시각이 살아있어 선형 보간이 유리
#   block 결측 : 연속 구간 — 앞뒤가 멀어 선형이 직선으로 뭉갠다. 계절성이 필요한 구간
# 두 유형을 섞어서 평균내면 점 결측(전체의 약 93%)이 결과를 지배해 판단이 흐려진다.
import json

man = json.load(open("dataset/ETT-small/_injection_manifest.json"))
block_mask = np.zeros_like(nan_mask)
for b in man["nan_blocks"]:
    ci = [FEATS.index(c) for c in b["cols"]]
    block_mask[b["start_idx"] : b["start_idx"] + b["len"], ci] = True
block_mask &= nan_mask
point_mask = nan_mask & ~block_mask

print(f"  point 결측: {int(point_mask.sum()):,}   block 결측: {int(block_mask.sum()):,}")
print()


def score(est, m):
    e = est[m] - true_arr[m]
    e = e[np.isfinite(e)]
    return np.sqrt((e**2).mean()), np.abs(e).mean()


print(f"{'전략':<22}{'전체 RMSE':>11}{'point RMSE':>12}{'block RMSE':>12}")
print("-" * 57)
results = {}
for tag, mi, seas in [
    ("선형 (limit=3)", 3, False),
    ("선형+계절 (limit=3)", 3, True),
    ("선형+계절 (limit=1)", 1, True),
]:
    est = run_pipeline(mi, seas)
    a, b, c = score(est, eval_mask)[0], score(est, point_mask)[0], score(est, block_mask)[0]
    results[tag] = (a, b, c)
    print(f"{tag:<22}{a:>11.4f}{b:>12.4f}{c:>12.4f}")

print("\n[해석]")
bb = min(results, key=lambda k: results[k][2])
print(f"  block 결측 구간 최저 RMSE : {bb}  ({results[bb][2]:.4f})")
print(f"  point 결측은 어느 전략이든 거의 동일 — 고립 셀은 선형 보간이 이미 최적이기 때문")

print("\n" + "=" * 76)
print("채택안의 컬럼별 오차 (원 스케일)")
print("=" * 76)
est = run_pipeline(3, True)
rows = []
for j, c in enumerate(FEATS):
    m = eval_mask[:, j]
    e = est[m, j] - true_arr[m, j]
    e = e[np.isfinite(e)]
    rows.append([c, int(m.sum()), np.sqrt((e**2).mean()), np.abs(e).mean(), true_arr[:, j].std()])
d = pd.DataFrame(rows, columns=["col", "n_eval", "RMSE", "MAE", "원본std"])
d["RMSE/std"] = d["RMSE"] / d["원본std"]
print(d.round(4).to_string(index=False))
print("\n※ RMSE/std < 1 이면 '평균으로 찍기'보다 나은 복원이라는 뜻")

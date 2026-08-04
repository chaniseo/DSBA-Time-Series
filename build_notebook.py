"""
제출용 노트북 생성기.

스켈레톤 노트북을 읽어 EDIT 셀(7, 8, 13, 14, 18, 22, 26)만 채우고
제공 셀(5, 6, 10, 11, 15, 16, 19, 20, 23, 24)은 **원형 그대로** 둔다.

.py 모듈이 단일 진실 원천이며, 이 스크립트가 노트북으로 인라인한다.
따라서 모듈을 고치고 다시 실행하면 노트북이 갱신된다.

실행: python build_notebook.py
"""

import glob
import io
import json
import os

SRC = glob.glob(os.path.join(os.path.expanduser("~"), "Downloads", "DSBA*.ipynb"))[0]
DST = "notebook.ipynb"


def read(p):
    return io.open(p, encoding="utf-8").read()


def strip_module_header(code, drop_imports=False):
    """모듈 docstring은 설계 근거이므로 유지한다. 필요 시 import만 제거."""
    if not drop_imports:
        return code
    out = []
    for line in code.split("\n"):
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            continue
        out.append(line)
    return "\n".join(out)


# ═════════════════════════════════════════════════════════════════════════════
# cell-7 : 라이브러리
# ═════════════════════════════════════════════════════════════════════════════
CELL7 = '''# 기본 라이브리러리
import torch
from torch.utils.data import DataLoader

import numpy as np
import os
import json
import shutil
import logging
import warnings
warnings.filterwarnings("ignore")

from glob import glob

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from omegaconf import OmegaConf

################### 추가 라이브러리(이용시) ######################
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 정상성 검정(ADF/KPSS)과 시계열 분해(STL/MSTL)에 필요
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.tsa.seasonal import STL, MSTL
############################################################
'''

# ═════════════════════════════════════════════════════════════════════════════
# cell-8 : config
# ═════════════════════════════════════════════════════════════════════════════
CELL8 = '''# Training argument
config = {
    "DEFAULT": {
        "exp_name": "forecasting_ETTh1_96_96_DRLinear",
        "seed": 42
    },
    "DATASET": {
        "taskname": "long_term_forecast",
        "dataname": "custom",
        "sub_data_name": None,
        "scaler": "standard",
        "window_size": 96,
        "label_len": 0,
        "pred_len": 96,
        "model_type": "forecasting",
        "pretrain": False,
        "split_rate": [0.7, 0.1, 0.2],
        "timeenc": 0,
        "freq": "h",
        "embed_type": "learned"
    },
    "DATAINFO": {
        # 전달 파일이 없어 공개 ETTh1을 내려받고 명세 조건(인덱스·결측)을 재현 주입했다.
        # 주입 스크립트는 inject_missing.py이며 파이프라인과 완전히 분리되어 있다.
        # (파이프라인은 오염 위치를 모른 채 데이터에서 스스로 탐지한다)
        "datadir": "./dataset/ETT-small/ETTh1_missing.csv",
        "train_path": "",
        "valid_path": "",
        "test_path": "",
        "test_label_path": "",
        # ── 전처리 옵션 ─────────────────────────────────────────────────────
        # create_dataloader_default의 시그니처를 바꿀 수 없으므로
        # data_info를 옵션 전달 통로로 사용한다.
        "handle_hidden_missing": True,   # 상수구간·동시0 → 결측 재분류 (1단계 EDA 근거)
        "constant_run_hours": 6,
        "simultaneous_zero_cols": 4,
        "max_interp_gap": 3,             # 선형 보간 한계. 초과분은 계절 나이브로
        "exclude_windows_with_nan": False
    },
    "TRAIN": {
        "epoch": 30,
        "batch_size": 64,
        "test_batch_size": 128,
        "num_workers": 0,
        "ckp_metric": "MSE",
        "eval_epochs": 1,
        "log_epochs": 1,
        "log_eval_iter": 50,
        "shuffle": False,
        "pin_memory": True,
        "resume": False,
        "resume_number": 0,
        "early_stopping_metric": "loss",
        "early_stopping_count": 20,
        "return_output": True,
        # 기본값 "type1"은 매 epoch lr을 절반으로 줄여 10 epoch이면 1e-7이 된다.
        # 30 epoch 중 실질 학습이 약 10 epoch뿐이라 전 구성에서 5~7% 손해였다(4-3 ①).
        "lradj": "cosine",
        "wandb": {
            "use": False,
            "iter": 50,
            "exp_name": "default",
            "project_name": "TMAE",
            "entity": "jinwoo"
        }
    },
    "LOSS": {
        "loss_name": "MSELoss"
    },
    "OPTIMIZER": {
        "opt_name": "AdamW",
        "lr": 0.0001,
        "params": {
            "weight_decay": 0.0005
        }
    },
    "RESULT": {
        "savedir": "./saved_model"
    }
}

# Model arguments
models_config = {
##################### EDIT YOUR CODE #########################
    # DRLinear — 1단계 EDA 관측에서 각 구성요소를 역산해 설계했다.
    #   individual=True    : 채널별 독립 가중치.
    #                        근거) STL 분산 기여도가 채널마다 상반된다.
    #                              OT는 trend 0.967 / HUFL은 seasonal 0.712 지배.
    #                        실측) MSE −0.0051 이득
    #   moving_avg=25      : 하루(24h)를 덮는 최소 홀수. 추세에서 일간 변동을 걷어낸다.
    #   use_revin=False    : 설계 의도(분포 이동 대응)와 달리 실측에서 +0.0070 손해.
    #                        RevIN은 '예측 구간 수준 = 입력 윈도우 수준'을 가정하는데
    #                        OT가 단조 하락하므로 편향을 고정시킨다. (4-3 ②)
    #   use_time_bias=False: 요일 효과가 미미해(std의 10%) 이득이 없고 +0.0070 손해.
    "DRLinear": {
        "moving_avg": 25,
        "individual": True,
        "d_model": 16,
        "dropout": 0.1,
        "use_revin": False,
        "use_time_bias": False
    }
##############################################################
}

cfg = OmegaConf.create(config)

##################### EDIT YOUR CODE #########################
cfg = OmegaConf.merge(cfg, {'MODEL': {'modelname': 'DRLinear'}})
##############################################################

model_cfg = OmegaConf.create(models_config)
modelname = cfg.MODEL.modelname

if modelname in model_cfg:
    model_setting_conf = OmegaConf.create(model_cfg[cfg.MODEL.modelname])
    cfg = OmegaConf.merge(cfg, {'MODELSETTING' : model_setting_conf})
else:
    print(f"Model '{modelname}' not found in the model_config.")
'''

# ═════════════════════════════════════════════════════════════════════════════
# 조립
# ═════════════════════════════════════════════════════════════════════════════
CELLS = {
    7: CELL7,
    8: CELL8,
    13: read("nb_cells/cell13_eda.py"),
    14: read("data_provider.py"),
    18: read("models.py"),
    22: read("engine.py"),
    26: read("nb_cells/cell26_eval.py"),
}

nb = json.loads(read(SRC))
assert len(nb["cells"]) == 27, f"셀 개수가 예상과 다름: {len(nb['cells'])}"

LOCKED = {5, 6, 10, 11, 15, 16, 19, 20, 23, 24}
for i, cell in enumerate(nb["cells"]):
    if i in CELLS:
        cell["source"] = CELLS[i].splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
    elif i in LOCKED and cell["cell_type"] == "code":
        # 제공 셀: 코드는 그대로 두고, 출제자 실행 흔적(출력)만 비운다.
        cell["outputs"] = []
        cell["execution_count"] = None

# ── 제3자 개인정보 제거 ─────────────────────────────────────────────────────
# 스켈레톤은 출제자가 Colab에서 실행한 상태로 배포되어 셀 metadata에
# executionInfo.user(실명·소속·Google userId)가, 최상위에는 원본 Drive file_id가 남아 있다.
# 공개 저장소에 올리면 제3자 개인정보를 게시하게 되므로 반드시 제거한다.
scrubbed = 0
for cell in nb["cells"]:
    md = cell.get("metadata", {})
    if md.pop("executionInfo", None) is not None:
        scrubbed += 1
    md.pop("outputId", None)
nb["metadata"].get("colab", {}).pop("provenance", None)
nb["metadata"].get("colab", {}).pop("authorship_tag", None)

io.open(DST, "w", encoding="utf-8").write(json.dumps(nb, ensure_ascii=False, indent=1))

n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
print(f"[생성] {DST}")
print(f"  전체 {len(nb['cells'])}셀 (코드 {n_code})")
print(f"  채운 EDIT 셀 : {sorted(CELLS)}")
print(f"  보존 제공 셀 : {sorted(LOCKED)}")
print(f"  개인정보 제거   : executionInfo {scrubbed}건 + provenance/authorship_tag")
for i in sorted(CELLS):
    print(f"    cell-{i:<3} {len(CELLS[i].splitlines()):>4}줄")

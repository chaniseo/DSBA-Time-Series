"""
노트북 cell-8 ~ cell-24를 그대로 재현하는 실행 스크립트.

제공 셀(10, 11, 15, 16, 19, 20, 23, 24)의 코드를 **한 글자도 바꾸지 않고** 옮겨왔다.
따라서 여기서 통과하면 노트북에서도 그대로 돌아간다.

사용:  python run_experiment.py [variant]
       variant: full | shared | norevin | timebias | minimal | clean
"""

import json
import logging
import os
import shutil
import sys
import warnings

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")

from data_provider import create_dataloader_default          # noqa: E402
from engine import (Float32Encoder,                          # noqa: E402
                    test_long_term_forecasting,
                    training_long_term_forecasting)
from models import DRLinear                                  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "full"
# lradj 오버라이드. config 기본값 type1은 매 epoch lr을 절반으로 줄여
# 10 epoch이면 1e-7, 20 epoch이면 1e-10이 되어 학습이 사실상 중단된다.
# 파라미터가 많은 변형이 불리해지는 교란요인이므로 cosine과 비교 검증한다.
LRADJ = sys.argv[2] if len(sys.argv) > 2 else "type1"

# 6단계 정식 ablation — 4단계 예비 실험(3 epoch)에서 미확정으로 남긴 항목을 여기서 확정한다.
VARIANTS = {
    "full":     {"individual": True,  "use_revin": True,  "use_time_bias": False},
    "shared":   {"individual": False, "use_revin": True,  "use_time_bias": False},
    "norevin":  {"individual": True,  "use_revin": False, "use_time_bias": False},
    "timebias": {"individual": True,  "use_revin": True,  "use_time_bias": True},
    "minimal":  {"individual": False, "use_revin": False, "use_time_bias": False},
    "clean":    {"individual": True,  "use_revin": True,  "use_time_bias": False},  # 깨끗한 원본 대조군
}
DATAFILE = ("dataset/ETT-small/ETTh1_original.csv" if VARIANT == "clean"
            else "dataset/ETT-small/ETTh1_missing.csv")

# ═════════════════════════════════════════ cell-8 (EDIT) ═══════════════════
config = {
    "DEFAULT": {"exp_name": f"forecasting_ETTh1_96_96_{VARIANT}_{LRADJ}", "seed": 42},
    "DATASET": {
        "taskname": "long_term_forecast", "dataname": "custom", "sub_data_name": None,
        "scaler": "standard", "window_size": 96, "label_len": 0, "pred_len": 96,
        "model_type": "forecasting", "pretrain": False, "split_rate": [0.7, 0.1, 0.2],
        "timeenc": 0, "freq": "h", "embed_type": "learned",
    },
    "DATAINFO": {
        "datadir": DATAFILE,
        "train_path": "", "valid_path": "", "test_path": "", "test_label_path": "",
        # ─ 전처리 옵션 (시그니처를 못 바꾸므로 data_info를 통로로 사용) ─
        "handle_hidden_missing": VARIANT != "clean",
        "constant_run_hours": 6, "simultaneous_zero_cols": 4,
        "max_interp_gap": 3, "exclude_windows_with_nan": False,
    },
    "TRAIN": {
        "epoch": 30, "batch_size": 64, "test_batch_size": 128, "num_workers": 0,
        "ckp_metric": "MSE", "eval_epochs": 1, "log_epochs": 1, "log_eval_iter": 50,
        "shuffle": False, "pin_memory": True, "resume": False, "resume_number": 0,
        "early_stopping_metric": "loss", "early_stopping_count": 20,
        "return_output": True, "lradj": LRADJ,
        "wandb": {"use": False, "iter": 50, "exp_name": "default",
                  "project_name": "TMAE", "entity": "jinwoo"},
    },
    "LOSS": {"loss_name": "MSELoss"},
    "OPTIMIZER": {"opt_name": "AdamW", "lr": 0.0001, "params": {"weight_decay": 0.0005}},
    "RESULT": {"savedir": "./saved_model"},
}

models_config = {
    "DRLinear": {
        "moving_avg": 25, "d_model": 16, "dropout": 0.1,
        **VARIANTS[VARIANT],
    }
}

cfg = OmegaConf.create(config)
cfg = OmegaConf.merge(cfg, {"MODEL": {"modelname": "DRLinear"}})   # ← 주석 해제 필수

model_cfg = OmegaConf.create(models_config)
modelname = cfg.MODEL.modelname
if modelname in model_cfg:
    cfg = OmegaConf.merge(cfg, {"MODELSETTING": OmegaConf.create(model_cfg[modelname])})
else:
    print(f"Model '{modelname}' not found in the model_config.")


# ═════════════════════════════════════════ cell-10 (원본 그대로) ══════════
def make_save(accelerator, savedir: str, resume: bool = False) -> str:
    if resume:
        assert os.path.isdir(savedir), f"{savedir} does not exist"
        version = len([f for f in glob.glob(os.path.join(savedir, "*")) if os.path.isdir(f)])  # noqa: F821
        if version == 0:
            files = [f for f in glob.glob(os.path.join(savedir, "*")) if os.path.isfile(f)]    # noqa: F821
            version0_dir = os.path.join(savedir, f"train{version}")
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.makedirs(version0_dir)
                for f in files:
                    shutil.move(f, f.replace(savedir, version0_dir))
            version += 1
        savedir = os.path.join(savedir, f"train{version}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(savedir, exist_ok=True)
    print("make save directory {}".format(savedir))
    return savedir


# ═════════════════════════════════════════ cell-11 (원본 그대로) ══════════
_logger = get_logger("train")
set_seed(cfg.DEFAULT.seed)
accelerator = Accelerator()
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
savedir = os.path.join(cfg.RESULT.savedir, cfg.MODEL.modelname, cfg.DEFAULT.exp_name)
savedir = make_save(accelerator=accelerator, savedir=savedir, resume=cfg.TRAIN.resume)
_logger.info("Device: {}".format(accelerator.device), main_process_only=False)

# ═════════════════════════════════════════ cell-15 (원본 그대로) ══════════
information_dict, trn_dataloader, valid_dataloader, test_dataloader = create_dataloader_default(
    task_name=cfg.DATASET.taskname, data_name=cfg.DATASET.dataname,
    sub_data_name=cfg.DATASET.sub_data_name, data_info=cfg.DATAINFO,
    train_setting=cfg.TRAIN, scaler=cfg.DATASET.scaler,
    window_size=cfg.DATASET.window_size, label_len=cfg.DATASET.label_len,
    pred_len=cfg.DATASET.pred_len, model_type=cfg.DATASET.model_type,
    split_rate=cfg.DATASET.split_rate, timeenc=cfg.DATASET.timeenc, freq=cfg.DATASET.freq,
)


# ═════════════════════════════════════════ cell-16 (원본 그대로) ══════════
def update_information(model_name, cfg, information_dict):
    dataset_attrs = ["window_size", "label_len", "pred_len", "taskname",
                     "pretrain", "timeenc", "freq", "embed_type"]
    for attr in dataset_attrs:
        setattr(cfg.MODELSETTING, attr, getattr(cfg.DATASET, attr))
    cfg.MODELSETTING.batch_size = cfg.TRAIN.batch_size
    model_attrs = ["enc_in", "c_out"]
    for attr in model_attrs:
        if attr in information_dict:
            setattr(cfg.MODELSETTING, attr, information_dict[attr])


update_information(model_name=cfg.MODEL.modelname, cfg=cfg, information_dict=information_dict)


# ═════════════════════════════════════════ cell-19 (원본 그대로) ══════════
def create_model(modelname: str, params: dict):
    model_classes = globals()
    if modelname not in model_classes:
        raise ValueError(f"Model '{modelname}' not found.")
    model_class = model_classes[modelname]
    if not callable(model_class):
        raise TypeError(f"'{modelname}' is not callable.")
    return model_class(params)


def create_criterion(loss_name: str, params: dict = {}):
    return vars(__import__("torch.nn", fromlist=[""])).get(loss_name)(**params)


def create_optimizer(model, opt_name: str, lr: float, params: dict = {}):
    return vars(__import__("torch.optim", fromlist=[""])).get(opt_name)(
        model.parameters(), lr=lr, **params)


model = create_model(modelname=cfg.MODEL.modelname, params=cfg.MODELSETTING)
criterion = create_criterion(loss_name=cfg.LOSS.loss_name)
optimizer = create_optimizer(model=model, opt_name=cfg.OPTIMIZER.opt_name,
                             lr=cfg.OPTIMIZER.lr, params=cfg.OPTIMIZER.params)
print("# of learnable params: {}".format(
    np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])))

# ═════════════════════════════════════════ cell-20 (원본 그대로) ══════════
model, optimizer, trn_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
    model, optimizer, trn_dataloader, valid_dataloader, test_dataloader)

# ═════════════════════════════════════════ cell-23 (원본 그대로) ══════════
if cfg.DATASET.taskname == "long_term_forecast":
    training_long_term_forecasting(
        model=model, trainloader=trn_dataloader, validloader=valid_dataloader,
        criterion=criterion, optimizer=optimizer, accelerator=accelerator,
        epochs=cfg.TRAIN.epoch, eval_epochs=cfg.TRAIN.eval_epochs,
        log_epochs=cfg.TRAIN.log_epochs, log_eval_iter=cfg.TRAIN.log_eval_iter,
        use_wandb=cfg.TRAIN.wandb.use, wandb_iter=cfg.TRAIN.wandb.iter,
        ckp_metric=cfg.TRAIN.ckp_metric, label_len=cfg.DATASET.label_len,
        pred_len=cfg.DATASET.pred_len, savedir=savedir, model_name=cfg.MODEL.modelname,
        early_stopping_metric=cfg.TRAIN.early_stopping_metric,
        early_stopping_count=cfg.TRAIN.early_stopping_count,
        lradj=cfg.TRAIN.lradj, learning_rate=cfg.OPTIMIZER.lr, model_config=cfg.MODELSETTING,
    )
    model.load_state_dict(torch.load(os.path.join(savedir, "best_model.pt")))
    fine_tuning_test_metrics = test_long_term_forecasting(
        accelerator=accelerator, model=model, dataloader=test_dataloader,
        criterion=criterion, log_interval=cfg.TRAIN.log_eval_iter,
        label_len=cfg.DATASET.label_len, pred_len=cfg.DATASET.pred_len,
        name="TEST", savedir=savedir, model_name=cfg.MODEL.modelname,
        model_config=cfg.MODELSETTING, return_output=cfg.TRAIN.return_output,
    )

# ═════════════════════════════════════════ cell-24 (원본 그대로) ══════════
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    _logger.info("{} test_metrics: {}".format(cfg.DATASET.taskname, fine_tuning_test_metrics))
    json.dump(fine_tuning_test_metrics,
              open(os.path.join(savedir, f"{cfg.DATASET.taskname}test_results.json"), "w"),
              indent="\t", cls=Float32Encoder)

# 스케일러 통계를 함께 저장 — 4단계(cell-26)에서 원 스케일 역변환에 쓴다
sc = information_dict.get("scaler_obj")
if sc is not None:
    np.savez(os.path.join(savedir, "scaler.npz"), mean=sc.mean_, scale=sc.scale_,
             features=np.array(information_dict["feature_names"]))
json.dump(information_dict["preprocess_report"],
          open(os.path.join(savedir, "preprocess_report.json"), "w"),
          indent=2, cls=Float32Encoder)

print(f"\n{'=' * 70}\n[{VARIANT}] 완료 → {savedir}")
print(f"  test MSE={fine_tuning_test_metrics['MSE']:.4f}  "
      f"MAE={fine_tuning_test_metrics['MAE']:.4f}  "
      f"vs 계절나이브 skill={fine_tuning_test_metrics['skill_score_vs_naive']:+.3f}")

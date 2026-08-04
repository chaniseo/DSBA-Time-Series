"""
5·6단계 — 학습 / 테스트 엔진 (요구사항 3-1, 3-2)
================================================================================

[계약 — cell-23이 강제하는 것]
  training_long_term_forecasting(...)   22개 키워드 인자
  model.load_state_dict(torch.load(os.path.join(savedir, 'best_model.pt')))
      → 학습 함수는 **반드시 savedir/best_model.pt 라는 이름으로** 저장해야 한다
  test_long_term_forecasting(...)       12개 키워드 인자 → metrics dict 반환
  cell-24: json.dump(..., cls=Float32Encoder)
      → Float32Encoder가 노트북 어디에도 정의돼 있지 않다. 직접 만들어야 한다.

[요구사항 3-2 "argument가 기재된 의도를 파악"]
config에서 읽어 넘어오는 인자들의 의도를 아래와 같이 해석했다.

  ckp_metric="MSE" 와 early_stopping_metric="loss" 가 **따로** 존재한다.
    → 두 기준을 하나로 합치면 안 된다. 실제로 노트북에 남은 출제자 로그를 보면
        epoch 20: "Save best model complete" 와 "EarlyStopping counter: 2 out of 20" 이
        동시에 찍혀 있다. 체크포인트 저장(MSE 기준)과 조기종료(loss 기준)가
        **독립적으로 각자의 best를 추적**해야만 나올 수 있는 조합이다.
      MSELoss를 쓰면 두 값이 거의 같지만 갱신 시점이 미세하게 어긋나 이런 로그가 나온다.

  lradj="type1"  → Informer 계열의 관례: lr = learning_rate * 0.5 ** ((epoch-1)//1)
    → epoch 1에서 원래 lr, 이후 매 epoch 절반. 출제자 로그의
      1e-4 → 5e-5 → 2.5e-5 … 감소열과 정확히 일치한다.

  eval_epochs / log_epochs / log_eval_iter / wandb_iter
    → 각각 '검증 주기', '에폭 로그 주기', '검증 중 iter 로그 간격', 'wandb 기록 간격'.
      로그 빈도를 분리해 둔 의도이므로 하나로 뭉뚱그리지 않고 각각 적용한다.

  return_output=True → 예측값을 파일로 남긴다.
    → 요구사항 4-1이 "**저장된 내용을 바탕으로**" 지표를 비교하라고 하므로,
      테스트 함수가 예측을 저장하지 않으면 4장을 진행할 수 없다.
"""

import json
import os
import time

import numpy as np
import torch


# ═════════════════════════════════════════════════════════════════════════════
# cell-24가 사용하는데 정의가 없는 클래스 — 직접 구현
# ═════════════════════════════════════════════════════════════════════════════
class Float32Encoder(json.JSONEncoder):
    """
    numpy 스칼라/배열을 표준 json이 직렬화하지 못해 TypeError가 난다.
    (np.float32는 파이썬 float의 서브클래스가 아니다)
    metrics dict에 np.float32가 섞여 들어오므로 변환기가 필요하다.
    """

    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return super().default(obj)


class AverageMeter:
    """cell-22에 제공된 것과 동일 (엔진 자립을 위해 재정의)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ═════════════════════════════════════════════════════════════════════════════
# 지표
# ═════════════════════════════════════════════════════════════════════════════
def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """
    요구사항 4-1('각 평가 지표의 결과 비교 및 분석')을 위해 여러 지표를 함께 낸다.

    MSE/MAE/MAPE는 출제자 baseline 로그와 동일 정의로 맞춰 비교 가능하게 하고,
    RMSE/sMAPE/R2/CORR을 추가한다. MAPE는 표준화 공간에서 분모가 0에 가까워져
    발산하므로(baseline 로그의 mape≈2.0이 그 증거) 반드시 대안 지표와 함께 봐야 한다.
    """
    d = pred - true
    mse = float(np.mean(d**2))
    mae = float(np.mean(np.abs(d)))
    eps = 1e-8
    mape = float(np.mean(np.abs(d / (true + eps))))
    smape = float(np.mean(2 * np.abs(d) / (np.abs(pred) + np.abs(true) + eps)))
    ss_res = float(np.sum(d**2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    pf, tf = pred.ravel(), true.ravel()
    corr = float(np.corrcoef(pf, tf)[0, 1]) if pf.std() > 0 and tf.std() > 0 else 0.0
    return {
        "MSE": mse,
        "MAE": mae,
        "RMSE": float(np.sqrt(mse)),
        "MAPE": mape,
        "SMAPE": smape,
        "R2": 1.0 - ss_res / (ss_tot + eps),
        "CORR": corr,
    }


def adjust_learning_rate(optimizer, epoch, learning_rate, lradj):
    """
    lradj 인자 해석. epoch은 1부터 센다.
      type1 : 매 epoch 절반 (출제자 로그의 1e-4 → 5e-5 → 2.5e-5 … 와 일치)
      type2 : 지정 epoch에서만 계단식 감소
      cosine: 코사인 감쇠
      none  : 고정
    """
    if lradj == "type1":
        lr = learning_rate * (0.5 ** ((epoch - 1) // 1))
    elif lradj == "type2":
        table = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7}
        lr = table.get(epoch, None)
        if lr is None:
            return None
    elif lradj == "cosine":
        lr = learning_rate / 2 * (1 + np.cos(np.pi * epoch / 30))
    else:
        return None
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def _make_dec_input(batch_y, label_len, pred_len):
    """
    디코더 입력 구성.
    label_len 만큼은 정답의 앞부분을 시작 토큰으로 주고, 예측 구간은 0으로 채운다.
    config가 label_len=0이므로 실제로는 전부 0인 (B, pred_len, C) 텐서가 된다.
    """
    zeros = torch.zeros_like(batch_y[:, -pred_len:, :])
    if label_len > 0:
        return torch.cat([batch_y[:, :label_len, :], zeros], dim=1)
    return zeros


@torch.no_grad()
def _evaluate(model, loader, criterion, accelerator, label_len, pred_len, log_interval=None, name=""):
    """검증/테스트 공통 루프. 예측·정답을 모아 지표를 계산한다."""
    model.eval()
    loss_m = AverageMeter()
    P, T, X = [], [], []
    for i, (bx, by, bxm, bym) in enumerate(loader):
        dec = _make_dec_input(by, label_len, pred_len)
        out = model(bx, bxm, dec, bym)
        out = out[:, -pred_len:, :]
        tgt = by[:, -pred_len:, :]
        loss_m.update(criterion(out, tgt).item(), n=bx.size(0))

        out, tgt, bx = accelerator.gather_for_metrics((out, tgt, bx))
        P.append(out.detach().cpu().numpy())
        T.append(tgt.detach().cpu().numpy())
        X.append(bx.detach().cpu().numpy())
        if log_interval and (i + 1) % log_interval == 0:
            print(f"  [{name}] iter {i + 1}/{len(loader)}  loss {loss_m.avg:.6f}")

    pred = np.concatenate(P, 0)
    true = np.concatenate(T, 0)
    hist = np.concatenate(X, 0)
    return loss_m.avg, pred, true, hist


# ═════════════════════════════════════════════════════════════════════════════
# 학습 — cell-23이 22개 인자로 호출
# ═════════════════════════════════════════════════════════════════════════════
def training_long_term_forecasting(
    model, trainloader, validloader, criterion, optimizer, accelerator,
    epochs, eval_epochs, log_epochs, log_eval_iter, wandb_iter, use_wandb,
    ckp_metric, savedir, model_name, pred_len, label_len,
    early_stopping_metric, early_stopping_count, lradj, learning_rate, model_config,
):
    os.makedirs(savedir, exist_ok=True)
    history = []

    # ckp_metric 과 early_stopping_metric 은 서로 다른 기준이므로 각자 best를 추적한다.
    best_ckp = np.inf          # 체크포인트 저장 기준 (config: "MSE")
    best_es = np.inf           # 조기 종료 기준        (config: "loss")
    es_counter = 0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        loss_m, batch_t = AverageMeter(), AverageMeter()
        tick = time.time()

        for it, (bx, by, bxm, bym) in enumerate(trainloader):
            dec = _make_dec_input(by, label_len, pred_len)
            out = model(bx, bxm, dec, bym)
            loss = criterion(out[:, -pred_len:, :], by[:, -pred_len:, :])

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            loss_m.update(loss.item(), n=bx.size(0))
            batch_t.update(time.time() - tick)
            tick = time.time()

            if use_wandb and wandb_iter and (it + 1) % wandb_iter == 0:
                try:
                    import wandb
                    wandb.log({"train/loss": loss_m.avg, "lr": optimizer.param_groups[0]["lr"]})
                except Exception:
                    pass

        rec = {"epoch": epoch, "train_loss": loss_m.avg,
               "lr": optimizer.param_groups[0]["lr"], "sec": time.time() - t0}

        # ── 검증 (eval_epochs 주기) ────────────────────────────────────────
        if (epoch + 1) % eval_epochs == 0:
            vloss, vp, vt, _ = _evaluate(model, validloader, criterion, accelerator,
                                         label_len, pred_len, log_eval_iter, "VALID")
            vm = compute_metrics(vp, vt)
            rec.update({"valid_loss": vloss, **{f"valid_{k}": v for k, v in vm.items()}})
            print(f"mse: {vm['MSE']:.4f}\t mae: {vm['MAE']:.4f}\t mape: {vm['MAPE']:.4f}")

            # (1) 체크포인트 — ckp_metric 기준
            cur_ckp = vm.get(ckp_metric.upper(), vloss)
            if cur_ckp <= best_ckp:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    torch.save(accelerator.unwrap_model(model).state_dict(),
                               os.path.join(savedir, "best_model.pt"))
                print(f"Save best model complete, epoch: {epoch}: Best metric has changed "
                      f"from {best_ckp:.5f} to {cur_ckp:.5f}")
                best_ckp = cur_ckp

            # (2) 조기 종료 — early_stopping_metric 기준 (독립 추적)
            cur_es = vloss if early_stopping_metric.lower() == "loss" \
                else vm.get(early_stopping_metric.upper(), vloss)
            if cur_es < best_es:
                best_es, es_counter = cur_es, 0
            else:
                es_counter += 1
                print(f"EarlyStopping counter: {es_counter} out of {early_stopping_count}")
                if es_counter >= early_stopping_count:
                    print(f"Early stopping at epoch {epoch}")
                    history.append(rec)
                    break

        if (epoch + 1) % log_epochs == 0:
            rec["batch_sec"] = batch_t.avg
        history.append(rec)

        lr_new = adjust_learning_rate(optimizer, epoch + 1, learning_rate, lradj)
        if lr_new is not None:
            print(f"Updating learning rate to {lr_new}")

    # 마지막 상태도 남긴다 (best와 별도)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(),
                   os.path.join(savedir, "latest_model.pt"))
        json.dump(history, open(os.path.join(savedir, "train_history.json"), "w"),
                  indent=2, cls=Float32Encoder)
    print(f"Save latest model complete, epoch: {history[-1]['epoch']}: "
          f"Best metric has changed from {best_ckp:.5f} to {best_ckp:.5f}")
    return history


# ═════════════════════════════════════════════════════════════════════════════
# 테스트 — cell-23이 12개 인자로 호출
# ═════════════════════════════════════════════════════════════════════════════
def test_long_term_forecasting(
    model, dataloader, criterion, accelerator, log_interval,
    pred_len, label_len, savedir, model_config, model_name, name, return_output,
):
    loss, pred, true, hist = _evaluate(model, dataloader, criterion, accelerator,
                                       label_len, pred_len, log_interval, name)
    metrics = compute_metrics(pred, true)
    metrics["loss"] = loss

    # ── 참조 baseline: 계절 나이브 (직전 24시간 패턴을 그대로 반복) ──────────
    # 지표의 절대값만으로는 모델이 좋은지 알 수 없다. 1단계 EDA에서 ACF lag24=0.940
    # 이었으므로 '어제 같은 시각 값'은 매우 강한 baseline이다. 이를 못 이기면 의미가 없다.
    season = 24
    reps = int(np.ceil(pred_len / season))
    naive = np.tile(hist[:, -season:, :], (1, reps, 1))[:, :pred_len, :]
    metrics["naive_seasonal"] = compute_metrics(naive, true)
    metrics["skill_score_vs_naive"] = 1.0 - metrics["MSE"] / metrics["naive_seasonal"]["MSE"]

    print(f"mse: {metrics['MSE']:.4f}\t mae: {metrics['MAE']:.4f}\t mape: {metrics['MAPE']:.4f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process and return_output:
        # 요구사항 4-1이 "저장된 내용을 바탕으로" 분석하라고 하므로 예측을 남긴다.
        np.savez_compressed(
            os.path.join(savedir, f"{name}_outputs.npz"),
            pred=pred.astype(np.float32), true=true.astype(np.float32),
            hist=hist.astype(np.float32), naive=naive.astype(np.float32),
        )
        json.dump(metrics, open(os.path.join(savedir, f"{name}_metrics.json"), "w"),
                  indent=2, cls=Float32Encoder)
    return metrics

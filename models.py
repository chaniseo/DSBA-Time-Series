"""
4단계 — 모델 구축 (요구사항 2-1, 2-2)
================================================================================

[계약 — cell-19가 강제하는 것]
    model_classes = globals()
    model_class   = model_classes[modelname]     # 이름으로 조회 → 노트북 전역에 정의 필요
    return model_class(params)                   # 위치인자 1개 = cfg.MODELSETTING

    → __init__(self, configs) : 인자 하나만 받는다
    → forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)  : cell-18 예시 형태

[요구사항 2-1 "기존 모델 코드 그대로 사용 금지"]
공개 구현을 복사하지 않는다. 아래 DRLinear는 1단계 EDA에서 관측한 사실로부터
각 구성요소를 역산해 직접 설계·구현한 모델이다.

================================================================================
DRLinear = Decomposition + RevIN + Linear
================================================================================
이름은 '설계 단계에서 검토한 구성요소 집합'을 가리킨다. 아래 ①의 검증 결과대로
RevIN은 실측에서 손해로 판명되어 **최종 구성에서는 비활성(use_revin=False)**이다.
구현을 남겨둔 것은 ablation 재현을 위해서이며, 이름을 바꾸지 않은 것은
저장된 실험 디렉터리·체크포인트와의 대응을 유지하기 위해서다.

EDA 관측 → 설계 결정의 대응이 1:1이다. 근거 없는 구성요소는 넣지 않았다.

  ① [관측] OT는 후반기 std가 전반기의 0.635배, HUFL/MUFL은 1.39~1.42배.
           30일 rolling std가 3.8~5.7배 변동. train과 test의 분포가 다르다.
           (출제자 baseline의 val MSE 0.376 vs test MSE 0.467 격차의 원인)
    [설계] RevIN — 윈도우마다 자체 평균·분산으로 정규화하고 출력에서 되돌린다.
           전역 통계에 의존하지 않으므로 분포가 이동해도 모델이 보는 값은 안정적이다.
           → 정상성 검정에서 '차분정상' 판정이 났지만 차분 대신 이 방법을 택한 이유이기도 하다.
             차분은 되돌릴 때 오차가 누적되지만 RevIN은 원 스케일 복원이 정확하다.
    [검증 결과 — 가설 반증]  use_revin 기본값을 False로 둔 이유
           30 epoch 정식 ablation에서 RevIN을 넣으면 test MSE가 +0.0070 나빠졌다.
               individual 有: 0.4144 → 0.4214 / individual 無: 0.4196 → 0.4265
           두 조건에서 부호가 같으므로 상호작용이 아니라 RevIN 고유의 효과다.
           원인 추정: RevIN은 출력에 '입력 윈도우의 평균'을 되돌려 더하므로
           '예측 구간의 수준 = 입력 구간의 수준'을 가정한다. 그런데 OT는 2년간
           40°C → 10°C로 단조 하락하여 입력 평균이 미래 평균보다 체계적으로 높다.
           분포 이동에 대응하려던 장치가 오히려 하락 추세를 무시하게 만들었다.

  ② [관측] STL 분산 기여도가 변수마다 정반대다.
           OT : trend 0.967 / seasonal 0.018   ← 추세 지배
           HUFL: trend 0.195 / seasonal 0.712  ← 계절 지배
    [설계] 이동평균으로 추세/계절을 분리해 **각각 독립된 선형 사상**을 학습한다.
           한 경로로 처리하면 성격이 반대인 두 성분을 같은 함수로 억지 근사하게 된다.

  ③ [관측] 채널별 성격이 다르고(위), HUFL↔MUFL 상관 0.987인 반면
           OT는 어떤 변수와도 상관 ≤ 0.224로 독립적이다.
    [설계] 채널별 독립 가중치(individual=True). 채널 간 정보를 섞지 않는다.
           OT 예측에 다른 채널이 기여할 여지가 작다는 상관 분석과 일치한다.

  ④ [관측] ACF lag24 = 0.940, FFT 최상위 주기 24.0h → 일간 주기가 압도적.
           config는 timeenc=0 + embed_type='learned'로 정수 시간특성 + 학습 임베딩을 지정.
    [설계] 미래 시각(x_mark_dec)은 예측 시점에 **이미 알고 있는 정보**다.
           이를 학습 임베딩으로 받아 예측 구간에 채널별 가산 보정을 더한다.
    [검증 결과 — 가설 반증]  use_time_bias 기본값을 False로 되돌린 이유
           3-epoch 예비 실험에서 TimeBias를 **제거했을 때** val MSE가 개선됐다.
               TimeBias 포함: val 0.3306 / 제거: val 0.3223
           이는 EDA와도 일치한다 — 요일별 OT 평균 편차는 0.85도로 전체 std 8.57의
           10%에 불과하고, MSTL의 seasonal_168 분산 기여도는 1.1%였다.
           즉 시각 정보가 추가로 설명할 여지가 거의 없는데 파라미터만 1,335개 늘어
           과적합만 유발했다. **관측이 지지하지 않는 구성요소는 뺀다.**
           코드는 ablation 재현용으로 남겨두되 기본값은 비활성으로 둔다.

  ⑤ [관측] 요일별 OT 평균 편차 0.85도(전체 std 8.57의 10%), MSTL seasonal_168 기여 1.1%.
    [설계] 주간 성분이 미미하므로 window_size=96(4일)로 충분하다. 별도 주간 모듈 없음.
           → 넣지 않은 것도 근거가 있는 결정이다.

================================================================================
차원 변화 (요구사항 2-2) — B=batch, L=96, P=96, C=7, T=4, D=16
================================================================================
  x_enc                                    (B, 96, 7)
  x_mark_enc                               (B, 96, 4)   [사용 안 함: ⑤ 참조]
  x_dec                                    (B, 96, 7)   [label_len=0이라 0으로 채워짐]
  x_mark_dec                               (B, 96, 4)   [미래 시각 — ④에서 사용]

  ┌ RevIN ─────────────────────────────────────────────────────────────
  │ mean = x_enc.mean(dim=1)               (B, 1, 7)
  │ std  = x_enc.std(dim=1)                (B, 1, 7)
  │ x    = (x_enc - mean) / std            (B, 96, 7)
  │ x    = x * affine_w + affine_b         (B, 96, 7)
  ├ Decomposition ─────────────────────────────────────────────────────
  │ pad  = replicate(x, 12, 12)            (B, 120, 7)
  │ trend    = AvgPool1d(k=25, s=1)        (B, 96, 7)
  │ seasonal = x - trend                   (B, 96, 7)
  ├ Channel-wise Linear (einsum) ──────────────────────────────────────
  │ trend    → permute                     (B, 7, 96)
  │   einsum('bcl,clp->bcp', ·, W_t)       (B, 7, 96)     W_t: (7, 96, 96)
  │ seasonal → 동일 경로                    (B, 7, 96)     W_s: (7, 96, 96)
  │ y = (t_out + s_out).permute            (B, 96, 7)
  ├ Time bias (미래 시각 활용) ─────────────────────────────────────────
  │ x_mark_dec → 4개 Embedding 합           (B, 96, 16)
  │ Dropout → Linear(16 → 7)                (B, 96, 7)
  │ y = y + bias                            (B, 96, 7)
  ├ RevIN 역변환 ───────────────────────────────────────────────────────
  │ y = (y - affine_b) / affine_w           (B, 96, 7)
  │ y = y * std + mean                      (B, 96, 7)
  └────────────────────────────────────────────────────────────────────
  output                                    (B, 96, 7)   ← c_out=7
"""

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.

    윈도우 하나하나를 그 자신의 평균·표준편차로 정규화하고, 출력에서 정확히 되돌린다.
    전역(train) 통계에 의존하지 않기 때문에 train/test 분포가 달라도 영향을 받지 않는다.
    EDA 관측 ①(분산 레짐 변화, 추세 하락)에 대한 직접적인 대응이다.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))

    def normalize(self, x):
        # x: (B, L, C) → 시간축(dim=1)에 대해 통계를 낸다. 채널별로 독립.
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        x = (x - self.mean) / self.std
        if self.affine:
            x = x * self.weight + self.bias
        return x

    def denormalize(self, x):
        if self.affine:
            x = (x - self.bias) / (self.weight + self.eps * self.eps)
        return x * self.std + self.mean


class SeriesDecomp(nn.Module):
    """
    이동평균으로 추세를 뽑고, 나머지를 계절 성분으로 본다.

    양 끝을 replicate padding 하는 이유: zero padding을 쓰면 계열 양 끝의 추세가
    0쪽으로 끌려가 왜곡된다. 끝값을 복제하면 경계에서 추세가 평평하게 이어진다.
    kernel=25는 하루(24h)를 덮는 최소 홀수 — 일간 변동을 추세에서 걷어내기 위함이다.
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):  # (B, L, C)
        front = x[:, :1, :].repeat(1, self.pad, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size - 1 - self.pad, 1)
        padded = torch.cat([front, x, end], dim=1)  # (B, L + k - 1, C)
        trend = self.avg(padded.permute(0, 2, 1)).permute(0, 2, 1)  # (B, L, C)
        return x - trend, trend


class ChannelWiseLinear(nn.Module):
    """
    채널마다 별개의 (L → P) 선형 사상.

    individual=True  : W (C, L, P) — 채널별 독립 가중치. EDA 관측 ②③의 귀결.
    individual=False : W (1, L, P) — 전 채널 공유. 파라미터 1/C, 과적합에 강함.
                       (출제자 baseline이 18,624 = Linear(96,96)×2 이므로 이쪽에 해당)
    einsum으로 채널 루프 없이 한 번에 계산한다.
    """

    def __init__(self, num_channels, in_len, out_len, individual=True):
        super().__init__()
        c = num_channels if individual else 1
        self.individual = individual
        self.weight = nn.Parameter(torch.empty(c, in_len, out_len))
        self.bias = nn.Parameter(torch.zeros(c, out_len))
        # 초기값: 모든 입력 시점을 균등 평균 내는 상태에서 출발한다.
        # 무작위 초기화보다 수렴이 빠르고, '최근값 평균'이라는 합리적 baseline에서 시작한다.
        nn.init.constant_(self.weight, 1.0 / in_len)

    def forward(self, x):  # (B, C, L)
        w = self.weight if self.individual else self.weight.expand(x.size(1), -1, -1)
        b = self.bias if self.individual else self.bias.expand(x.size(1), -1)
        return torch.einsum("bcl,clp->bcp", x, w) + b


class TimeBias(nn.Module):
    """
    미래 시각 정보로 예측 구간에 채널별 가산 보정을 만든다.

    timeenc=0 이면 x_mark는 [month, day, weekday, hour] 정수다.
    config의 embed_type='learned'가 지시하는 대로 nn.Embedding 룩업을 쓴다.
    (timeenc=1이면 값이 실수 정규화라 임베딩을 못 쓰므로 선형 사상으로 대체한다)
    """

    VOCAB = {"month": 13, "day": 32, "weekday": 7, "hour": 24, "minute_q": 4}

    def __init__(self, timeenc, freq, d_model, c_out, dropout):
        super().__init__()
        self.timeenc = timeenc
        names = ["month", "day", "weekday", "hour"]
        if str(freq).lower() in ("t", "m", "s"):
            names.append("minute_q")
        self.names = names
        if timeenc == 0:
            self.embs = nn.ModuleList([nn.Embedding(self.VOCAB[n], d_model) for n in names])
        else:
            self.proj_in = nn.Linear(len(names), d_model)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, c_out)
        nn.init.zeros_(self.proj.weight)  # 보정항이 0에서 출발 → 초기에는 순수 선형 모델
        nn.init.zeros_(self.proj.bias)

    def forward(self, x_mark):  # (B, P, T)
        if self.timeenc == 0:
            idx = x_mark.long()
            h = sum(emb(idx[:, :, i]) for i, emb in enumerate(self.embs))  # (B, P, D)
        else:
            h = self.proj_in(x_mark)
        return self.proj(self.drop(h))  # (B, P, C)


class DRLinear(nn.Module):
    """Decomposition + RevIN + channel-wise Linear. cell-19가 configs 하나로 생성한다."""

    def __init__(self, configs):
        super().__init__()
        g = lambda k, d: getattr(configs, k, d)

        self.task_name = g("taskname", "long_term_forecast")
        self.seq_len = int(g("window_size", 96))
        self.pred_len = int(g("pred_len", 96))
        self.label_len = int(g("label_len", 0))
        self.enc_in = int(g("enc_in", 7))
        self.c_out = int(g("c_out", 7))

        # 아래 둘은 기본 False — 설계 의도와 달리 실측에서 성능을 떨어뜨렸다(위 ①, ④).
        # 구현을 남겨둔 이유는 ablation 재현 가능성을 위해서다.
        self.use_revin = bool(g("use_revin", False))
        self.use_time = bool(g("use_time_bias", False))
        individual = bool(g("individual", True))
        kernel = int(g("moving_avg", 25))
        d_model = int(g("d_model", 16))
        dropout = float(g("dropout", 0.1))

        if self.use_revin:
            self.revin = RevIN(self.enc_in)
        self.decomp = SeriesDecomp(kernel)
        self.lin_seasonal = ChannelWiseLinear(self.enc_in, self.seq_len, self.pred_len, individual)
        self.lin_trend = ChannelWiseLinear(self.enc_in, self.seq_len, self.pred_len, individual)
        if self.use_time:
            self.time_bias = TimeBias(int(g("timeenc", 0)), g("freq", "h"), d_model, self.c_out, dropout)

    # ── 차원 추적용 (요구사항 2-2) ───────────────────────────────────────────
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, trace=None):
        def log(name, t):
            if trace is not None:
                trace.append((name, tuple(t.shape)))

        log("x_enc (입력)", x_enc)
        if x_mark_enc is not None:
            log("x_mark_enc (미사용)", x_mark_enc)
        if x_mark_dec is not None:
            log("x_mark_dec (미래 시각)", x_mark_dec)

        x = x_enc
        if self.use_revin:
            x = self.revin.normalize(x)
            log("RevIN.normalize", x)

        seasonal, trend = self.decomp(x)
        log("decomp → seasonal", seasonal)
        log("decomp → trend", trend)

        s = seasonal.permute(0, 2, 1)
        t = trend.permute(0, 2, 1)
        log("permute (B,C,L)", s)

        s = self.lin_seasonal(s)
        t = self.lin_trend(t)
        log("ChannelWiseLinear → seasonal", s)
        log("ChannelWiseLinear → trend", t)

        y = (s + t).permute(0, 2, 1)
        log("합산 + permute (B,P,C)", y)

        if self.use_time and x_mark_dec is not None:
            bias = self.time_bias(x_mark_dec[:, -self.pred_len :, :])
            log("TimeBias", bias)
            y = y + bias

        if self.use_revin:
            y = self.revin.denormalize(y)
            log("RevIN.denormalize", y)

        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            out = y[:, -self.pred_len :, :]
            log("output (출력)", out)
            return out
        return None

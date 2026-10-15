"""In-context personalization transformer.

The query meal attends to the person's own history meals, each given together with the glucose curve it produced,
so the network infers the person's response function from examples (no per-person training at prediction time).
Outputs the 0-120 min glucose-change curve (5-min grid) at three quantiles. Missing modalities (no photo, no heart
rate, no insulin, macros only estimated) are handled with explicit masks, so one model can be pretrained on the
large T1D dataset (carbs + insulin, no photos) and fine-tuned on CGMacros (photos + macros, no insulin).
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn

NUM_COLS = ["k_carbs", "k_protein", "k_fat", "k_fiber",            # macros known at prediction time (logged / entered)
            "e_carbs", "e_protein", "e_fat", "e_fiber", "e_carbs_sd",  # macros estimated from the photo
            "mt_breakfast", "mt_lunch", "mt_dinner", "mt_snack",
            "tod_sin", "tod_cos", "G0", "g0_rel24", "slope30", "slope60", "sd60",
            "hr_pre", "hr_max60", "prev_gap", "prev_carbs", "meal_bolus", "bolus_lead", "iob", "basal"]
NUM_COLS_V2 = NUM_COLS + ["age", "sex_female", "bmi"]          # v2 adds person descriptors (both datasets have them)
NUM_COLS_BY_VERSION = {"v1": NUM_COLS, "v2": NUM_COLS_V2}
DEVICE = os.environ.get("PMRF_DEVICE", "cpu")                    # training/inference device; the service runs on the CPU
SCALE = dict(age=50, sex_female=1, bmi=30, k_carbs=50, k_protein=30, k_fat=20, k_fiber=10, e_carbs=50, e_protein=30, e_fat=20, e_fiber=10, e_carbs_sd=30,
             G0=100, g0_rel24=40, slope30=1, slope60=1, sd60=15, hr_pre=20, hr_max60=30, prev_gap=720, prev_carbs=50,
             meal_bolus=8, bolus_lead=20, iob=4, basal=1)
QUANTILES = (0.1, 0.5, 0.9)
CURVE_SCALE = 50.0
SCALAR_SCALE = np.array([50.0, 3000.0, 60.0], np.float32)      # peak rise (mg/dL), iAUC 0-120 (mg/dL·min), time to peak (min)
N_CURVE, N_PRE = 25, 25


def numeric_block(df, cols=None):
    """(n x 2*len(cols)) float32: scaled values with NaN->0, followed by availability masks"""
    cols = cols or NUM_COLS
    X = np.zeros((len(df), len(cols)), np.float32)
    for j, c in enumerate(cols):
        if c in df:
            X[:, j] = df[c].to_numpy(float) / SCALE.get(c, 1.0)
        else:
            X[:, j] = np.nan
    m = np.isfinite(X).astype(np.float32)
    return np.concatenate([np.nan_to_num(X), m], 1)


class ICLTransformer(nn.Module):
    def __init__(self, img_dim=512, d=96, heads=4, layers=3, dropout=0.1, n_num=None):
        super().__init__()
        self.n_num = n_num or len(NUM_COLS)                              # number of numeric input columns (v1 28, v2 31)
        nnum = 2 * self.n_num
        mlp = lambda i: nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, d))
        self.img = nn.Linear(img_dim, d)
        self.num = mlp(nnum)
        self.pre = mlp(2 * N_PRE)
        self.resp = mlp(N_CURVE)
        self.role = nn.Embedding(2, d)                   # 0 = query, 1 = history example
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, (N_CURVE - 1) * len(QUANTILES)))
        # direct heads for peak rise, iAUC and time to peak: the max/area of a median curve is biased low
        self.scalar = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, 3 * len(QUANTILES)))

    def token(self, img, has_img, num, pre, resp=None):
        t = self.num(num) + self.pre(pre) + self.img(img) * has_img.unsqueeze(-1)
        if resp is not None:
            t = t + self.resp(resp)
        return t

    def forward(self, q, ctx, ctx_pad):
        """q: dict of (B, ...) query tensors; ctx: dict of (B, K, ...) history tensors; ctx_pad: (B, K) True = padding"""
        return self.forward_all(q, ctx, ctx_pad)[0]

    def forward_all(self, q, ctx, ctx_pad):
        """returns the curve (B, 25, 3) and the scalar outcomes (B, 3, 3), both scaled"""
        tq = self.token(q["img"], q["has_img"], q["num"], q["pre"]).unsqueeze(1) + self.role.weight[0]
        tc = self.token(ctx["img"], ctx["has_img"], ctx["num"], ctx["pre"], ctx["resp"]) + self.role.weight[1]
        x = torch.cat([tq, tc], 1)
        pad = torch.cat([torch.zeros_like(ctx_pad[:, :1]), ctx_pad], 1)
        h = self.encoder(x, src_key_padding_mask=pad)[:, 0]
        out = self.head(h).view(-1, N_CURVE - 1, len(QUANTILES))
        out = torch.cumsum(out, 1)                                          # curve as cumulative increments from 0
        out = torch.sort(out, -1).values                                    # non-crossing quantiles
        sc = torch.sort(self.scalar(h).view(-1, 3, len(QUANTILES)), -1).values
        return torch.cat([torch.zeros_like(out[:, :1]), out], 1), sc        # (B, 25, 3) first point = 0; (B, 3, 3)


def pinball(pred, y, mask):
    qs = torch.tensor(QUANTILES, dtype=pred.dtype, device=pred.device).view(1, 1, -1)
    e = y.unsqueeze(-1) - pred
    l = torch.maximum(qs * e, (qs - 1) * e)
    return (l * mask.unsqueeze(-1)).sum() / (mask.sum() * len(QUANTILES)).clamp(min=1)


class EpisodeData:
    """Holds token inputs for all meals; builds (query, history) episodes per person."""

    def __init__(self, num, pre, resp, img, has_img, person, scalars=None):
        self.num = torch.tensor(num, dtype=torch.float32)
        pre = np.asarray(pre, np.float32) / CURVE_SCALE
        self.pre = torch.tensor(np.concatenate([np.nan_to_num(pre), np.isfinite(pre).astype(np.float32)], 1))
        r = np.asarray(resp, np.float32) / CURVE_SCALE
        self.resp_mask = torch.tensor(np.isfinite(r).astype(np.float32))
        self.resp = torch.tensor(np.nan_to_num(r))
        self.img = torch.tensor(np.asarray(img, np.float32)) if img is not None else torch.zeros(len(num), 1)
        self.has_img = torch.tensor(np.asarray(has_img, np.float32))
        self.person = np.asarray(person)
        if scalars is not None:
            sc = np.asarray(scalars, np.float32) / SCALAR_SCALE
            self.scal_mask = torch.tensor(np.isfinite(sc).astype(np.float32)); self.scal = torch.tensor(np.nan_to_num(sc))
        else:
            self.scal = None

    def batch(self, q_idx, ctx_lists, q_num_override=None):
        B = len(q_idx); K = max(1, max(len(c) for c in ctx_lists))
        ci = np.zeros((B, K), int); pad = np.ones((B, K), bool)
        for b, c in enumerate(ctx_lists):
            ci[b, :len(c)] = c; pad[b, :len(c)] = False
        ci_t = torch.tensor(ci)
        qn = self.num[q_idx] if q_num_override is None else q_num_override
        q = dict(img=self.img[q_idx], has_img=self.has_img[q_idx], num=qn, pre=self.pre[q_idx])
        ctx = dict(img=self.img[ci_t], has_img=self.has_img[ci_t], num=self.num[ci_t], pre=self.pre[ci_t], resp=self.resp[ci_t])
        return q, ctx, torch.tensor(pad), self.resp[q_idx], self.resp_mask[q_idx]


def train(model, data, queries, history_of, steps, lr=3e-4, batch=128, kmax=32, mask_query_known=None, seed=0, log_every=500,
          val=None, val_every=500, schedule_steps=None, scalar_weight=1.0):
    """queries: candidate query indices; history_of(i) -> candidate history indices for query i (same person, not i).
    mask_query_known: indices of NUM columns to blank for queries (e.g. logged macros that are unknown at prediction time)."""
    g = torch.Generator().manual_seed(seed); rng = np.random.default_rng(seed)
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # schedule_steps > steps reproduces an early-stopped run exactly: same learning-rate trajectory, stopped at `steps`
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, schedule_steps or steps), pct_start=0.05)
    best, best_state, hist = math.inf, None, []
    for step in range(steps):
        model.train()
        qi = rng.choice(queries, batch)
        ctx = []
        for i in qi:
            h = history_of(i)
            k = rng.integers(0, min(kmax, len(h)) + 1) if len(h) else 0
            ctx.append(rng.choice(h, k, replace=False) if k else [])
        qn = data.num[qi].clone()
        if mask_query_known is not None:
            nn_ = data.num.shape[1] // 2
            qn[:, mask_query_known] = 0; qn[:, [nn_ + j for j in mask_query_known]] = 0
        q, c, pad, y, ym = _to(data.batch(qi, ctx, qn))
        curve, sc = model.forward_all(q, c, pad)
        loss = pinball(curve, y, ym)
        if data.scal is not None:
            loss = loss + scalar_weight * pinball(sc, data.scal[qi].to(DEVICE), data.scal_mask[qi].to(DEVICE))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if val is not None and (step + 1) % val_every == 0:
            v = val(model)
            hist.append((step + 1, float(loss), v))
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
            if log_every:
                print(f"  step {step + 1}: train {loss.item():.4f}  val {v:.4f}", flush=True)
        elif log_every and (step + 1) % log_every == 0:
            print(f"  step {step + 1}: train {loss.item():.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to("cpu"), hist


def _to(batch):
    """move a (q, ctx, pad, y, ymask) batch to DEVICE"""
    q, c, pad, y, ym = batch
    mv = lambda d: {k: v.to(DEVICE) for k, v in d.items()}
    return mv(q), mv(c), pad.to(DEVICE), y.to(DEVICE), ym.to(DEVICE)


@torch.no_grad()
def predict(model, data, q_idx, ctx_lists, mask_query_known=None, batch=256):
    model.eval(); model.to(DEVICE)
    outs = []
    for s in range(0, len(q_idx), batch):
        qi = np.asarray(q_idx[s:s + batch]); cl = ctx_lists[s:s + batch]
        qn = data.num[qi].clone()
        if mask_query_known is not None:
            nn_ = data.num.shape[1] // 2
            qn[:, mask_query_known] = 0; qn[:, [nn_ + j for j in mask_query_known]] = 0
        q, c, pad, _, _ = _to(data.batch(qi, cl, qn))
        outs.append(model(q, c, pad).cpu().numpy() * CURVE_SCALE)
    model.to("cpu")
    return np.concatenate(outs)          # (n, 25, 3) glucose change, mg/dL


@torch.no_grad()
def predict_scalars(model, data, q_idx, ctx_lists, mask_query_known=None, batch=256):
    """direct-head predictions (n, 3, 3): [peak rise, iAUC, time to peak] x quantiles, in natural units"""
    model.eval(); model.to(DEVICE)
    outs = []
    for s in range(0, len(q_idx), batch):
        qi = np.asarray(q_idx[s:s + batch]); cl = ctx_lists[s:s + batch]
        qn = data.num[qi].clone()
        if mask_query_known is not None:
            nn_ = data.num.shape[1] // 2
            qn[:, mask_query_known] = 0; qn[:, [nn_ + j for j in mask_query_known]] = 0
        q, c, pad, _, _ = _to(data.batch(qi, cl, qn))
        outs.append(model.forward_all(q, c, pad)[1].cpu().numpy() * SCALAR_SCALE[None, :, None])
    model.to("cpu")
    return np.concatenate(outs)


def curve_to_targets(curves, g0):
    """median curve (n, 25) on the 5-min grid -> challenge outputs"""
    c = curves
    k = np.argmax(c, 1)
    return dict(G30=g0 + c[:, 6], G60=g0 + c[:, 12], G90=g0 + c[:, 18], G120=g0 + c[:, 24], Gmax=g0 + c.max(1), dpeak=c.max(1),
                ttp=k * 5.0, iauc=np.trapz(np.clip(c, 0, None), dx=5, axis=1))

"""
EntropyAdam — PyTorch LLM Test
===============================

Tiny Transformer (char-level LM) on Shakespeare.
Adam vs EntropyAdam head-to-head.

loss trajectory의 bigram entropy → 적응형 학습률.
"관찰이 짧으면 혼돈이고, 관찰이 길면 질서다"

Author: 쵸리 (Chori)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# ═════════════════════════════════════════════
# EntropyAdam (PyTorch Optimizer)
# ═════════════════════════════════════════════

N_BINS = 5


class EntropyAdam(torch.optim.Optimizer):
    """
    Adam + Shannon bigram entropy 가속기.

    사용법:
        optimizer = EntropyAdam(model.parameters(), lr=3e-4)
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.report_loss(loss)  # <-- 이것만 추가
            optimizer.step()
            optimizer.zero_grad()
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0,
                 window=50, n_bins=N_BINS, max_boost=5.0,
                 accel_alpha=0.05, brake_alpha=0.3):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.window = window
        self.n_bins = n_bins
        self.max_boost = max_boost
        self.accel_alpha = accel_alpha
        self.brake_alpha = brake_alpha

        self._loss_history = []
        self._deltas = []
        self._states = []
        self.H_history = []
        self.mult_history = []
        self._current_mult = 1.0

    # ── public API ──

    def report_loss(self, loss_val):
        """매 스텝 loss 보고. Tensor도 float도 OK."""
        if isinstance(loss_val, torch.Tensor):
            loss_val = loss_val.item()

        if self._loss_history:
            prev = self._loss_history[-1]
            delta = (loss_val - prev) / (abs(prev) + 1e-10)
            self._deltas.append(delta)
            if len(self._deltas) > self.window:
                self._deltas = self._deltas[-self.window:]

            if len(self._deltas) >= 10:
                ds = np.array(self._deltas)
                thr = np.percentile(ds, [20, 40, 60, 80])
            else:
                thr = np.array([-0.01, -0.001, 0.001, 0.01])

            state = 0
            for i, t in enumerate(thr):
                if delta < t:
                    state = i
                    break
            else:
                state = len(thr)

            self._states.append(min(state, self.n_bins - 1))
            if len(self._states) > self.window:
                self._states = self._states[-self.window:]

        self._loss_history.append(loss_val)
        if len(self._loss_history) > self.window + 10:
            self._loss_history = self._loss_history[-(self.window + 10):]

    # ── internal ──

    def _bigram_entropy(self):
        states = self._states
        n = self.n_bins
        if len(states) < 2:
            return np.log2(n)

        T = np.zeros((n, n))
        for i in range(len(states) - 1):
            T[states[i]][states[i + 1]] += 1

        total = np.sum(T)
        if total == 0:
            return np.log2(n)

        H = 0.0
        for x in range(n):
            rs = np.sum(T[x])
            if rs == 0:
                continue
            px = rs / total
            for y in range(n):
                if T[x][y] > 0:
                    pyx = T[x][y] / rs
                    H -= px * pyx * np.log2(pyx)
        return H

    def _get_mult(self):
        H_max = np.log2(self.n_bins)

        if len(self._states) < 10:
            self.H_history.append(H_max)
            self.mult_history.append(1.0)
            return 1.0

        H = self._bigram_entropy()
        H_norm = min(H / H_max, 1.0)

        raw = 1.0 + (self.max_boost - 1.0) * (1.0 - H_norm) ** 2

        if raw > self._current_mult:
            alpha = self.accel_alpha
        else:
            alpha = self.brake_alpha

        self._current_mult = (1 - alpha) * self._current_mult + alpha * raw

        self.H_history.append(H)
        self.mult_history.append(self._current_mult)
        return self._current_mult

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        mult = self._get_mult()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if wd != 0:
                    grad = grad.add(p.data, alpha=wd)

                st = self.state[p]
                if len(st) == 0:
                    st['step'] = 0
                    st['m'] = torch.zeros_like(p.data)
                    st['v'] = torch.zeros_like(p.data)

                st['step'] += 1
                m, v = st['m'], st['v']

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1 ** st['step'])
                v_hat = v / (1 - beta2 ** st['step'])

                p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr * mult)

        return loss


# ═════════════════════════════════════════════
# Tiny Transformer (Character-Level LM)
# ═════════════════════════════════════════════

class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, n_layers=2,
                 dim_ff=512, max_seq=256, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.max_seq = max_seq

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        h = self.tok_emb(x) * math.sqrt(self.d_model) + self.pos_emb(pos)

        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        h = self.blocks(h, mask=mask)
        h = self.ln_f(h)
        return self.head(h)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ═════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════

def load_data(path, seq_len=128):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(set(text))
    c2i = {c: i for i, c in enumerate(chars)}

    data = torch.tensor([c2i[c] for c in text], dtype=torch.long)
    n = len(data)

    # train/val split (90/10)
    split = int(n * 0.9)
    return data[:split], data[split:], len(chars), c2i


def get_batch(data, seq_len, batch_size, device):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix]).to(device)
    return x, y


# ═════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════

def train_model(model, optimizer, train_data, val_data, n_steps,
                seq_len=128, batch_size=32, device='cpu',
                report_fn=None, eval_every=100):
    model.train()
    train_losses = []
    val_losses = []

    for step in range(1, n_steps + 1):
        x, y = get_batch(train_data, seq_len, batch_size, device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if report_fn:
            report_fn(loss)

        optimizer.step()
        train_losses.append(loss.item())

        if step % eval_every == 0:
            val_loss = evaluate(model, val_data, seq_len, batch_size, device)
            val_losses.append((step, val_loss))
            avg_train = np.mean(train_losses[-eval_every:])
            print(f"    step {step:4d} | train {avg_train:.4f} | val {val_loss:.4f}")
            model.train()

    return train_losses, val_losses


@torch.no_grad()
def evaluate(model, data, seq_len=128, batch_size=32, device='cpu', n_batches=10):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, seq_len, batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    return np.mean(losses)


# ═════════════════════════════════════════════
# Head-to-Head
# ═════════════════════════════════════════════

def run_comparison(data_path, n_steps=1000, seq_len=128, batch_size=32,
                   lr=3e-4, d_model=128, n_layers=2, seed=42):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  device: {device}")

    train_data, val_data, vocab_size, c2i = load_data(data_path, seq_len)
    print(f"  vocab: {vocab_size}, train: {len(train_data):,}, val: {len(val_data):,}")

    # ── Adam ──
    print(f"\n{'='*50}")
    print(f"  Adam (lr={lr})")
    print(f"{'='*50}")

    torch.manual_seed(seed)
    model_a = CharTransformer(vocab_size, d_model=d_model, n_layers=n_layers).to(device)
    print(f"  params: {model_a.count_params():,}")
    opt_a = torch.optim.Adam(model_a.parameters(), lr=lr)

    torch.manual_seed(seed * 7 + 1)  # batch sampling seed
    t0 = time.time()
    losses_a, val_a = train_model(model_a, opt_a, train_data, val_data,
                                  n_steps, seq_len, batch_size, device)
    time_a = time.time() - t0

    # ── EntropyAdam ──
    print(f"\n{'='*50}")
    print(f"  EntropyAdam (lr={lr}, max_boost=5.0)")
    print(f"{'='*50}")

    torch.manual_seed(seed)
    model_e = CharTransformer(vocab_size, d_model=d_model, n_layers=n_layers).to(device)
    opt_e = EntropyAdam(model_e.parameters(), lr=lr, max_boost=5.0)

    torch.manual_seed(seed * 7 + 1)  # same batch sampling seed
    t0 = time.time()
    losses_e, val_e = train_model(model_e, opt_e, train_data, val_data,
                                  n_steps, seq_len, batch_size, device,
                                  report_fn=opt_e.report_loss)
    time_e = time.time() - t0

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  RESULTS ({n_steps} steps)")
    print(f"{'='*60}")
    best_a = min(losses_a)
    best_e = min(losses_e)
    final_val_a = val_a[-1][1] if val_a else float('inf')
    final_val_e = val_e[-1][1] if val_e else float('inf')

    print(f"  Adam:        best_train={best_a:.4f}  final_val={final_val_a:.4f}  time={time_a:.1f}s")
    print(f"  EntropyAdam: best_train={best_e:.4f}  final_val={final_val_e:.4f}  time={time_e:.1f}s")

    if final_val_e < final_val_a:
        imp = (final_val_a - final_val_e) / final_val_a * 100
        print(f"\n  --> EntropyAdam wins (val loss {imp:.1f}% lower)")
    else:
        imp = (final_val_e - final_val_a) / final_val_e * 100
        print(f"\n  --> Adam wins (val loss {imp:.1f}% lower)")

    # ── Plot ──
    if HAS_PLT:
        plot_results(losses_a, losses_e, val_a, val_e, opt_e, n_steps, batch_size)

    return losses_a, losses_e, opt_e


def plot_results(losses_a, losses_e, val_a, val_e, opt_e, n_steps, batch_size=32):
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=False)

    # 1. Training loss (smoothed)
    w = 20
    smooth_a = np.convolve(losses_a, np.ones(w)/w, mode='valid')
    smooth_e = np.convolve(losses_e, np.ones(w)/w, mode='valid')
    axes[0].plot(smooth_a, 'b-', alpha=0.7, label=f'Adam (final={smooth_a[-1]:.3f})')
    axes[0].plot(smooth_e, 'r-', alpha=0.7, label=f'EntropyAdam (final={smooth_e[-1]:.3f})')
    axes[0].set_ylabel('Train Loss (smoothed)')
    axes[0].legend()
    axes[0].set_title(f'Tiny Transformer on Shakespeare (bs={batch_size}) -- Adam vs EntropyAdam')
    axes[0].grid(True, alpha=0.3)

    # 2. Validation loss
    if val_a and val_e:
        steps_a, vl_a = zip(*val_a)
        steps_e, vl_e = zip(*val_e)
        axes[1].plot(steps_a, vl_a, 'b-o', alpha=0.7, label='Adam')
        axes[1].plot(steps_e, vl_e, 'r-o', alpha=0.7, label='EntropyAdam')
        axes[1].set_ylabel('Val Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # 3. Bigram Entropy
    if opt_e.H_history:
        H_max = np.log2(N_BINS)
        axes[2].plot(opt_e.H_history, 'g-', alpha=0.7, label='H(state_t | state_{t-1})')
        axes[2].axhline(y=H_max, color='gray', ls='--', alpha=0.5, label=f'H_max={H_max:.2f}')
        axes[2].set_ylabel('Bigram Entropy')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    # 4. LR Multiplier
    if opt_e.mult_history:
        axes[3].plot(opt_e.mult_history, color='orange', alpha=0.7, label='LR multiplier')
        axes[3].axhline(y=1.0, color='gray', ls='--', alpha=0.5, label='Adam baseline')
        axes[3].set_ylabel('LR Multiplier')
        axes[3].set_xlabel('Step')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'entropy_adam_llm_bs{batch_size}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {path}")


# ═════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  EntropyAdam vs Adam — LLM Test")
    print("  Tiny Transformer on Shakespeare (char-level)")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    data_path = os.path.join(data_dir, 'tinyshakespeare.txt')

    if not os.path.exists(data_path):
        print(f"  ERROR: {data_path} not found")
        exit(1)

    hamlet_path = os.path.join(data_dir, 'hamlet_only.txt')

    # Hamlet only (homogeneous) vs Shakespeare (heterogeneous)
    datasets = []
    if os.path.exists(hamlet_path):
        datasets.append(('HAMLET ONLY (homogeneous)', hamlet_path, 64))
    datasets.append(('SHAKESPEARE (heterogeneous)', data_path, 128))

    for label, dpath, seq_len in datasets:
        print(f"\n{'#'*60}")
        print(f"  {label}")
        print(f"{'#'*60}")
        run_comparison(
            dpath,
            n_steps=2000,
            seq_len=seq_len,
            batch_size=64,
            lr=3e-4,
            d_model=128,
            n_layers=2,
        )

"""Simulasi mainan: efek projector terhadap ketahanan geometri lintasan ala ReStraV.
DATA SINTETIS, bukan DINOv2 / video asli. Tujuan: ilustrasi mekanisme, bukan bukti empiris skripsi.
"""
import numpy as np, torch, torch.nn as nn, json, sys
from sklearn.metrics import roc_auc_score

D, T = 64, 24
SIG_LEVELS = [0.0, 0.006, 0.012, 0.02, 0.03, 0.045]   # q0=source ... q5 (q4,q5 tidak dipakai saat training)
TRAIN_Q = [1, 2, 3]

def make_clean(n, label, rng):
    """Lintasan halus di R^64. Real: belokan kecil; AI: belokan lebih besar. Panjang langkah acak (gerak kecil/besar)."""
    out = np.zeros((n, T, D))
    scale = np.ones(D); scale[32:] = 0.5            # dimensi 32-63 membawa sinyal lebih lemah
    for k in range(n):
        bend = np.radians(rng.normal(22, 6) if label == 0 else rng.normal(38, 8))
        step = np.exp(rng.uniform(np.log(0.15), np.log(1.0)))
        v = rng.normal(size=D); v /= np.linalg.norm(v); z = np.zeros(D); path = [z.copy()]
        for t in range(T - 1):
            r = rng.normal(size=D); r -= r.dot(v) * v; r /= np.linalg.norm(r)
            v = np.cos(bend) * v + np.sin(bend) * r
            z = z + step * v; path.append(z.copy())
        out[k] = np.array(path) * scale
    return out

def compress(Z, q, rng):
    """Noise per-frame anisotropik: kuat di dim 32-63, lemah (30%) di dim 0-31. Frame 'B' (t%3!=0) sedikit lebih bising."""
    s = SIG_LEVELS[q]
    if s == 0: return Z.copy()
    std = np.full(D, s); std[:32] *= 0.3
    fr = np.where(np.arange(T) % 3 == 0, 1.0, 1.3)[None, :, None]
    return Z + rng.normal(size=Z.shape) * std * fr

def geometry(u):
    """21 fitur ReStraV (Internò dkk. 2025, Bag. 6): d1..d7, θ1..θ6, mean/var/min/max dari d dan θ."""
    du = u[:, 1:] - u[:, :-1]
    d = du.norm(dim=-1)
    cos = (du[:, :-1] * du[:, 1:]).sum(-1) / (d[:, :-1] * d[:, 1:] + 1e-8)
    th = torch.rad2deg(torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6)))
    st = lambda x: [x.mean(1), x.var(1), x.min(1).values, x.max(1).values]
    return torch.cat([d[:, :7], th[:, :6], torch.stack(st(d) + st(th), 1)], 1)

class Model(nn.Module):
    def __init__(self, use_proj):
        super().__init__()
        self.proj = nn.Linear(D, D, bias=False) if use_proj else None
        if use_proj: nn.init.eye_(self.proj.weight)       # mulai identik dengan ReStraV (W = I)
        self.norm = nn.BatchNorm1d(21, affine=False)
        self.mlp = nn.Sequential(nn.Linear(21, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def feats(self, z):
        return geometry(self.proj(z) if self.proj is not None else z)
    def forward(self, z):
        f = self.feats(z); return self.mlp(self.norm(f)).squeeze(1), f

def std_feats(fs, fc):
    sd = fs.std(0, keepdim=True).detach() + 1e-6    # skala dari sumber, di-detach: cegah 'curang' lewat skala
    mu = fs.mean(0, keepdim=True).detach()
    return (fs - mu) / sd, (fc - mu) / sd

def loss_consistency(fs, fc):
    a, b = std_feats(fs, fc); return ((a - b) ** 2).mean()

def loss_contrastive(fs, fc, tau=0.2):
    a, b = std_feats(fs, fc)
    a = nn.functional.normalize(a, dim=1); b = nn.functional.normalize(b, dim=1)
    logits = a @ b.t() / tau; tgt = torch.arange(len(a))
    return 0.5 * (nn.functional.cross_entropy(logits, tgt) + nn.functional.cross_entropy(logits.t(), tgt))

def auroc_per_q(m, data, rng_seed):
    m.eval(); res = []
    with torch.no_grad():
        for q in range(len(SIG_LEVELS)):
            rng = np.random.default_rng(rng_seed + q)
            Zq = torch.tensor(compress(data['Z'], q, rng), dtype=torch.float32)
            s, _ = m(Zq); res.append(roc_auc_score(data['y'], s.numpy()))
    return res

def train(kind, tr, va, lam=1.0, seed=0, epochs=60, cls=True):
    torch.manual_seed(seed); rng = np.random.default_rng(seed + 100)
    m = Model(use_proj=kind in ('P2', 'P3', 'P4', 'CONS_ONLY'))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Ztr = tr['Z']; ytr = torch.tensor(tr['y'], dtype=torch.float32); n = len(ytr)
    best, best_state, bad = -1, None, 0
    bce = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        m.train(); perm = rng.permutation(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            zs = torch.tensor(Ztr[idx], dtype=torch.float32)
            q = rng.choice(TRAIN_Q)
            zc = torch.tensor(compress(Ztr[idx], q, rng), dtype=torch.float32)
            y = ytr[idx]
            if kind == 'B0':
                s, _ = m(zs); loss = bce(s, y)
            else:
                ss, fs = m(zs); sc, fc = m(zc)
                loss = (bce(ss, y) + bce(sc, y)) if cls else 0.0
                if kind == 'P3' or kind == 'CONS_ONLY': loss = loss + lam * loss_consistency(fs, fc)
                if kind == 'P4': loss = loss + lam * loss_contrastive(fs, fc)
            opt.zero_grad(); loss.backward(); opt.step()
        if kind == 'CONS_ONLY': continue
        v = np.mean(auroc_per_q(m, va, 999)[:4])           # validasi: source + CRF yang dilatih
        if v > best + 1e-4: best, best_state, bad = v, {k: x.clone() for k, x in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8: break                              # early stopping
    if best_state is not None: m.load_state_dict(best_state)
    return m, best

def build(seed):
    rng = np.random.default_rng(seed)
    def split(n):
        Z = np.concatenate([make_clean(n, 0, rng), make_clean(n, 1, rng)])
        y = np.r_[np.zeros(n), np.ones(n)]; return {'Z': Z, 'y': y}
    return split(600), split(200), split(300)

if __name__ == '__main__':
    seeds = [0, 1, 2]; out = {}
    for sd in seeds:
        tr, va, te = build(sd)
        r = {}
        for kind in ['B0', 'B1', 'P2']:
            m, _ = train(kind, tr, va, seed=sd); r[kind] = auroc_per_q(m, te, 5000)
        for kind in ['P3', 'P4']:                           # pilih λ dengan validation set
            best = (-1, None, None)
            for lam in [0.1, 1.0, 10.0]:
                m, v = train(kind, tr, va, lam=lam, seed=sd)
                if v > best[0]: best = (v, lam, m)
            r[kind] = auroc_per_q(best[2], te, 5000); r[kind + '_lam'] = best[1]
            if kind == 'P3': torch.save(best[2].state_dict(), f'./p3_seed{sd}.pt')
        out[sd] = r; print(sd, json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in r.items()}), flush=True)
    json.dump(out, open('./results.json', 'w'))

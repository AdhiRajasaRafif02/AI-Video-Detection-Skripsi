import numpy as np, torch, json
from sim_projector import *
res = json.load(open('results.json'))
print("== AUROC mean±sd over seeds ==")
for k in ['B0','B1','P2','P3','P4']:
    a = np.array([res[s][k] for s in res]); print(k, ' '.join(f"{m:.3f}±{s:.3f}" for m,s in zip(a.mean(0),a.std(0))))
print("lam P3", [res[s]['P3_lam'] for s in res], "P4", [res[s]['P4_lam'] for s in res])
rows=[]
for sd in [0,1,2]:
    tr,va,te = build(sd)
    m = Model(True); m.load_state_dict(torch.load(f'p3_seed{sd}.pt')); m.eval()
    W = m.proj.weight.detach()
    cn = W.norm(dim=0).numpy()
    Z = te['Z']; y = te['y']
    out={'col_norm_dim0_31':cn[:32].mean(),'col_norm_dim32_63':cn[32:].mean()}
    with torch.no_grad():
        f0r = geometry(torch.tensor(Z,dtype=torch.float32)); f0p = m.feats(torch.tensor(Z,dtype=torch.float32))
        for q in range(6):
            Zq = torch.tensor(compress(Z,q,np.random.default_rng(5000+q)),dtype=torch.float32)
            fr = geometry(Zq); fp = m.feats(Zq)
            # mean curvature = kolom index 13+4 = 17 (mean θ)
            gap_r = (fr[y==1,17].mean()-fr[y==0,17].mean()).item(); gap_p=(fp[y==1,17].mean()-fp[y==0,17].mean()).item()
            mr = fr[y==0,17].mean().item(); mp = fp[y==0,17].mean().item()
            dr = (((fr-f0r)/(f0r.std(0)+1e-6))**2).mean().sqrt().item(); dp=(((fp-f0p)/(f0p.std(0)+1e-6))**2).mean().sqrt().item()
            out[f'q{q}']=dict(gap_raw=gap_r,gap_proj=gap_p,realcurv_raw=mr,realcurv_proj=mp,drift_raw=dr,drift_proj=dp)
    rows.append(out)
print("== P3 projector analysis (mean over seeds) ==")
print("col norm dims0-31 %.3f  dims32-63 %.3f (init = 1.000)"%(np.mean([r['col_norm_dim0_31'] for r in rows]),np.mean([r['col_norm_dim32_63'] for r in rows])))
for q in range(6):
    g=lambda k: np.mean([r[f'q{q}'][k] for r in rows])
    print(f"q{q} gap raw {g('gap_raw'):6.1f} proj {g('gap_proj'):6.1f} | real mean curv raw {g('realcurv_raw'):6.1f} proj {g('realcurv_proj'):6.1f} | drift raw {g('drift_raw'):.3f} proj {g('drift_proj'):.3f}")

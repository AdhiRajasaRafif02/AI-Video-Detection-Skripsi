import numpy as np, torch, sim_projector as S
from sim_projector import *
def stats(m, te):
    W=m.proj.weight.detach(); cn=W.norm(dim=0).numpy()
    Z=te['Z']
    with torch.no_grad():
        f0=m.feats(torch.tensor(Z,dtype=torch.float32))
        Zq=torch.tensor(compress(Z,3,np.random.default_rng(5003)),dtype=torch.float32); fq=m.feats(Zq)
        drift=(((fq-f0)/(f0.std(0)+1e-6))**2).mean().sqrt().item()
    return cn[:32].mean(), cn[32:].mean(), W.norm().item(), drift, f0[:,17].std().item()
for sd in [0,1,2]:
    tr,va,te=build(sd)
    m,_=train('P2',tr,va,seed=sd); print('P2 seed',sd,'colnorm0-31 %.3f 32-63 %.3f |W| %.2f drift_q3 %.3f std_meancurv %.2f'%stats(m,te))
# collapse demo: consistency only, no classification (standardized consistency)
tr,va,te=build(0)
m,_=train('CONS_ONLY',tr,va,lam=1.0,seed=0,epochs=40,cls=False); print('CONS_ONLY(std)  colnorm %.3f %.3f |W| %.2f drift %.3f std_meancurv %.2f'%stats(m,te))
# scale-cheat demo: unstandardized consistency only
S.loss_consistency=lambda fs,fc: ((fs-fc)**2).mean()
m,_=train('CONS_ONLY',tr,va,lam=1.0,seed=0,epochs=40,cls=False); print('CONS_ONLY(raw)  colnorm %.3f %.3f |W| %.2f drift %.3f std_meancurv %.2f'%stats(m,te))
print('init identity |W| = %.2f'%np.sqrt(64))

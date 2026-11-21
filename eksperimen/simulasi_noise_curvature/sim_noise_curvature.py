import numpy as np
rng=np.random.default_rng(0)
D,T=384,24
def curv(Z):
    d=np.diff(Z,axis=0); n=np.linalg.norm(d,axis=1)
    c=(d[:-1]*d[1:]).sum(1)/(n[:-1]*n[1:]); return np.degrees(np.arccos(np.clip(c,-1,1))).mean(), n.mean()
def traj(bend):
    # smooth path: step direction rotates slightly each step; 'bend' controls true curvature
    v=rng.normal(size=D); v/=np.linalg.norm(v); Z=[np.zeros(D)]
    for t in range(T-1):
        r=rng.normal(size=D); r-=r.dot(v)*v; r/=np.linalg.norm(r)
        v=np.cos(bend)*v+np.sin(bend)*r; Z.append(Z[-1]+v)
    return np.array(Z)
print("noise/step  real(30deg)  ai(70deg)  gap")
for s in [0,0.02,0.05,0.1,0.2,0.5]:
    R=[curv(traj(np.radians(30))+rng.normal(scale=s,size=(T,D)))[0] for _ in range(200)]
    A=[curv(traj(np.radians(70))+rng.normal(scale=s,size=(T,D)))[0] for _ in range(200)]
    print(f"{s*np.sqrt(D):9.2f}  {np.mean(R):9.1f}  {np.mean(A):9.1f}  {np.mean(A)-np.mean(R):6.1f}")

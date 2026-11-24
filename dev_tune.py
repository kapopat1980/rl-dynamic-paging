"""Hyper-parameter tuning on DEVELOPMENT seeds 100-105 only (test seeds 0-19 never used here)."""
import os, csv, itertools
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from multiprocessing import Pool
from paging_sim import run_method, MS
DS=list(range(100,106)); MODELS=['GM','RWP','MC','ROUT']
def job(a):
    grp,model,method,seed,over=a
    eps,rt=run_method(method,model,seed,None,**over)
    ev=[e for e in eps if e['phase']=='eval']
    return dict(grp=grp,model=model,method=method,seed=seed,param=';'.join(f'{k}={v}' for k,v in over.items()),
                overhead=np.mean([e['overhead'] for e in ev]),delay=np.mean([e['delay'] for e in ev]))
J=[]
for cv,kd in itertools.product([1.0,1.5,2.0],[0.5,0.8,1.0]):
    for mo in MODELS:
        for s in DS: J.append(('H',mo,'DCMA-H',s,dict(cv=cv,kd=kd)))
for m,k in itertools.product(MS,[1,3,6,7]):
    for mo in MODELS:
        for s in DS: J.append(('MB',mo,'MB-tuned',s,dict(mb_m=m,mb_k=k)))
for w in [1.0,2.0,4.0]:
    for mo in MODELS:
        for s in DS[:4]: J.append(('W','%s'%mo,'TabQ',s,dict(w=w))); J.append(('W',mo,'DQN',s,dict(w=w)))
if __name__=='__main__':
    with Pool(2) as p: out=p.map(job,J,chunksize=6)
    with open('dev_results.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print('done',len(out))

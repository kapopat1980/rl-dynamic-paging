"""Final experiments on held-out TEST seeds 200-219 (hyper-parameters fixed on dev seeds 100-105)."""
import os, sys, csv, time
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from multiprocessing import Pool
from paging_sim import run_method
SEEDS = list(range(200, 220))
KEYS = ['cost', 'delay', 'succ', 'lu', 'overhead', 'lam', 'mean_m', 'events']
MB = {'GM': (4, 7), 'RWP': (3, 7), 'MC': (4, 7), 'ROUT': (3, 7)}     # tuned on dev seeds
def tuned(method, model, over):
    o = dict(over)
    if method == 'MB-tuned': o.update(mb_m=MB[model][0], mb_k=MB[model][1])
    if method in ('TabQ', 'DQN'): o.setdefault('w', 1.0)                # tuned on dev seeds
    return o
def job(a):
    exp, model, method, seed, over = a
    o = tuned(method, model, over)
    eps, rt = run_method(method, model, seed, None, **o)
    ev = [e for e in eps if e['phase'] == 'eval']
    row = dict(exp=exp, model=model, method=method, seed=seed, param=';'.join(f'{k}={v}' for k, v in over.items()), runtime=rt)
    for k in KEYS: row[k] = float(np.mean([e[k] for e in ev]))
    cur = [dict(exp=exp, model=model, method=method, seed=seed, ep=e['ep'], phase=e['phase'], overhead=e['overhead'],
                cost=e['cost'], delay=e['delay'], lam=e['lam']) for e in eps] if exp.startswith('F1') else []
    return row, cur
ALL12 = ['Blanket', 'Sequential', 'LA-2step', 'Markov-Prob', 'TabQ', 'DQN', 'CDQN', 'DCMA-Page', 'DCMA-RL', 'DCMA-fixedArea', 'MB4-Seq', 'MB-tuned']
def jobs(exp):
    J = []
    if exp in ('F1a', 'F1b'):
        for mo in (['GM', 'RWP'] if exp == 'F1a' else ['MC', 'ROUT']):
            for m in ALL12:
                for s in SEEDS: J.append((exp, mo, m, s, {}))
    elif exp in ('F3a', 'F3b'):
        for lam in ([0.02, 0.05] if exp == 'F3a' else [0.1]):
            for sp in [0.25, 0.5, 1.0, 2.0, 3.0]:
                for m in ['Sequential', 'MB4-Seq', 'DCMA-fixedArea', 'DCMA-Page']:
                    for s in SEEDS: J.append((exp, 'GM', m, s, dict(lam=lam, speed=sp)))
    elif exp == 'F4':
        for mo in ['GM', 'MC']:
            for m in ['DCMA-noDual', 'DCMA-noWarm', 'DCMA-noHeading']:
                for s in SEEDS: J.append((exp, mo, m, s, {}))
    elif exp == 'F5':
        for m, ov in [('DCMA-Page', {}), ('DCMA-Page', dict(margin=0.1)), ('DCMA-noDual', {}), ('CDQN', {}), ('TabQ', {}), ('Sequential', {}), ('MB4-Seq', {})]:
            for s in SEEDS: J.append((exp, 'GM', m, s, dict(Dmax=1.5, **ov)))
    elif exp == 'F6':
        for b in [2.0, 5.0, 20.0]:
            for m in ['DCMA-Page', 'MB4-Seq', 'Sequential']:
                for s in SEEDS: J.append((exp, 'GM', m, s, dict(beta=b)))
        for ne in [5, 10, 40]:
            for m in ['DCMA-Page', 'TabQ']:
                for s in SEEDS: J.append((exp, 'GM', m, s, dict(E_tr=ne)))
    elif exp == 'F7':
        for m in ['DCMA-Page', 'TabQ', 'DQN']:
            for U in [30, 120, 240]:
                for s in SEEDS: J.append((exp, 'GM', m, s, dict(U=U)))
            for Rg in [4, 9, 12]:
                for s in SEEDS: J.append((exp, 'GM', m, s, dict(Rg=Rg)))
    elif exp == 'F8':
        for dm in [1.2, 1.5, 2.0, 3.0]:
            for s in SEEDS: J.append((exp, 'GM', 'DCMA-Page', s, dict(Dmax=dm)))
        for w in [0.5, 2.0, 4.0, 8.0]:
            for s in SEEDS: J.append((exp, 'GM', 'TabQ', s, dict(w=w)))
    return J
if __name__ == '__main__':
    for exp in sys.argv[1:]:
        t0 = time.time(); J = jobs(exp)
        with Pool(2) as p: out = p.map(job, J, chunksize=4)
        rows = [o[0] for o in out]; cur = [c for o in out for c in o[1]]
        with open(f'res2_{exp}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        if cur:
            with open(f'curves2_{exp}.csv', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(cur[0].keys())); w.writeheader(); w.writerows(cur)
        open(f'wall2_{exp}.txt', 'w').write(f'{time.time()-t0:.1f}\n'); print(exp, len(rows), 'runs', round(time.time() - t0, 1), 's', flush=True)

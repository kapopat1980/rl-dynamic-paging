import glob, os, re, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.makedirs('figs', exist_ok=True); os.makedirs('src/tables', exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .3, 'figure.dpi': 150})
R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('res2_*.csv'))], ignore_index=True); R.to_csv('results.csv', index=False)
CUR = pd.concat([pd.read_csv(f) for f in glob.glob('curves2_*.csv')]) if glob.glob('curves2_*.csv') else None
M12 = ['Blanket', 'Sequential', 'LA-2step', 'Markov-Prob', 'MB4-Seq', 'MB-tuned', 'TabQ', 'DQN', 'CDQN', 'DCMA-RL', 'DCMA-fixedArea', 'DCMA-Page']
COL = dict(zip(M12, ['#7f7f7f', '#8c564b', '#bcbd22', '#17becf', '#2ca02c', '#98df8a', '#1f77b4', '#ff7f0e', '#9467bd', '#f7b6d2', '#e377c2', '#d62728']))
MODELS = ['GM', 'RWP', 'MC', 'ROUT']; out = []
def P(*a): out.append(' '.join(str(x) for x in a))
def ci(x):
    x = np.asarray(x, float); return stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
def f(x, d=3): return f'{np.mean(x):.{d}f} ± {ci(x):.{d}f}'
def md(h, rows, w=None):
    s = '| ' + ' | '.join(h) + ' |\n|' + '|'.join(['-' * x for x in (w or [4] * len(h))]) + '|\n'
    for r in rows: s += '| ' + ' | '.join(str(c) for c in r) + ' |\n'
    return s
def save(n, s): open(f'src/tables/{n}.md', 'w').write(s)
F1 = R[R.exp.isin(['F1a', 'F1b'])]
def get(df, **kw):
    for k, v in kw.items(): df = df[df[k] == v]
    return df.sort_values('seed')
def wil(a, b):
    try: return stats.wilcoxon(a, b).pvalue
    except ValueError: return 1.0
def holm(ps):
    o = np.argsort(ps); adj = np.zeros(len(ps)); run = 0
    for rank, i in enumerate(o): run = max(run, min(1, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj
# ---- main tables
P('## F1 main comparison (test seeds 200-219; mean ± 95% CI)')
for mo in MODELS:
    rows = []
    for m in M12:
        d = get(F1, model=mo, method=m)
        rows.append([m, f(d.cost, 2), f(d.delay, 2), f(d.succ, 3), f(d.lu, 4), f(d.overhead, 3), f(d.runtime, 1), f'{100*(d.delay<=2.0).mean():.0f}'])
        P(f"{mo} | {m} | cost {f(d.cost,2)} | delay {f(d.delay,2)} | succ {f(d.succ,3)} | LU {f(d.lu,4)} | overhead {f(d.overhead,3)} | rt {f(d.runtime,1)} | feas {100*(d.delay<=2.0).mean():.0f}%")
    save(f'main_{mo}', md(['Method', 'Paging cost (msgs/call)', 'Delay (rounds)', 'Success (D≤3)', 'LU per user-step', 'Overhead Ω', 'Runtime (s/run)', 'Seeds with mean delay ≤ 2.0 (%)'], rows, [13, 11, 9, 9, 9, 10, 9, 12]))
# ---- significance
P('\n## Paired Wilcoxon, DCMA-Page minus comparator on overhead, Holm within model (11 comparisons)')
sigrows = {m: [m] for m in M12[:-1]}; SG = []
for mo in MODELS:
    b = get(F1, model=mo, method='DCMA-Page'); ps = []; names = []
    for m in M12[:-1]:
        o = get(F1, model=mo, method=m); ps.append(wil(b.overhead.values, o.overhead.values)); names.append((m, (b.overhead.values - o.overhead.values).mean(), (b.overhead.values < o.overhead.values).mean()))
    adj = holm(np.array(ps))
    for (m, dm, fr), p, pa in zip(names, ps, adj):
        P(f"{mo} | vs {m} | diff {dm:+.3f} | DCMA lower in {100*fr:.0f}% seeds | p={p:.2g} | holm={pa:.2g}"); sigrows[m].append(f'{dm:+.3f} ({pa:.2g})'); SG.append(dict(model=mo, vs=m, diff=dm, p=p, holm=pa, frac=fr))
save('sig', md(['Comparator'] + [f'{mo}: ΔΩ (Holm p)' for mo in MODELS], list(sigrows.values()), [14, 14, 14, 14, 14]))
pd.DataFrame(SG).to_csv('table_sig.csv', index=False)
# ---- convergence
if CUR is not None:
    P('\n## Convergence episodes (first training episode within 10% of final eval mean; 21 = never)'); rows = []
    for m in ['TabQ', 'DQN', 'CDQN', 'DCMA-RL', 'DCMA-fixedArea', 'DCMA-Page']:
        r = [m]
        for mo in MODELS:
            ev = get(F1, model=mo, method=m).set_index('seed').overhead; cs = []
            for s in ev.index:
                d = CUR[(CUR.model == mo) & (CUR.method == m) & (CUR.seed == s) & (CUR.phase == 'train')].sort_values('ep')
                h = np.nonzero(d.overhead.values <= 1.1 * ev[s])[0]; cs.append(h[0] + 1 if len(h) else 21)
            r.append(f'{np.mean(cs):.1f} ± {ci(cs):.1f}'); P(f'{mo} | {m} | {np.mean(cs):.1f}±{ci(cs):.1f} never {100*np.mean(np.array(cs)==21):.0f}%')
        rows.append(r)
    save('conv', md(['Method'] + MODELS, rows, [14, 12, 12, 12, 12]))
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.1))
    for ax, mo in zip(axs, MODELS):
        for m in ['TabQ', 'DQN', 'CDQN', 'DCMA-Page']:
            d = CUR[(CUR.model == mo) & (CUR.method == m)].pivot(index='ep', columns='seed', values='overhead')
            mu = d.mean(axis=1); h = d.apply(ci, axis=1); ax.plot(mu.index + 1, mu, color=COL[m], label=m); ax.fill_between(mu.index + 1, mu - h, mu + h, color=COL[m], alpha=.15)
        for m in ['Sequential', 'MB-tuned']: ax.axhline(get(F1, model=mo, method=m).overhead.mean(), color=COL[m], ls='--', lw=1, label=m)
        ax.axvline(20.5, color='k', ls=':', lw=.8); ax.set_title(mo); ax.set_xlabel('episode (train 1-20 | eval 21-24)')
    axs[0].set_ylabel('overhead (msgs / user / step)'); axs[0].legend(fontsize=6); plt.tight_layout(); plt.savefig('figs/fig2_learning_curves.png'); plt.close()
    P('first-episode overhead (GM): ' + ', '.join(f"{m} {CUR[(CUR.model=='GM')&(CUR.method==m)&(CUR.ep==0)].overhead.mean():.2f}" for m in ['TabQ', 'DQN', 'CDQN', 'DCMA-Page']))
# ---- fig main bars
fig, axs = plt.subplots(1, 4, figsize=(14, 3.3))
for ax, mo in zip(axs, MODELS):
    mu = [get(F1, model=mo, method=m).overhead.mean() for m in M12]; h = [ci(get(F1, model=mo, method=m).overhead) for m in M12]
    ax.bar(range(12), mu, yerr=h, color=[COL[m] for m in M12], capsize=1.5); ax.set_xticks(range(12)); ax.set_xticklabels(M12, rotation=70, ha='right', fontsize=6.5); ax.set_title(mo)
axs[0].set_ylabel('signalling overhead'); plt.tight_layout(); plt.savefig('figs/fig1_main_overhead.png'); plt.close()
# ---- regime map F3
F3 = pd.concat([R[R.exp.isin(['F3a', 'F3b'])]]); 
if len(F3):
    F3 = F3.copy(); F3['lam_'] = F3.param.str.extract(r'lam=([\d.]+)')[0].astype(float); F3['sp'] = F3.param.str.extract(r'speed=([\d.]+)')[0].astype(float)
    lams = sorted(F3.lam_.unique()); sps = sorted(F3.sp.unique()); D = np.zeros((len(lams), len(sps))); PP = []; lab = np.empty((len(lams), len(sps)), object)
    P('\n## F3 regime map (GM): DCMA-Page minus best of {Sequential, MB4-Seq, DCMA-fixedArea} by mean; raw p vs that best; Holm over cells')
    cells = []
    for i, l in enumerate(lams):
        for j, s in enumerate(sps):
            g = F3[(F3.lam_ == l) & (F3.sp == s)]; means = {m: g[g.method == m].overhead.mean() for m in ['Sequential', 'MB4-Seq', 'DCMA-fixedArea']}; b = min(means, key=means.get)
            x = get(g, method='DCMA-Page').overhead.values; y = get(g, method=b).overhead.values
            D[i, j] = x.mean() - y.mean(); cells.append((i, j, l, s, b, D[i, j], wil(x, y), x.mean()))
    adj = holm(np.array([c[6] for c in cells]))
    for c, a in zip(cells, adj):
        P(f'lam {c[2]} speed {c[3]}: best other {c[4]}; diff {c[5]:+.3f}; p={c[6]:.2g}; holm={a:.2g}; DCMA overhead {c[7]:.3f}'); lab[c[0], c[1]] = f'{c[5]:+.3f}' + ('*' if a < 0.05 else '')
    fig, ax = plt.subplots(figsize=(5.2, 3.0)); v = np.abs(D).max(); im = ax.imshow(D, cmap='RdBu', vmin=-v, vmax=v, aspect='auto')
    ax.set_xticks(range(len(sps))); ax.set_xticklabels(sps); ax.set_yticks(range(len(lams))); ax.set_yticklabels(lams); ax.set_xlabel('speed scale'); ax.set_ylabel('call rate per user-step')
    for i in range(len(lams)):
        for j in range(len(sps)): ax.text(j, i, lab[i, j], ha='center', va='center', fontsize=7)
    plt.colorbar(im, label='Ω(DCMA-Page) − Ω(best other)'); plt.tight_layout(); plt.savefig('figs/fig4_regime_map.png'); plt.close()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.1)); g5 = F3[F3.lam_ == 0.05]
    for ax, y, t in zip(axs, ['overhead', 'cost', 'delay'], ['overhead', 'paging cost (msgs/call)', 'paging delay (rounds)']):
        for m in ['Sequential', 'MB4-Seq', 'DCMA-fixedArea', 'DCMA-Page']:
            gg = g5[g5.method == m].groupby('sp')[y]; ax.errorbar(gg.mean().index, gg.mean(), yerr=gg.apply(ci), marker='o', ms=3, label=m, color=COL[m], capsize=2, lw=1)
        ax.set_xlabel('mean-speed scale'); ax.set_title(t)
    axs[0].legend(fontsize=6); plt.tight_layout(); plt.savefig('figs/fig3_cost_vs_mobility.png'); plt.close()
    rows = []
    for sp in sps:
        g = g5[g5.sp == sp]; rows.append([f'{sp:g}'] + [f(g[g.method == m].overhead) for m in ['Sequential', 'MB4-Seq', 'DCMA-fixedArea', 'DCMA-Page']])
    save('e2', md(['Speed scale (λ = 0.05)', 'Sequential', 'MB4-Seq', 'DCMA-fixedArea', 'DCMA-Page'], rows, [12, 12, 12, 12, 12]))
    rows = []
    for i, l in enumerate(lams): rows.append([f'{l:g}'] + [lab[i, j] for j in range(len(sps))])
    save('regime', md(['λ \\ speed'] + [f'{s:g}' for s in sps], rows, [8] + [8] * len(sps)))
# ---- F4 ablation
F4 = R[R.exp == 'F4']
if len(F4):
    P('\n## F4 ablation (final method = hybrid)'); rows = []; ab = {}
    names = [('DCMA-Page', 'Full DCMA-Page'), ('DCMA-noDual', 'No dual (w = 2)'), ('DCMA-noWarm', 'No warm start'), ('DCMA-noHeading', 'No heading / predicted centre'), ('DCMA-fixedArea', 'Fixed area (m = 4)'), ('DCMA-RL', 'RL threshold head (earlier version)')]
    for mo in ['GM', 'MC']:
        full = get(F1, model=mo, method='DCMA-Page')
        for m, lb in names:
            d = get(pd.concat([F1, F4]), model=mo, method=m); p = '–' if m == 'DCMA-Page' else f'{wil(full.overhead.values, d.overhead.values):.2g}'
            rows.append([mo, lb, f(d.overhead), f(d.cost, 2), f(d.delay, 2), f(d.succ, 3), f(d.lu, 4), p]); P(f'{mo} | {lb} | ov {f(d.overhead)} cost {d.cost.mean():.2f} delay {d.delay.mean():.2f} succ {d.succ.mean():.3f} LU {d.lu.mean():.4f} p={p}')
    save('abl', md(['Model', 'Variant', 'Overhead Ω', 'Paging cost', 'Delay', 'Success', 'LU per user-step', 'p vs full (raw)'], rows, [6, 24, 12, 9, 7, 9, 10, 10]))
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.1)); allv = pd.concat([F1, F4])
    for ax, y, t in zip(axs, ['overhead', 'cost', 'delay'], ['overhead', 'paging cost', 'delay']):
        for k, mo in enumerate(['GM', 'MC']):
            mu = [get(allv, model=mo, method=m)[y].mean() for m, _ in names]; h = [ci(get(allv, model=mo, method=m)[y]) for m, _ in names]
            ax.bar(np.arange(6) + 0.4 * k - 0.2, mu, 0.4, yerr=h, capsize=1.5, label=mo, color=['#d62728' if k == 0 else '#1f77b4'] * 6, alpha=1 if k == 0 else .6)
        ax.set_xticks(range(6)); ax.set_xticklabels(['full', 'noDual', 'noWarm', 'noHead', 'fixedArea', 'RLthr'], rotation=40, ha='right'); ax.set_title(t)
    axs[0].legend(fontsize=7); plt.tight_layout(); plt.savefig('figs/fig5_ablation.png'); plt.close()
# ---- F5 tight budget + F8 sweep
F5 = R[R.exp == 'F5']
if len(F5):
    P('\n## F5 tight budget Dmax=1.5 (GM)'); rows = []; F5 = F5.copy(); F5['lab'] = F5.method + np.where(F5.param.str.contains('margin'), ' (margin 0.1)', '')
    for lb in ['DCMA-Page', 'DCMA-Page (margin 0.1)', 'DCMA-noDual', 'CDQN', 'TabQ', 'Sequential', 'MB4-Seq']:
        d = F5[F5.lab == lb].sort_values('seed'); rows.append([lb, f(d.delay), f(d.cost, 2), f(d.overhead), f(d.succ, 3), f'{100*(d.delay<=1.5).mean():.0f}'])
        P(f"{lb}: delay {f(d.delay)} cost {f(d.cost,2)} ov {f(d.overhead)} succ {d.succ.mean():.3f} feas {100*(d.delay<=1.5).mean():.0f}% lam {d.lam.mean():.2f}")
    a = F5[F5.lab == 'DCMA-Page'].sort_values('seed')
    for lb in ['DCMA-Page (margin 0.1)', 'DCMA-noDual', 'CDQN', 'TabQ', 'MB4-Seq']:
        o = F5[F5.lab == lb].sort_values('seed'); P(f"F5 DCMA vs {lb}: ov diff {np.mean(a.overhead.values-o.overhead.values):+.3f} p={wil(a.overhead.values,o.overhead.values):.2g}; delay diff {np.mean(a.delay.values-o.delay.values):+.3f} p={wil(a.delay.values,o.delay.values):.2g}")
    save('e7', md(['Method', 'Delay (rounds)', 'Paging cost', 'Overhead Ω', 'Success (D≤3)', 'Seeds with delay ≤ 1.5 (%)'], rows, [20, 12, 11, 11, 11, 14]))
F8 = R[R.exp == 'F8']
if len(F8):
    F8 = F8.copy(); F8['Dm'] = F8.param.str.extract(r'Dmax=([\d.]+)')[0].astype(float); F8['w'] = F8.param.str.extract(r'w=([\d.]+)')[0].astype(float); rows = []
    P('\n## F8 delay budget sweep')
    fig, ax = plt.subplots(figsize=(4.8, 3.6)); g = get(F1, model='GM')
    for m in ['Blanket', 'Sequential', 'LA-2step', 'Markov-Prob', 'MB4-Seq', 'MB-tuned', 'TabQ', 'DQN', 'CDQN', 'DCMA-Page']:
        d = g[g.method == m]; ax.errorbar(d.delay.mean(), d.cost.mean(), xerr=ci(d.delay), yerr=ci(d.cost), fmt='o', color=COL[m], label=m, capsize=2, ms=4)
    dm = F8[F8.method == 'DCMA-Page'].groupby('Dm')[['delay', 'cost']].mean(); ax.plot(dm.delay, dm.cost, '-^', color=COL['DCMA-Page'], ms=4, lw=1, label='DCMA-Page (Dmax sweep)')
    for k, r in dm.iterrows(): ax.annotate(f'{k}', (r.delay, r.cost), fontsize=6, xytext=(3, 3), textcoords='offset points')
    tw = F8[F8.method == 'TabQ'].groupby('w')[['delay', 'cost']].mean(); ax.plot(tw.delay, tw.cost, '-s', color=COL['TabQ'], ms=3, lw=1, label='TabQ (w sweep)')
    ax.set_xlabel('mean paging delay (rounds)'); ax.set_ylabel('paging cost (msgs/call)'); ax.legend(fontsize=5.5); plt.tight_layout(); plt.savefig('figs/fig6_tradeoff.png'); plt.close()
    for dmv in sorted(F8.Dm.dropna().unique()):
        d = F8[(F8.method == 'DCMA-Page') & (F8.Dm == dmv)]; rows.append([f'DCMA-Page, D_max={dmv:g}', f(d.delay), f(d.cost, 2), f(d.overhead), f'{d.lam.mean():.2f}', f'{100*(d.delay<=dmv).mean():.0f}']); P(f'Dmax {dmv}: delay {f(d.delay)} cost {f(d.cost,2)} ov {f(d.overhead)} lam {d.lam.mean():.2f} feas {100*(d.delay<=dmv).mean():.0f}%')
    for wv in sorted(F8.w.dropna().unique()):
        d = F8[(F8.method == 'TabQ') & (F8.w == wv)]; rows.append([f'TabQ, w={wv:g}', f(d.delay), f(d.cost, 2), f(d.overhead), '–', '–']); P(f'TabQ w {wv}: delay {f(d.delay)} cost {f(d.cost,2)} ov {f(d.overhead)}')
    save('e3', md(['Configuration', 'Delay (rounds)', 'Paging cost', 'Overhead Ω', 'Final λ', 'Seeds with delay ≤ D_max (%)'], rows, [22, 11, 11, 11, 8, 14]))
# ---- F6 sensitivity
F6 = R[R.exp == 'F6']
if len(F6):
    F6 = F6.copy()
    for nm in ['beta', 'E_tr']: F6[nm] = F6.param.str.extract(rf'{nm}=([\d.]+)')[0].astype(float)
    base = get(F1, model='GM'); rows = []; P('\n## F6 sensitivity')
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.1))
    for ax, nm, ms_, dflt in [(axs[0], 'beta', ['Sequential', 'MB4-Seq', 'DCMA-Page'], 10.0), (axs[1], 'E_tr', ['TabQ', 'DCMA-Page'], 20.0)]:
        for m in ms_:
            d = pd.concat([F6[(F6.method == m) & F6[nm].notna()], base[base.method == m].assign(**{nm: dflt})]); gg = d.groupby(nm).overhead
            ax.errorbar(gg.mean().index, gg.mean(), yerr=gg.apply(ci), marker='o', ms=3, color=COL[m], label=m, capsize=2, lw=1); P(f"{nm} | {m}: " + ' '.join(f'[{k:g}] {f(g.overhead)}' for k, g in sorted(d.groupby(nm))))
        ax.set_xlabel('LU cost β' if nm == 'beta' else 'training episodes'); ax.legend(fontsize=6)
        if nm == 'beta': ax.set_xscale('log')
    axs[0].set_ylabel('overhead'); plt.tight_layout(); plt.savefig('figs/fig7_sensitivity.png'); plt.close()
    for nm, ms_, dflt, lb in [('beta', ['Sequential', 'MB4-Seq', 'DCMA-Page'], 10.0, 'LU cost β'), ('E_tr', ['TabQ', 'DCMA-Page'], 20.0, 'Training episodes')]:
        for v in sorted(set(F6[F6[nm].notna()][nm]) | {dflt}):
            r = [f'{lb} = {v:g}']
            for m in ['Sequential', 'MB4-Seq', 'TabQ', 'DCMA-Page']:
                if m not in ms_: r.append('–'); continue
                d = base[base.method == m] if v == dflt else F6[(F6.method == m) & (F6[nm] == v)]; r.append(f(d.overhead))
            rows.append(r)
    save('sens', md(['Setting (GM)', 'Sequential', 'MB4-Seq', 'TabQ', 'DCMA-Page'], rows, [18, 12, 12, 12, 12]))
# ---- F7 scalability
F7 = R[R.exp == 'F7']
if len(F7):
    F7 = F7.copy(); F7['U'] = F7.param.str.extract(r'U=(\d+)')[0].astype(float); F7['Rg'] = F7.param.str.extract(r'Rg=(\d+)')[0].astype(float); rows = []; P('\n## F7 scalability')
    base = get(F1, model='GM'); fig, axs = plt.subplots(1, 4, figsize=(13, 3.1))
    for j, (key, dflt, xl) in enumerate([('U', 60.0, 'users'), ('Rg', 6.0, 'grid radius (cells = 3Rg(Rg+1)+1)')]):
        for m in ['TabQ', 'DQN', 'DCMA-Page']:
            d = pd.concat([F7[(F7.method == m) & F7[key].notna()], base[base.method == m].assign(**{key: dflt})])
            for ax, y in zip(axs[2 * j:2 * j + 2], ['runtime', 'overhead']):
                g = d.groupby(key)[y]; ax.errorbar(g.mean().index, g.mean(), yerr=g.apply(ci), marker='o', ms=3, color=COL[m], label=m, capsize=2, lw=1); ax.set_xlabel(xl); ax.set_ylabel(y + (' (s/run)' if y == 'runtime' else ''))
            P(f'{key} | {m}: ' + ' '.join(f'[{k:g}] rt {g.runtime.mean():.1f}s ov {f(g.overhead)}' for k, g in sorted(d.groupby(key))))
    axs[0].legend(fontsize=6); plt.tight_layout(); plt.savefig('figs/fig8_scalability.png'); plt.close()
    for key, vals, lbl in [('U', [30, 60, 120, 240], 'users U'), ('Rg', [4, 6, 9, 12], 'grid radius R_g')]:
        for v in vals:
            r = [f'{lbl} = {v}' + (f' ({3*v*(v+1)+1} cells)' if key == 'Rg' else '')]
            for m in ['TabQ', 'DQN', 'DCMA-Page']:
                d = base[base.method == m] if (key == 'U' and v == 60) or (key == 'Rg' and v == 6) else F7[(F7.method == m) & (F7[key] == v)]; r.append(f'{d.runtime.mean():.1f} s / {f(d.overhead)}')
            rows.append(r)
    save('scal', md(['Setting (GM)', 'TabQ: runtime / Ω', 'DQN', 'DCMA-Page'], rows, [22, 18, 18, 18]))
P('\n## wall-clock (2 cores)')
for fn in sorted(glob.glob('wall2_*.txt')): P(fn[6:-4], open(fn).read().strip(), 's')
open('RESULTS_SUMMARY_auto.md', 'w').write('\n'.join(out)); print('\n'.join(out))

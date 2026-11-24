"""Simulator, baselines and DCMA-Page for dynamic paging (all data simulated)."""
import numpy as np, time
from functools import lru_cache

SQ3 = np.sqrt(3.0)
NB = np.array([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])
MS = [2, 3, 4, 6, 8]          # movement thresholds (paging-area radius A = m-1)
NT = 8                        # schedule templates
NA = 2 * NT                   # actions: template x centre (last / predicted)
KMAX = 9
TAU_B = [1, 3, 8, 20]         # elapsed-time bucket edges
SP_E = [0.04, 0.20]           # speed-estimate bucket edges (cells/step)
DEF = dict(Rg=6, U=60, T=150, lam=0.05, beta=10.0, pm=0.02, Dmax=2.0, Dsucc=3,
           speed=1.0, E_tr=20, E_ev=4, w=2.0, B=4)


@lru_cache(maxsize=8)
def grid(Rg, B=4):
    qs, rs = [], []
    for q in range(-Rg, Rg + 1):
        for r in range(-Rg, Rg + 1):
            if max(abs(q), abs(r), abs(q + r)) <= Rg:
                qs.append(q); rs.append(r)
    q = np.array(qs); r = np.array(rs); N = len(q)
    L = -np.ones((2 * Rg + 1, 2 * Rg + 1), int); L[q + Rg, r + Rg] = np.arange(N)
    dq = q[:, None] - q[None]; dr = r[:, None] - r[None]
    D = ((np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) // 2).astype(np.int16)
    key = ((q + Rg) // B) * 100 + (r + Rg) // B
    _, la = np.unique(key, return_inverse=True)
    same = la[:, None] == la[None]
    DM = int(D.max())
    G = dict(Rg=Rg, N=N, q=q, r=r, L=L, D=D, la=la, DM=DM,
             cnt=np.stack([(D <= d).sum(1) for d in range(DM + 1)], 1),
             lacnt=np.stack([((D <= d) & same).sum(1) for d in range(DM + 1)], 1),
             ringla=np.stack([((D == d) & same).sum(1) for d in range(DM + 1)], 1),
             lasize=same.sum(1), x=SQ3 * (q + r / 2.0), y=1.5 * r)
    return G


def xy2cell(G, x, y):
    Rg = G['Rg']
    qf = SQ3 / 3 * x - y / 3; rf = 2 * y / 3; sf = -qf - rf
    rq, rr, rs = np.round(qf), np.round(rf), np.round(sf)
    dq, dr, ds = abs(rq - qf), abs(rr - rf), abs(rs - sf)
    m1 = (dq > dr) & (dq > ds); m2 = (~m1) & (dr > ds)
    rq = np.where(m1, -rr - rs, rq); rr = np.where(m2, -rq - rs, rr)
    qi = rq.astype(int); ri = rr.astype(int)
    ok = (np.abs(qi) <= Rg) & (np.abs(ri) <= Rg) & (np.abs(qi + ri) <= Rg)
    idx = G['L'][np.clip(qi + Rg, 0, 2 * Rg), np.clip(ri + Rg, 0, 2 * Rg)]
    return np.where(ok, idx, -1)


# ------------------------------------------------------------------ mobility
class Mobility:
    def __init__(self, G, model, U, speed, rng):
        self.G, self.model, self.U, self.rng = G, model, U, rng
        cls = rng.choice(3, size=U, p=[0.5, 0.3, 0.2])
        self.v = np.array([0.06, 0.25, 0.8])[cls] * speed * rng.lognormal(0, 0.2, U)  # cells/step
        self.cell = rng.integers(0, G['N'], U)
        N = G['N']
        if model in ('GM', 'RWP', 'ROUT'):
            self.xy = np.stack([G['x'][self.cell], G['y'][self.cell]], 1) + rng.uniform(-.5, .5, (U, 2))
            c = xy2cell(G, self.xy[:, 0], self.xy[:, 1]); bad = c < 0
            self.xy[bad] = np.stack([G['x'][self.cell[bad]], G['y'][self.cell[bad]]], 1)
            self.cell = xy2cell(G, self.xy[:, 0], self.xy[:, 1])
        if model == 'GM':
            self.sp = self.v * SQ3; self.th = rng.uniform(0, 2 * np.pi, U); self.thb = self.th.copy()
        elif model == 'RWP':
            self.pause = np.zeros(U); self._newtarget(np.arange(U))
        elif model == 'ROUT':
            home = rng.integers(0, N, U)
            work = np.array([rng.choice(np.nonzero(G['D'][h] <= max(3, G['Rg']))[0]) for h in home])
            leis = np.array([rng.choice(np.nonzero(G['D'][h] <= max(3, G['Rg']))[0]) for h in home])
            self.anch = np.stack([work, home, leis, home], 1); self.ptr = np.zeros(U, int)
            self.xy = np.stack([G['x'][home], G['y'][home]], 1).astype(float); self.cell = home.copy()
            self.pause = rng.geometric(0.04, U).astype(float); self._newtarget(np.arange(U))
        else:
            self.dir = rng.integers(0, 6, U)
            self.qa = G['q'][self.cell].copy(); self.ra = G['r'][self.cell].copy()
            self.offs = np.array([0, 1, -1, 2, -2, 3]); self.pc = np.cumsum([.55, .15, .15, .04, .04, .07])

    def _newtarget(self, u):
        G = self.G
        if self.model == 'ROUT':
            tc = self.anch[u, self.ptr[u]]; self.ptr[u] = (self.ptr[u] + 1) % 4
        else: tc = self.rng.integers(0, G['N'], len(u))
        if not hasattr(self, 'tg'):
            self.tg = np.zeros((self.U, 2)); self.vs = np.zeros(self.U)
        self.tg[u] = np.stack([G['x'][tc], G['y'][tc]], 1)
        self.vs[u] = self.v[u] * self.rng.uniform(.5, 1.5, len(u)) * SQ3

    def step(self):
        G, rng, U = self.G, self.rng, self.U
        if self.model == 'GM':
            a = 0.9; vm = self.v * SQ3
            self.sp = np.maximum(0, a * self.sp + (1 - a) * vm + np.sqrt(1 - a * a) * 0.3 * vm * rng.standard_normal(U))
            self.th = a * self.th + (1 - a) * self.thb + np.sqrt(1 - a * a) * 0.6 * rng.standard_normal(U)
            nxy = self.xy + self.sp[:, None] * np.stack([np.cos(self.th), np.sin(self.th)], 1)
            c = xy2cell(G, nxy[:, 0], nxy[:, 1]); bad = c < 0
            self.th[bad] += np.pi; self.thb[bad] = self.th[bad]
            ok = ~bad; self.xy[ok] = nxy[ok]; self.cell[ok] = c[ok]
        elif self.model in ('RWP', 'ROUT'):
            mov = self.pause <= 0
            d = self.tg - self.xy; dist = np.hypot(d[:, 0], d[:, 1]) + 1e-9
            arr = mov & (dist <= self.vs)
            go = mov & ~arr
            self.xy[go] += (d[go] / dist[go, None]) * self.vs[go, None]
            self.xy[arr] = self.tg[arr]
            self.pause[arr] = rng.geometric(0.04 if self.model == 'ROUT' else 0.1, arr.sum())
            self.pause[~mov] -= 1
            done = np.nonzero((~mov) & (self.pause <= 0))[0]
            if done.size: self._newtarget(done)
            c = xy2cell(G, self.xy[:, 0], self.xy[:, 1]); c[c < 0] = self.cell[c < 0]
            self.cell = c
        else:
            nh = np.minimum(rng.poisson(self.v), 3)
            for h in range(3):
                m = nh > h
                if not m.any(): break
                u = np.nonzero(m)[0]
                off = self.offs[np.searchsorted(self.pc, rng.random(len(u)))]
                nd = (self.dir[u] + off) % 6
                nq = self.qa[u] + NB[nd, 0]; nr = self.ra[u] + NB[nd, 1]
                Rg = G['Rg']; ok = (np.abs(nq) <= Rg) & (np.abs(nr) <= Rg) & (np.abs(nq + nr) <= Rg)
                self.dir[u] = np.where(ok, nd, (nd + 3) % 6)
                self.qa[u[ok]] = nq[ok]; self.ra[u[ok]] = nr[ok]
            self.cell = G['L'][self.qa + G['Rg'], self.ra + G['Rg']]
        return self.cell


# --------------------------------------------------------------- schedules
def build_templates():
    RAD = -np.ones((len(MS), NT, KMAX), int); NR = np.zeros((len(MS), NT), int)
    for mi, m in enumerate(MS):
        A = m - 1
        T = [[A], [0, A], [1, A], [1, 2, A], [0, 1, 2, A], [2, A],
             [int(np.ceil(A / 3)), int(np.ceil(2 * A / 3)), A], list(range(A + 1))]
        for k, rl in enumerate(T):
            rl = sorted(set(min(x, A) for x in rl))
            RAD[mi, k, :len(rl)] = rl; NR[mi, k] = len(rl)
    return RAD, NR


RAD, NR = build_templates()


def tau_b(tau):
    return np.searchsorted(TAU_B, tau, side='left')   # 0..4


def sp_b(sp):
    return np.searchsorted(SP_E, sp, side='right')    # 0..2


class State:
    def __init__(self, G, mob, lam_u, m0):
        U = mob.U; self.G = G; self.U = U
        self.cell = mob.cell.copy(); self.last = self.cell.copy(); self.prev = self.cell.copy()
        self.t_last = np.zeros(U); self.t_prev = -np.ones(U)
        self.cnt = np.zeros(U); self.m = np.full(U, m0); self.mi = np.full(U, MS.index(m0))
        self.sp = np.full(U, 0.1); self.gap = np.full(U, 40.0); self.t_call = np.zeros(U)
        self.hit = np.full(U, 0.5)
        self.ep_s = -np.ones(U, int); self.ep_a = np.zeros(U, int); self.ep_t = np.zeros(U)
        self.ep_x = np.zeros((U, 4))

    def predict(self, ev, t):
        G = self.G; Rg = G['Rg']; A = self.m[ev] - 1
        L, P = self.last[ev], self.prev[ev]
        dt = np.maximum(1.0, self.t_last[ev] - self.t_prev[ev]); tau = t - self.t_last[ev]
        dq = np.rint((G['q'][L] - G['q'][P]) / dt * tau); dr = np.rint((G['r'][L] - G['r'][P]) / dt * tau)
        dist = (np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) / 2
        sc = np.where(dist > A, A / np.maximum(dist, 1), 1.0)
        dq = np.rint(dq * sc).astype(int); dr = np.rint(dr * sc).astype(int)
        nq = G['q'][L] + dq; nr = G['r'][L] + dr
        ok = (np.abs(nq) <= Rg) & (np.abs(nr) <= Rg) & (np.abs(nq + nr) <= Rg)
        idx = G['L'][np.clip(nq + Rg, 0, 2 * Rg), np.clip(nr + Rg, 0, 2 * Rg)]
        return np.where(ok & (idx >= 0), idx, L)

    def update_report(self, u, t, kind):
        G = self.G; c = self.cell[u]
        if kind == 1:
            pred = self.predict(u, t)
            moved = pred != self.last[u]
            imp = (G['D'][c, pred] < G['D'][c, self.last[u]]).astype(float)
            self.hit[u] = np.where(moved, 0.8 * self.hit[u] + 0.2 * imp, self.hit[u])
            self.gap[u] = 0.7 * self.gap[u] + 0.3 * (t - self.t_call[u]); self.t_call[u] = t
        d = G['D'][c, self.last[u]]; dt = np.maximum(1.0, t - self.t_last[u])
        self.sp[u] = 0.7 * self.sp[u] + 0.3 * d / dt
        self.prev[u] = self.last[u]; self.t_prev[u] = self.t_last[u]
        self.last[u] = c; self.t_last[u] = t; self.cnt[u] = 0

    def exec_schedule(self, ev, k, use_pred, t, rng, pm):
        G = self.G; D = G['D']; E = len(ev)
        mi = self.mi[ev]; A = self.m[ev] - 1; last = self.last[ev]; cell = self.cell[ev]
        rad = RAD[mi, k].copy(); nr = NR[mi, k]; ar = np.arange(E)
        rad[ar, nr - 1] = 999
        c0 = last.copy()
        if use_pred.any():
            pr = self.predict(ev, t); c0 = np.where(use_pred, pr, last)
        dc = D[cell, c0]
        j = (dc[:, None] <= rad).argmax(1)
        final = j == nr - 1
        rj = rad[ar, j]
        cost = np.where(final, G['cnt'][last, A], 0).astype(float)
        nf = ~final
        lp = nf & ~use_pred
        if lp.any(): cost[lp] = G['cnt'][last[lp], np.minimum(rj[lp], G['DM'])]
        pp = nf & use_pred
        if pp.any():
            cost[pp] = ((D[c0[pp]] <= rj[pp, None]) & (D[last[pp]] <= A[pp, None])).sum(1)
        delay = (j + 1).astype(float)
        miss = rng.random(E) < pm
        delay += miss; cost += miss * G['cnt'][last, A]
        return cost, delay


# ----------------------------------------------------------------- policies
class StaticLA:
    reporting = 'LA'

    def __init__(self, kind, rng, P):
        self.kind, self.rng, self.P = kind, rng, P; self.learn = False; self.lam = 0.0

    def init_users(self, S): pass
    def begin_episode(self, S, ep, learn): pass
    def end_step(self, t): pass
    def on_report(self, *a): pass

    def page(self, S, ev, t):
        G = S.G; last = S.last[ev]; d = G['D'][S.cell[ev], last]; size = G['lasize'][last].astype(float)
        if self.kind == 'blanket': cost, delay = size, np.ones(len(ev))
        elif self.kind == 'seq': cost, delay = G['lacnt'][last, d].astype(float), (d + 1).astype(float)
        else:
            cost = np.where(d == 0, 1.0, size); delay = np.where(d == 0, 1.0, 2.0)
        miss = self.rng.random(len(ev)) < self.P['pm']
        return cost + miss * size, delay + miss


class StaticMB:
    """Non-learning movement-based reporting (m=4) with a fixed schedule template (confound control)."""
    reporting = 'MB'
    KS = {'MB4-Blanket': 0, 'MB4-2step': 1, 'MB4-Seq': 7}

    def __init__(self, name, rng, P):
        if name == 'MB-tuned': self.k = P.get('mb_k', 7); self.m = P.get('mb_m', 4)
        else: self.k = self.KS[name]; self.m = 4
        self.rng, self.P = rng, P; self.lam = 0.0; self.learn = False

    def init_users(self, S): S.m[:] = self.m; S.mi[:] = MS.index(self.m)
    def begin_episode(self, S, ep, learn): self.init_users(S)
    def end_step(self, t): pass
    def on_report(self, *a): pass

    def page(self, S, ev, t):
        k = np.full(len(ev), self.k)
        return S.exec_schedule(ev, k, np.zeros(len(ev), bool), t, self.rng, self.P['pm'])


class MarkovProb:
    """Probability-based paging: ring distance distribution given (elapsed, speed) learned
    online (Markov ring-chain prior), rings grouped into K rounds by probability density."""
    reporting = 'LA'

    def __init__(self, rng, P, K=3):
        self.rng, self.P, self.K = rng, P, K; self.lam = 0.0
        self.R = 10
        self.counts = np.ones((15, self.R)) * 0.5
        self.counts[:, 0] += 1.0; self.counts[:, 1] += 0.6
        self.grp = np.zeros((15, self.R), int); self.learn = True; self.replan(None)

    def replan(self, G):
        ring = np.array([1] + [6 * d for d in range(1, self.R)], float)
        for c in range(15):
            p = self.counts[c] / self.counts[c].sum()
            order = np.argsort(-p / ring)
            cum = np.cumsum(p[order]); g = np.minimum((cum - 1e-9) * self.K, self.K - 1).astype(int)
            g = np.minimum(np.floor((cum - p[order] / 2) * self.K).astype(int), self.K - 1)
            self.grp[c, order] = g

    def init_users(self, S): pass
    def begin_episode(self, S, ep, learn):
        self.learn = learn
        if learn: self.replan(S.G)
    def end_step(self, t): pass
    def on_report(self, *a): pass

    def page(self, S, ev, t):
        G = S.G; last = S.last[ev]; d = np.minimum(G['D'][S.cell[ev], last], self.R - 1)
        ctx = tau_b(t - S.t_last[ev]) * 3 + sp_b(S.sp[ev])
        g = self.grp[ctx, d]
        rl = np.pad(G['ringla'][last][:, :self.R], ((0, 0), (0, max(0, self.R - min(self.R, G['ringla'].shape[1])))))
        cost = (rl * (self.grp[ctx] <= g[:, None])).sum(1).astype(float)
        delay = g + 1.0
        if self.learn: np.add.at(self.counts, (ctx, d), 1.0)
        size = G['lasize'][last].astype(float)
        miss = self.rng.random(len(ev)) < self.P['pm']
        return cost + miss * size, delay + miss


def ring_prior(P):
    """Prior ring-distance distribution from a ring-level Markov chain (used for warm start)."""
    Rm = 12
    M = np.zeros((Rm, Rm)); M[0, 1] = 1
    for d in range(1, Rm - 1):
        M[d, d - 1] = M[d, d] = M[d, d + 1] = 1 / 3
    M[Rm - 1, Rm - 2] = M[Rm - 1, Rm - 1] = .5
    out = np.zeros((5, 3, Rm)); taus = [0.5, 2, 5, 13, 30]; vs = [0.03, 0.12, 0.5]
    from math import exp, factorial
    for si, v in enumerate(vs):
        Ps = sum(exp(-v) * v ** n / factorial(n) * np.linalg.matrix_power(M, n) for n in range(6))
        for ei, ta in enumerate(taus):
            p = np.zeros(Rm); p[0] = 1
            p = p @ np.linalg.matrix_power(Ps, max(1, int(round(ta))))
            out[ei, si] = p
    return out


def cum_cells(r): return 1 + 3 * r * (r + 1)


class QPolicy:
    reporting = 'MB'

    def __init__(self, name, rng, P, mode='tab', dual=False, warm=False, heading=False,
                 adaptive=True, pred_actions=True, gamma=0.98, thr='rl'):
        self.name, self.rng, self.P = name, rng, P
        self.mode, self.dual, self.warm, self.heading, self.adaptive = mode, dual, warm, heading, adaptive
        self.thr = thr if adaptive else 'fixed'; self.pred_actions = pred_actions; self.gamma = gamma; self.rc = 5.0; self.rt = 10.0
        self.lam = P['w'] if not dual else 2.0; self.eta = 0.5
        self.eps = 0.3; self.learn = True; self.blk = []; self.na = NA if pred_actions else NT
        self.H = 3 if heading else 1
        self.nS = 5 * 5 * 3 * self.H; self.nM = 3 * 3 * 5
        if mode == 'tab':
            self.Lc = np.zeros((self.nS, NA)); self.Ld = np.zeros((self.nS, NA)); self.n = np.zeros((self.nS, NA))
            self.Qm = np.zeros((self.nM, len(MS))); self.nm = np.zeros((self.nM, len(MS)))
            if warm: self._warm()
        else:
            k = 2 if dual else 1
            self.k = k
            self.net_s = MLP(5, k * NA, rng); self.net_m = MLP(4, len(MS), rng)
            self.tgt_m = self.net_m.copy(); self.buf_s = []; self.buf_m = []; self.nupd = 0

    # ---- warm start from ring-chain prior
    def _warm(self):
        pr = ring_prior(self.P); w0 = 2.0; pm = self.P['pm']
        for ei in range(5):
            for mi, m in enumerate(MS):
                A = m - 1
                for si in range(3):
                    p = pr[ei, si, :A + 1]; p = p / p.sum()
                    for k in range(NT):
                        rl = RAD[mi, k, :NR[mi, k]]
                        ec = ed = 0.0
                        for d in range(A + 1):
                            j = next((jj for jj, r in enumerate(rl[:-1]) if d <= r), len(rl) - 1)
                            c = cum_cells(A) if j == len(rl) - 1 else cum_cells(int(rl[j]))
                            ec += p[d] * (c + pm * cum_cells(A)); ed += p[d] * (j + 1 + pm)
                        for h in range(self.H):
                            s = ((ei * 5 + mi) * 3 + si) * self.H + h
                            for c0 in (0, NT):
                                if self.dual:
                                    self.Lc[s, k + c0] = ec; self.Ld[s, k + c0] = ed
                                else: self.Lc[s, k + c0] = ec + self.lam * ed
                                self.n[s, k + c0] = 3

    # ---- indices / features
    def sidx(self, S, u, t):
        h = np.minimum((S.hit[u] * 3).astype(int), 2) if self.heading else 0
        return ((tau_b(t - S.t_last[u]) * 5 + S.mi[u]) * 3 + sp_b(S.sp[u])) * self.H + h

    def midx(self, S, u):
        cr = np.where(S.gap[u] < 20, 2, np.where(S.gap[u] < 60, 1, 0))
        return (sp_b(S.sp[u]) * 3 + cr) * 5 + S.mi[u]

    def xs(self, S, u, t):
        return np.stack([np.log1p(t - S.t_last[u]) / 3.5, S.m[u] / 8.0, np.minimum(S.sp[u], 1.5),
                         S.hit[u], 10.0 / S.gap[u]], 1)

    def xm(self, S, u):
        return np.stack([np.minimum(S.sp[u], 1.5), 10.0 / S.gap[u], S.m[u] / 8.0, S.hit[u]], 1)

    # ---- episode control
    def begin_episode(self, S, ep, learn):
        self.learn = learn; E = self.P['E_tr']
        self.eps = max(0.03, 0.3 * (1 - ep / max(1, E * 0.7))) if learn else 0.0
        self.blk = []
        self.init_users(S)

    def analytic_m(self, S, u):
        cv, kd, beta = self.P.get('cv', 1.5), self.P.get('kd', 0.8), self.P['beta']
        v = np.maximum(S.sp[u] * cv, 1e-3); lam = 1.0 / np.maximum(S.gap[u], 1.0); r = v / (v + lam)
        out = []
        for m in MS:
            j = np.arange(m); Pj = (r[:, None] ** j[None, :]) * (1 - r)[:, None]
            d = np.minimum(np.ceil(kd * j), m - 1); cp = 1 + 3 * d * (d + 1)
            q = r ** m
            out.append((beta * q + (Pj * cp[None, :]).sum(1)) / np.maximum((1 - q) / lam, 1e-6))
        return np.stack(out, 1).argmin(1)

    def init_users(self, S):
        if self.thr == 'fixed':
            S.m[:] = 4; S.mi[:] = 2; return
        u = np.arange(S.U)
        if self.thr == 'analytic':
            a = self.analytic_m(S, u); S.mi[u] = a; S.m[u] = np.array(MS)[a]; return
        self._choose(S, u, 0)

    def _choose(self, S, u, t):
        lam = self.lam
        if self.mode == 'tab':
            s = self.midx(S, u); q = self.Qm[s]
        else:
            q = self.net_m.predict(self.xm(S, u)) * 50
        a = q.argmin(1)
        ex = self.rng.random(len(u)) < self.eps
        a = np.where(ex, self.rng.integers(0, len(MS), len(u)), a)
        S.mi[u] = a; S.m[u] = np.array(MS)[a]
        S.ep_s[u] = self.midx(S, u) if self.mode == 'tab' else 0
        S.ep_a[u] = a; S.ep_t[u] = t
        if self.mode != 'tab': S.ep_x[u] = self.xm(S, u)

    def on_report(self, S, u, kind, cost, delay, t):
        if self.thr == 'fixed': return
        if self.thr == 'analytic':
            a = self.analytic_m(S, u); S.mi[u] = a; S.m[u] = np.array(MS)[a]; return
        c = np.full(len(u), self.P['beta']) if kind == 0 else cost + self.lam * delay
        had = S.ep_s[u] >= 0
        if self.learn and had.any():
            uu = u[had]; cc = c[had]; dt = np.maximum(1.0, t - S.ep_t[uu]); g = dt
            rho = self.rc / self.rt; self.rc = 0.9995 * self.rc + cc.sum(); self.rt = 0.9995 * self.rt + dt.sum()
            if self.mode == 'tab':
                s2 = self.midx(S, uu); s1 = S.ep_s[uu]; a = S.ep_a[uu]
                tgt = cc - rho * dt + self.Qm[s2].min(1)
                for i in range(len(uu)):
                    self.nm[s1[i], a[i]] += 1; al = max(0.05, 1.0 / self.nm[s1[i], a[i]])
                    self.Qm[s1[i], a[i]] += al * (tgt[i] - self.Qm[s1[i], a[i]])
            else:
                x2 = self.xm(S, uu)
                for i in range(len(uu)):
                    self.buf_m.append((S.ep_x[uu[i]].copy(), S.ep_a[uu[i]], cc[i] - rho * g[i], g[i], x2[i]))
                self._train_m()
        S.ep_s[u] = 0 if self.mode != 'tab' else S.ep_s[u]
        self._choose(S, u, t)

    # ---- paging
    def page(self, S, ev, t):
        E = len(ev); lam = self.lam
        if self.mode == 'tab':
            s = self.sidx(S, ev, t); v = self.Lc[s] + (lam * self.Ld[s] if self.dual else 0)
        else:
            x = self.xs(S, ev, t); o = self.net_s.predict(x) * 50
            v = o[:, :NA] + (lam * o[:, NA:] if self.dual else 0)
        if not self.pred_actions: v = v.copy(); v[:, NT:] = np.inf
        a = v.argmin(1)
        ex = self.rng.random(E) < self.eps
        a = np.where(ex, self.rng.integers(0, self.na, E), a)
        k = a % NT; up = a >= NT
        cost, delay = S.exec_schedule(ev, k, up, t, self.rng, self.P['pm'])
        if self.learn:
            if self.mode == 'tab':
                for i in range(E):
                    self.n[s[i], a[i]] += 1; al = max(0.05, 1.0 / self.n[s[i], a[i]])
                    if self.dual:
                        self.Lc[s[i], a[i]] += al * (cost[i] - self.Lc[s[i], a[i]])
                        self.Ld[s[i], a[i]] += al * (delay[i] - self.Ld[s[i], a[i]])
                    else:
                        tg = cost[i] + self.lam * delay[i]
                        self.Lc[s[i], a[i]] += al * (tg - self.Lc[s[i], a[i]])
            else:
                for i in range(E):
                    tg = (cost[i], delay[i]) if self.dual else (cost[i] + self.lam * delay[i],)
                    self.buf_s.append((x[i], a[i], np.array(tg)))
                self._train_s()
        self.blk.append(delay.sum()); self.blk_n = getattr(self, 'blk_n', 0) + E
        return cost, delay

    def end_step(self, t):
        if self.dual and t % 10 == 0 and getattr(self, 'blk_n', 0) > 0:
            md = sum(self.blk) / self.blk_n
            self.lam = float(np.clip(self.lam + self.eta * (md - (self.P['Dmax'] - self.P.get('margin', 0.0))), 0.0, 60.0))
            self.blk = []; self.blk_n = 0

    def _train_s(self):
        if len(self.buf_s) < 128: return
        if len(self.buf_s) > 20000: self.buf_s = self.buf_s[-15000:]
        idx = self.rng.integers(0, len(self.buf_s), 64)
        X = np.array([self.buf_s[i][0] for i in idx]); a = np.array([self.buf_s[i][1] for i in idx])
        T = np.array([self.buf_s[i][2] for i in idx]) / 50
        self.net_s.train(X, a, T, NA)

    def _train_m(self):
        if len(self.buf_m) < 128: return
        if len(self.buf_m) > 20000: self.buf_m = self.buf_m[-15000:]
        idx = self.rng.integers(0, len(self.buf_m), 64)
        b = [self.buf_m[i] for i in idx]
        X = np.array([x[0] for x in b]); a = np.array([x[1] for x in b])
        c = np.array([x[2] for x in b]); g = np.array([x[3] for x in b]); X2 = np.array([x[4] for x in b])
        tgt = (c / 50 + self.tgt_m.predict(X2).min(1))[:, None]
        self.net_m.train(X, a, tgt, len(MS))
        self.nupd += 1
        if self.nupd % 200 == 0: self.tgt_m = self.net_m.copy()


class MLP:
    def __init__(self, din, dout, rng, h=32, lr=3e-3):
        self.p = [rng.normal(0, 1 / np.sqrt(din), (din, h)), np.zeros(h),
                  rng.normal(0, 1 / np.sqrt(h), (h, h)), np.zeros(h),
                  rng.normal(0, 0.1 / np.sqrt(h), (h, dout)), np.zeros(dout)]
        self.m = [np.zeros_like(x) for x in self.p]; self.v = [np.zeros_like(x) for x in self.p]
        self.lr = lr; self.t = 0

    def copy(self):
        n = MLP.__new__(MLP); n.p = [x.copy() for x in self.p]; return n

    def fwd(self, X):
        a1 = np.maximum(0, X @ self.p[0] + self.p[1]); a2 = np.maximum(0, a1 @ self.p[2] + self.p[3])
        return a1, a2, a2 @ self.p[4] + self.p[5]

    def predict(self, X): return self.fwd(X)[2]

    def train(self, X, act, tgt, nact):
        B = len(X); a1, a2, out = self.fwd(X); k = out.shape[1] // nact
        o = out.reshape(B, k, nact); ar = np.arange(B)
        ch = o[ar, :, act]; go = np.zeros_like(o); go[ar, :, act] = (ch - tgt) * 2 / B
        go = go.reshape(B, -1)
        g4 = a2.T @ go; gb4 = go.sum(0); d2 = (go @ self.p[4].T) * (a2 > 0)
        g2 = a1.T @ d2; gb2 = d2.sum(0); d1 = (d2 @ self.p[2].T) * (a1 > 0)
        g0 = X.T @ d1; gb0 = d1.sum(0)
        gr = [g0, gb0, g2, gb2, g4, gb4]; self.t += 1
        for i, g in enumerate(gr):
            g = np.clip(g, -5, 5)
            self.m[i] = 0.9 * self.m[i] + 0.1 * g; self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t); vh = self.v[i] / (1 - 0.999 ** self.t)
            self.p[i] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


METHODS = ['Blanket', 'Sequential', 'LA-2step', 'Markov-Prob', 'TabQ', 'DQN', 'CDQN', 'DCMA-Page']


def make_policy(name, rng, P):
    if name == 'Blanket': return StaticLA('blanket', rng, P)
    if name == 'Sequential': return StaticLA('seq', rng, P)
    if name == 'LA-2step': return StaticLA('la2', rng, P)
    if name == 'Markov-Prob': return MarkovProb(rng, P)
    if name == 'TabQ': return QPolicy(name, rng, P, 'tab')
    if name == 'DQN': return QPolicy(name, rng, P, 'dqn', heading=True)
    if name == 'CDQN': return QPolicy(name, rng, P, 'dqn', dual=True, heading=True)
    if name == 'DCMA-Page':   # final method: learned schedule + analytic per-user threshold
        return QPolicy(name, rng, P, 'tab', dual=True, warm=True, heading=True, thr='analytic')
    if name == 'DCMA-RL':     # earlier version: RL threshold head
        return QPolicy(name, rng, P, 'tab', dual=True, warm=True, heading=True, thr='rl')
    if name in StaticMB.KS or name == 'MB-tuned': return StaticMB(name, rng, P)
    # ablations of the final method
    if name == 'DCMA-noDual': return QPolicy(name, rng, P, 'tab', dual=False, warm=True, heading=True, thr='analytic')
    if name == 'DCMA-noWarm': return QPolicy(name, rng, P, 'tab', dual=True, warm=False, heading=True, thr='analytic')
    if name == 'DCMA-noHeading': return QPolicy(name, rng, P, 'tab', dual=True, warm=True, heading=False, pred_actions=False, thr='analytic')
    if name == 'DCMA-fixedArea': return QPolicy(name, rng, P, 'tab', dual=True, warm=True, heading=True, adaptive=False)
    raise ValueError(name)


def run_episode(G, model, pol, envseed, P, t_pol_learn, ep):
    rng = np.random.default_rng(envseed)
    U, T = P['U'], P['T']
    mob = Mobility(G, model, U, P['speed'], rng)
    lam_u = P['lam'] * rng.lognormal(0, 0.5, U); lam_u *= P['lam'] / lam_u.mean()
    S = State(G, mob, lam_u, 4)
    pol.begin_episode(S, ep, t_pol_learn)
    cost = delay = succ = n = lu = 0.0
    msum = 0.0
    for t in range(1, T + 1):
        new = mob.step().copy(); S.cnt += G["D"][S.cell, new]; S.cell = new
        if pol.reporting == 'LA': trig = G['la'][S.cell] != G['la'][S.last]
        else: trig = S.cnt >= S.m
        ul = np.nonzero(trig)[0]
        if ul.size:
            S.update_report(ul, t, 0); pol.on_report(S, ul, 0, None, None, t); lu += ul.size
        ev = np.nonzero(rng.random(U) < lam_u)[0]
        if ev.size:
            c, d = pol.page(S, ev, t)
            cost += c.sum(); delay += d.sum(); succ += (d <= P['Dsucc']).sum(); n += ev.size
            S.update_report(ev, t, 1); pol.on_report(S, ev, 1, c, d, t)
        pol.end_step(t); msum += S.m.mean()
    n = max(n, 1)
    return dict(cost=cost / n, delay=delay / n, succ=succ / n, lu=lu / (U * T),
                events=n, overhead=(cost + P['beta'] * lu) / (U * T), lam=getattr(pol, 'lam', 0.0),
                mean_m=msum / T)


def run_method(method, model, seed, P0=None, **over):
    P = dict(DEF); P.update(P0 or {}); P.update(over)
    G = grid(P['Rg'], P['B'])
    prng = np.random.default_rng(10_000_003 * (seed + 1) + (hash(method) % 9973) * 0 + sum(map(ord, method)))
    pol = make_policy(method, prng, P)
    t0 = time.perf_counter(); eps = []
    for ep in range(P['E_tr'] + P['E_ev']):
        learn = ep < P['E_tr']
        r = run_episode(G, model, pol, 1_000_003 * (seed + 1) + ep, P, learn, ep)
        r['ep'] = ep; r['phase'] = 'train' if learn else 'eval'; eps.append(r)
    rt = time.perf_counter() - t0
    return eps, rt

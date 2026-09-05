"""
SparseEA-AGDS search algorithm for SearchLibrium.

Implements the *Evolution algorithm with Adaptive Genetic operator and Dynamic
Scoring mechanism* of Wang et al. (Scientific Reports, 2025, "Evolution
algorithm with adaptive genetic operator and dynamic scoring mechanism for
large-scale sparse many-objective optimization").

SparseEA represents an individual as ``X = dec x mask``: a real vector ``dec``
and a binary ``mask`` controlling sparsity.  Mapped onto the choice-model
specification search this becomes:

    mask_d  ->  whether candidate variable ``d`` (``param.asvarnames``) is in
                the specification.
    dec_d   ->  the distribution / transform / correlation choice that variable
                carries once selected.

Three stackable strategies from the paper are implemented on top of the shared
``Search`` base class:

    (A) Adaptive genetic operator - per-individual crossover / mutation
        probabilities scale with the non-dominated rank
        ``P_s,i = (maxr - r_i + 1)/maxr`` (Eq. 5), ``P_c,i = pc0 * P_s,i``
        (Eq. 6), ``P_m,i = pm0 * P_s,i`` (Eq. 7).
    (B) Dynamic scoring mechanism - each generation the decision-variable score
        is recomputed from the current layers: ``S_i_r = maxr - r_i + 1``
        (Eq. 8), ``sumS_d = sum_i S_i_r * mask_{i,d}`` (Eq. 9),
        ``S_d = maxS - sumS_d + 1`` (Eq. 10); a binary tournament on ``S_d``
        chooses which mask bit to flip (Figs. 3-4).
    (C) Reference-point-based environmental selection (NSGA-III niching) keeps
        selection pressure under many objectives (``param.nb_crit > 1``); the
        single-objective case degenerates to best-N-by-first-objective.

The class mirrors ``HarmonySearch`` / ``SA``: construct with a ``Parameters``
object and call ``run()`` (which dispatches to ``run_search``).
"""

import copy
import datetime
import logging

import numpy as np

try:
    from .search import *          # Search, Solution, get_unique, BOUND, ...
except ImportError:                # running as a top-level module
    from search import *

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
#  NSGA-III reference-point helpers (numpy, self-contained)               #
# ---------------------------------------------------------------------- #
def reference_points(n_obj, divisions):
    """Das-Dennis structured reference points on the (n_obj-1)-simplex."""
    def _recurse(left, depth):
        if depth == n_obj - 1:
            return [[left]]
        pts = []
        for i in range(left + 1):
            for tail in _recurse(left - i, depth + 1):
                pts.append([i] + tail)
        return pts
    if n_obj == 1:
        return np.array([[1.0]])
    raw = _recurse(divisions, 0)
    return np.array(raw, dtype=float) / float(divisions)


def associate(points, ref):
    """Associate each point with the nearest reference line (index into ref)."""
    if len(points) == 0:
        return np.array([], dtype=int)
    ref_norm = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12)
    out = np.empty(len(points), dtype=int)
    for i, p in enumerate(points):
        proj = points[i] @ ref_norm.T
        perp = np.linalg.norm(p[None, :] - proj[:, None] * ref_norm, axis=1)
        out[i] = int(np.argmin(perp))
    return out


def perp_distance(points, ref, assoc):
    """Perpendicular distance of each point to its associated reference line."""
    d = np.empty(len(points))
    for i, p in enumerate(points):
        w = ref[assoc[i]]
        wn = w / (np.linalg.norm(w) + 1e-12)
        proj = np.dot(p, wn) * wn
        d[i] = np.linalg.norm(p - proj)
    return d


class SparseEAAGDS(Search):
    """SparseEA-AGDS evolutionary search (see module docstring)."""

    # ------------------------------------------------------------------ #
    #  construction                                                       #
    # ------------------------------------------------------------------ #
    def set_control_parameters(self, pop_size=20, maxiter=30, pc0=0.9,
                               pm0=None, ref_divisions=12, generate_plots=False):
        self.pop_size = int(pop_size)
        self.maxiter = int(maxiter)
        self.pc0 = float(pc0)
        D = max(len(getattr(self.param, 'asvarnames', []) or []), 1)
        self.pm0 = float(pm0) if pm0 is not None else 1.0 / D
        self.ref_divisions = int(ref_divisions)
        self.generate_plots = generate_plots

    def __init__(self, param, ctrl=None, idnum=None, **kwargs):
        super().__init__(param, **kwargs)
        self.idnum = idnum or 'AGDS'

        if ctrl is not None:
            pop_size, maxiter, pc0, pm0 = (tuple(ctrl) + (20, 30, 0.9, None))[:4]
            self.set_control_parameters(pop_size=int(pop_size), maxiter=int(maxiter),
                                        pc0=float(pc0), pm0=pm0)
        else:
            self.set_control_parameters()

        self.memory = []
        self.best_sol = None
        self.start = None

    # ------------------------------------------------------------------ #
    #  dec x mask helpers                                                 #
    # ------------------------------------------------------------------ #
    def _candidate_vars(self):
        return list(getattr(self.param, 'asvarnames', []) or [])

    @staticmethod
    def _active_vars(sol):
        """The mask: the set of candidate variables present in the solution."""
        return set(sol.get('asvars', []) or [])

    def _sort_mem(self, mem):
        if self.nb_crit > 1:
            return self.non_dominant_sorting(mem)
        return sorted(mem, key=lambda s: s.obj(0))

    # ------------------------------------------------------------------ #
    #  non-dominated ranking (r_i, maxr)  -- Eq. 5 / Eq. 8               #
    # ------------------------------------------------------------------ #
    def _compute_ranks(self, mem):
        n = len(mem)
        if n == 0:
            return [], 1
        if self.nb_crit > 1:
            try:
                fronts = self.get_fronts(mem)
                ranks = [1] * n
                for layer, idxs in enumerate(fronts.values(), start=1):
                    for idx in idxs:
                        ranks[idx] = layer
                return ranks, max(ranks)
            except Exception:
                pass
        order = sorted(range(n), key=lambda i: mem[i].obj(0))
        ranks = [1] * n
        layer, prev = 1, None
        for i in order:
            val = mem[i].obj(0)
            if prev is not None and val > prev:
                layer += 1
            ranks[i] = layer
            prev = val
        return ranks, max(ranks)

    # ------------------------------------------------------------------ #
    #  (B) dynamic decision-variable score  -- Eqs. 8-10                 #
    # ------------------------------------------------------------------ #
    def _dynamic_scores(self, mem, ranks, maxr):
        cand = self._candidate_vars()
        si_r = [maxr - r + 1 for r in ranks]                   # Eq. 8
        sumS = {v: 0.0 for v in cand}
        for i, sol in enumerate(mem):
            active = self._active_vars(sol)
            for v in cand:
                if v in active:
                    sumS[v] += si_r[i]                          # Eq. 9
        maxS = max(sumS.values()) if sumS else 0.0
        return {v: maxS - sumS[v] + 1 for v in cand}           # Eq. 10 (lower = better)

    def _tournament_var(self, scores, candidates):
        """Binary tournament on the dynamic score - smaller S_d wins."""
        if not candidates:
            return None
        a = self.random_choice(candidates)
        b = self.random_choice(candidates)
        return a if scores.get(a, 0.0) <= scores.get(b, 0.0) else b

    # ------------------------------------------------------------------ #
    #  (A) adaptive genetic operator + (B) mask operator -> offspring    #
    # ------------------------------------------------------------------ #
    def _make_offspring(self, parent, partner, ps_i, scores):
        pc_i = self.pc0 * ps_i                                 # Eq. 6
        pm_i = self.pm0 * ps_i                                 # Eq. 7
        child = copy.deepcopy(parent)

        # ---- dec crossover: adopt some of the partner's variable draws -----
        if self.param.generator.rand() < pc_i:
            shared = self._active_vars(child) & self._active_vars(partner)
            p_rand = partner.get('randvars', {}) or {}
            c_rand = child.get('randvars', {}) or {}
            for v in shared:
                if v in p_rand and self.param.generator.rand() < 0.5:
                    c_rand[v] = p_rand[v]
            child['randvars'] = {k: val for k, val in c_rand.items()
                                 if k in self._active_vars(child)}

        # ---- dec mutation: redraw one variable's distribution --------------
        if self.param.generator.rand() < max(pm_i, self.pm0):
            rands = list((child.get('randvars', {}) or {}).keys())
            if rands and getattr(self.param, 'distr', None):
                v = self.random_choice(rands)
                child['randvars'][v] = self.param.generator.choice(self.param.distr)

        # ---- (B) mask crossover: adopt a differing variable from partner ---
        cand = self._candidate_vars()
        differing = [v for v in cand
                     if (v in self._active_vars(child)) != (v in self._active_vars(partner))]
        if differing and self.param.generator.rand() < pc_i:
            v = self._tournament_var(scores, differing)
            if v is not None:
                if v in self._active_vars(partner):
                    child = self.add_asvar(v, child)
                else:
                    child = self.remove_asvar(v, child)

        # ---- (B) mask mutation: invert a tournament-chosen dimension -------
        if cand and self.param.generator.rand() < max(pm_i, self.pm0):
            v = self._tournament_var(scores, cand)
            if v is not None:
                if v in self._active_vars(child):
                    child = self.remove_asvar(v, child)
                else:
                    child = self.add_asvar(v, child)

        child = self.repair_solution(child)
        return child

    def _reproduce(self, mem):
        ranks, maxr = self._compute_ranks(mem)
        scores = self._dynamic_scores(mem, ranks, maxr)
        n = len(mem)
        children = []
        for i in range(n):
            ps_i = (maxr - ranks[i] + 1) / maxr                # Eq. 5
            j = int(self.param.generator.choice(n)) if n > 1 else 0
            while j == i and n > 1:
                j = int(self.param.generator.choice(n))
            child = self._make_offspring(mem[i], mem[j], ps_i, scores)
            child, converged = self.evaluate_solution(child)
            children.append(child if converged else mem[i])
        return children

    # ------------------------------------------------------------------ #
    #  (C) reference-point-based environmental selection (NSGA-III)       #
    # ------------------------------------------------------------------ #
    def _obj_vector(self, sol):
        return [float(sol.obj(k)) for k in range(self.nb_crit)]

    def _environmental_selection(self, combined, n_select):
        combined = get_unique(combined, 0)
        if len(combined) <= n_select:
            return combined
        if self.nb_crit <= 1:
            return sorted(combined, key=lambda s: s.obj(0))[:n_select]

        fronts = self.get_fronts(combined)
        selected, critical = [], None
        for idxs in fronts.values():
            if len(selected) + len(idxs) <= n_select:
                selected.extend(idxs)
            else:
                critical = idxs
                break
        if len(selected) >= n_select or critical is None:
            return [combined[i] for i in selected[:n_select]]

        F = np.array([self._obj_vector(combined[i]) for i in range(len(combined))],
                     dtype=float)
        M = F.shape[1]
        ideal = F.min(axis=0)
        Fn = F - ideal
        nadir = Fn.max(axis=0)
        nadir[nadir == 0] = 1.0
        Fn = Fn / nadir

        ref = reference_points(M, self.ref_divisions)
        sel_assoc = associate(Fn[selected], ref) if selected else np.array([], dtype=int)
        niche = np.zeros(len(ref), dtype=int)
        for a in sel_assoc:
            niche[a] += 1
        crit_assoc = associate(Fn[critical], ref)
        crit_dist = perp_distance(Fn[critical], ref, crit_assoc)

        chosen = list(selected)
        available = list(range(len(critical)))
        while len(chosen) < n_select and available:
            rmin = None
            for k in set(crit_assoc[c] for c in available):
                if rmin is None or niche[k] < niche[rmin]:
                    rmin = k
            members = [c for c in available if crit_assoc[c] == rmin]
            if niche[rmin] == 0:
                pick = min(members, key=lambda c: crit_dist[c])
            else:
                pick = int(self.random_choice(members))
            chosen.append(critical[pick])
            niche[rmin] += 1
            available.remove(pick)
        return [combined[i] for i in chosen[:n_select]]

    # ------------------------------------------------------------------ #
    #  initial population                                                 #
    # ------------------------------------------------------------------ #
    def _initialize(self, existing_sols=None):
        mem = []
        if existing_sols:
            for sol in existing_sols:
                sol, converged = self.evaluate_solution(sol)
                if converged:
                    mem.append(sol)
        attempts = 0
        while len(mem) < self.pop_size and attempts < 30000:
            attempts += 1
            sol = self.generate_solution()
            sol, converged = self.evaluate_solution(sol)
            if converged and abs(sol.obj(0)) < BOUND:
                mem.append(sol)
            mem = get_unique(mem, 0)
        if not mem:
            raise RuntimeError(
                "AGDS initial population is empty: none of the candidate "
                "specifications converged. Check Parameters (n_draws/maxiter), "
                "model setup, and run with verbose_convergence=True to print "
                "per-fit failures.")
        return self._sort_mem(mem)[:self.pop_size]

    # ------------------------------------------------------------------ #
    #  main entry point                                                   #
    # ------------------------------------------------------------------ #
    def run_search(self, existing_sols=None):
        import os, time as _time
        self.start = datetime.datetime.now()

        # Open progress CSV
        crit_names = [c[0] for c in self.param.criterions[:self.nb_crit]]
        try:
            run_id = self.idnum or 'agds'
            ts = _time.strftime("%Y%m%d_%H%M%S")
            pf = open(f"agds_{run_id}_{ts}_progress.csv", "w")
        except Exception:
            pf = open("agds_progress.csv", "w")
        print("iteration," + ",".join(crit_names), file=pf)
        pf.flush()

        memory = self._initialize(existing_sols)
        for sol in memory:
            sol.data['is_initial_sol'] = True

        for gen in range(self.maxiter):
            children = self._reproduce(memory)
            memory = self._environmental_selection(memory + children, self.pop_size)
            best = self._sort_mem(memory)[0]
            if self.best_sol is None or best.obj(0) < self.best_sol.obj(0):
                self.best_sol = best
            logger.info("[AGDS] gen {}: best obj0 = {:.6g}".format(gen, best.obj(0)))

            # Log per-generation best values
            try:
                _best = {}
                for sol in memory:
                    for ci, cn in enumerate(crit_names):
                        try:
                            v = float(sol.obj(ci))
                        except Exception:
                            v = float('inf')
                        if cn not in _best or abs(v) < abs(_best[cn]):
                            _best[cn] = v
                _vals = ','.join(str(_best.get(cn, '')) for cn in crit_names)
                print(f"{gen},{_vals}", file=pf)
                pf.flush()
            except Exception:
                pass

        pf.close()
        self.memory = self._sort_mem(memory)
        logger.info("[AGDS] search complete; best obj0 = {:.6g}"
                    .format(self.memory[0].obj(0)))

        # Generate convergence plot
        if self.generate_plots:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                pf_path = pf.name
                lines = open(pf_path).read().strip().split('\n')
                if len(lines) >= 2:
                    header = lines[0].split(',')
                    data = [l.split(',') for l in lines[1:] if l.strip()]
                    iters = [int(d[0]) for d in data]
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for ci, col in enumerate(header[1:], 1):
                        vals = []
                        for d in data:
                            try: vals.append(float(d[ci]))
                            except: vals.append(None)
                        vc = [(i, v) for i, v in zip(iters, vals) if v is not None]
                        if vc:
                            xs, ys = zip(*vc)
                            ax.plot(xs, ys, label=col.strip(), linewidth=1.5)
                    ax.set_xlabel('Generation')
                    ax.set_ylabel('Objective Value')
                    ax.set_title('SparseEA-AGDS Convergence')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    fig.savefig("convergence_agds.png", dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    logger.info("[AGDS] convergence plot saved: convergence_agds.png")
            except Exception as exc:
                logger.warning(f"Convergence plot failed: {exc}")

        return self.memory

    # ------------------------------------------------------------------ #
    #  interface parity with HarmonySearch / SA (used by call_meta)       #
    # ------------------------------------------------------------------ #
    def return_best(self):
        """Return the best solution found (mirrors HarmonySearch.return_best)."""
        if self.best_sol is not None:
            return self.best_sol
        if self.memory:
            return self._sort_mem(self.memory)[0]
        return None

    def close_files(self):
        """No log files are opened by AGDS; provided for call_meta parity."""
        return None

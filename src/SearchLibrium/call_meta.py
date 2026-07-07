# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# META HEURISTIC OPTIMISATION — SearchLibrium
#
# Provides a consistent interface for all search algorithms:
#   call_siman   — Simulated Annealing
#   call_harmony — Harmony Search
#   call_search  — unified entry point (algorithm='sa' | 'hs')
#
# All functions accept:
#   parameters  — a Parameters object (from search.py)
#   init_sol    — optional warm-start solution
#   ctrl        — algorithm-specific hyperparameter tuple (auto-estimated if omitted)
#   id_num      — run identifier (used for log files)
#
# Auto-hyperparameter estimation:
#   If ctrl is None the function calls estimate_ctrl(parameters, algorithm) to
#   derive appropriate defaults based on problem size (number of candidate
#   variables, alternatives, observations, model complexity).
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

try:
    from harmony import*
    from siman import*
    from banditsa import*
    from threshold import*
    from sapbil import SAPBIL, ProbabilityMatrix
    from hspbil import HSPBIL
except ImportError:
    from .harmony import*
    from .siman import*
    from .banditsa import*
    from .threshold import*
    from .sapbil import SAPBIL, ProbabilityMatrix
    from .hspbil import HSPBIL

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Problem-size helpers
# ─────────────────────────────────────────────────────────────────────────────

def _problem_size(parameters):
    """
    Return a dict of problem-size indicators used to calibrate hyperparameters.

    Keys
    ----
    n_vars     : total number of candidate variables
    n_alts     : number of alternatives
    n_obs      : number of observations (rows / alternatives)
    n_models   : number of model classes in the search
    allow_random: whether random parameters are enabled
    complexity  : composite score (higher = harder problem)
    """
    n_vars    = len(getattr(parameters, 'asvarnames', []) or []) + \
                len(getattr(parameters, 'isvarnames', []) or [])
    n_alts    = len(getattr(parameters, 'choice_set', []) or [])
    df        = getattr(parameters, 'df', None)
    n_obs     = len(df) if df is not None else 1000
    n_models  = len(getattr(parameters, 'models_avail', ['multinomial']))
    allow_rnd = bool(getattr(parameters, 'allow_random', False))
    allow_bc  = bool(getattr(parameters, 'allow_bcvars', False))

    # Composite complexity: more variables/models/options → harder
    complexity = n_vars * n_alts * n_models
    if allow_rnd:
        complexity *= 2
    if allow_bc:
        complexity = int(complexity * 1.5)

    return dict(n_vars=n_vars, n_alts=n_alts, n_obs=n_obs,
                n_models=n_models, allow_random=allow_rnd,
                complexity=complexity)


def estimate_ctrl(parameters, algorithm='sa'):
    """
    Estimate appropriate default hyperparameters for *algorithm* based on
    the problem size encoded in *parameters*.

    Returns
    -------
    tuple
        SA  : (tI, tF, max_temp_steps, max_iter)
        HS  : (max_mem, maxiter, max_harm, min_harm, max_pitch, min_pitch)
    """
    ps = _problem_size(parameters)
    c  = ps['complexity']

    if algorithm == 'sa':
        # Temperature ladder scales with complexity
        # Small problem  (<  50): few variables, one model type
        # Medium problem (50-300): several variables, mixed models
        # Large problem  (> 300): many variables, random params, multiple models

        if c < 50:
            tI, tF            = 500,   0.01
            max_temp_steps    = 100
            max_iter          = 20
        elif c < 200:
            tI, tF            = 1000,  0.001
            max_temp_steps    = 200
            max_iter          = 30
        elif c < 600:
            tI, tF            = 2000,  0.001
            max_temp_steps    = 250
            max_iter          = 40
        else:
            tI, tF            = 5000,  0.0001
            max_temp_steps    = 300
            max_iter          = 50

        ctrl = (tI, tF, max_temp_steps, max_iter)

    elif algorithm == 'hs':
        # Harmony memory size and improvisation iterations scale with complexity
        if c < 50:
            max_mem, maxiter  = 10,  100
        elif c < 200:
            max_mem, maxiter  = 15,  300
        elif c < 600:
            max_mem, maxiter  = 20,  500
        else:
            max_mem, maxiter  = 25,  800

        ctrl = (max_mem, maxiter, 0.9, 0.6, 0.85, 0.3)

    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Use 'sa' or 'hs'.")

    return ctrl


def _describe_ctrl(ctrl, algorithm):
    """Return a human-readable string for the ctrl tuple."""
    if algorithm == 'sa':
        names = ('tI', 'tF', 'max_temp_steps', 'max_iter')
        hints = (
            'initial temperature  — higher = more exploration',
            'final temperature    — lower  = more exploitation',
            'number of cooling steps',
            'evaluations per cooling step',
        )
    else:
        names = ('max_mem', 'maxiter', 'max_harm', 'min_harm', 'max_pitch', 'min_pitch')
        hints = (
            'harmony memory size',
            'improvisation iterations',
            'max harmony consideration rate',
            'min harmony consideration rate',
            'max pitch adjustment rate',
            'min pitch adjustment rate',
        )
    lines = ['  ctrl tuple: ' + str(ctrl)]
    for name, val, hint in zip(names, ctrl, hints):
        lines.append(f'    {name:<20s} = {val}  ({hint})')
    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard printer (shared by all algorithms)
# ─────────────────────────────────────────────────────────────────────────────

def _print_dashboard(solver, best_sol, algorithm='SA'):
    """Print a formatted summary dashboard after a search run completes."""
    W    = 60
    LINE = '═' * W

    def _pad(text, width=W):
        return f"║{text}{' ' * max(width - len(text), 0)}║"

    def row(label, value, flag=''):
        text = f"  {label:<22s}: {value}"
        if flag:
            text += f'   {flag}'
        return _pad(text)

    def section(text):
        return f"╠{LINE}╣\n" + _pad(f"  {text}")

    # Title
    print(f"\n╔{LINE}╗")
    title = f'SEARCHLIBRIUM — {algorithm.upper()} DASHBOARD'
    tpad  = (W - len(title)) // 2
    print(f"║{' ' * tpad}{title}{' ' * (W - tpad - len(title))}║")

    if best_sol is None:
        print(section('No solution found'))
        print(f"╚{LINE}╝\n")
        return

    # Specification
    print(section('Best specification'))
    model_n  = best_sol.get('model_n', 'unknown')
    asvars   = best_sol.get('asvars',   [])
    isvars   = best_sol.get('isvars',   [])
    randvars = best_sol.get('randvars', {})
    bcvars   = best_sol.get('bcvars',   [])
    corvars  = best_sol.get('corvars',  [])

    print(row('Model type',  str(model_n)))
    print(row('AS variables', ', '.join(asvars) if asvars else '—'))
    if isvars:
        print(row('IS variables', ', '.join(isvars)))
    if randvars:
        print(row('Random params', ', '.join(f"{k}~{v}" for k, v in randvars.items())))
    if bcvars:
        print(row('Box-Cox vars', ', '.join(bcvars)))
    if corvars:
        print(row('Correlated vars', ', '.join(corvars)))

    # Fit statistics
    print(section('Fit statistics'))
    criterions = getattr(solver.param, 'criterions', []) if solver else []
    crit_names = [c[0] for c in criterions]

    def fmt(v):
        if v is None:
            return '—'
        try:
            return f'{float(v):.4f}'
        except Exception:
            return str(v)

    marker = lambda n: '◄ criterion' if n in crit_names else ''
    print(row('Log-likelihood', fmt(best_sol.get('loglik'))))
    print(row('AIC',            fmt(best_sol.get('aic')),    marker('aic')))
    print(row('BIC',            fmt(best_sol.get('bic')),    marker('bic')))
    if best_sol.get('mae') is not None:
        print(row('MAE',        fmt(best_sol.get('mae')),    marker('mae')))

    # Search statistics
    print(section('Search statistics'))
    conv     = getattr(solver, 'converged',     '?') if solver else '?'
    not_conv = getattr(solver, 'not_converged', '?') if solver else '?'
    acc      = getattr(solver, 'accepted',      '?') if solver else '?'
    total    = ((conv     if isinstance(conv,     int) else 0) +
                (not_conv if isinstance(not_conv, int) else 0))
    print(row('Evaluations', str(total)))
    print(row('Converged',   str(conv)))
    print(row('Accepted',    str(acc)))

    # Pareto archive (multi-objective only)
    nb_crit = getattr(solver, 'nb_crit', 1) if solver else 1
    if nb_crit > 1:
        archive = getattr(solver, 'archive', getattr(solver, 'memory', [])) or []
        print(section(f'Pareto archive  ({len(archive)} non-dominated solutions)'))
        header = '  #   ' + '   '.join(f'{c[0].upper()[:7]:<8}' for c in criterions)
        print(_pad(header[:W]))
        for idx, sol in enumerate(archive[:15]):
            vals = '   '.join(f"{float(sol.obj(i)):>8.3f}" for i in range(nb_crit))
            print(_pad(f"  {idx+1:<3d} {vals}"[:W]))
        if len(archive) > 15:
            print(_pad(f"  ... and {len(archive) - 15} more"))

    print(f"╚{LINE}╝\n")


# ─────────────────────────────────────────────────────────────────────────────
# Simulated Annealing
# ─────────────────────────────────────────────────────────────────────────────

def call_siman(parameters, init_sol=None, ctrl=None, **kwargs):
    """
    Run Simulated Annealing search.

    Parameters
    ----------
    parameters : Parameters
        Problem definition (variables, data, criteria, models).
    init_sol : Solution, optional
        Warm-start solution.  None = generate automatically.
    ctrl : tuple, optional
        ``(tI, tF, max_temp_steps, max_iter)``.
        If omitted the values are estimated from the problem size.
    **kwargs
        ``id_num``  — run identifier (int, used in log file names).
        Any other kwargs are forwarded to the SA constructor.

    Returns
    -------
    Solution
        Best converged, all-significant solution found.
    """
    # Backwards-compat: accept ctrl inside kwargs
    if ctrl is None:
        ctrl = kwargs.pop('ctrl', None)

    id_num = kwargs.pop('id_num', None)

    if ctrl is None:
        ctrl = estimate_ctrl(parameters, algorithm='sa')
        print(f"[SA] Auto-estimated hyperparameters (problem complexity "
              f"= {_problem_size(parameters)['complexity']}):")
    else:
        print(f"[SA] Using provided hyperparameters:")

    print(_describe_ctrl(ctrl, 'sa'))
    print()

    solver = SA(parameters, init_sol, ctrl, id_num, **kwargs)
    solver.run()
    solver.close_files()
    best = solver.return_best()
    _print_dashboard(solver, best, algorithm='SA')
    return best


def call_sapbil(parameters, init_sol=None, ctrl=None, **kwargs):
    """
    Run SA+PBIL (Simulated Annealing coupled with Population-Based Incremental
    Learning) search.

    Parameters
    ----------
    parameters : Parameters
        Problem definition (variables, data, criteria, models).
    init_sol : Solution, optional
        Warm-start solution.  None = generate automatically.
    ctrl : tuple, optional
        ``(tI, tF, max_temp_steps, max_iter)``.
        If omitted the values are estimated from the problem size.
    **kwargs
        ``id_num`` — run identifier (int, used in log file names).
        Any other kwargs are forwarded to the SAPBIL constructor.

    Returns
    -------
    Solution
        Best converged, all-significant solution found.
    """
    if ctrl is None:
        ctrl = kwargs.pop("ctrl", None)

    id_num = kwargs.pop("id_num", None)

    if ctrl is None:
        ctrl = estimate_ctrl(parameters, algorithm="sa")
        print(
            f"[SA+PBIL] Auto-estimated hyperparameters (problem complexity "
            f"= {_problem_size(parameters)['complexity']}):"
        )
    else:
        print("[SA+PBIL] Using provided hyperparameters:")

    print(_describe_ctrl(ctrl, "sa"))
    print()

    solver = SAPBIL(parameters, init_sol, ctrl, id_num, **kwargs)
    solver.run()
    solver.close_files()
    best = solver.return_best()
    _print_dashboard(solver, best, algorithm="SA+PBIL")
    return best


def call_banditsa(parameters, init_sol=None, ctrl=None, **kwargs):
    """
    Run Bandit-guided Simulated Annealing (Thompson Sampling on perturbation arms).

    Parameters
    ----------
    parameters : Parameters
        Problem definition (variables, data, criteria, models).
    init_sol : Solution, optional
        Warm-start solution. None = generate automatically.
    ctrl : tuple, optional
        ``(tI, tF, max_temp_steps, max_iter)``.
        If omitted the values are estimated from the problem size.
    **kwargs
        ``id_num``  - run identifier (int, used in log file names).
        Any other kwargs are forwarded to the BanditSA constructor.

    Returns
    -------
    Solution
        Best converged, all-significant solution found.
    """
    if ctrl is None:
        ctrl = kwargs.pop('ctrl', None)

    id_num = kwargs.pop('id_num', None)

    if ctrl is None:
        ctrl = estimate_ctrl(parameters, algorithm='sa')
        print(f"[BanditSA] Auto-estimated hyperparameters (problem complexity "
              f"= {_problem_size(parameters)['complexity']}):")
    else:
        print("[BanditSA] Using provided hyperparameters:")

    print(_describe_ctrl(ctrl, 'sa'))
    print()

    solver = BanditSA(parameters, init_sol, ctrl, id_num, **kwargs)
    solver.run()
    solver.close_files()
    best = solver.return_best()
    _print_dashboard(solver, best, algorithm='BanditSA')
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Harmony Search
# ─────────────────────────────────────────────────────────────────────────────

def call_harmony(parameters, init_sol=None, ctrl=None, **kwargs):
    """
    Run Harmony Search.

    Parameters
    ----------
    parameters : Parameters
        Problem definition.
    init_sol : Solution, optional
        Warm-start solution (passed as existing memory).
    ctrl : tuple, optional
        ``(max_mem, maxiter, max_harm, min_harm, max_pitch, min_pitch)``.
        If omitted the values are estimated from the problem size.
    **kwargs
        ``id_num`` — run identifier.

    Returns
    -------
    Solution
        Best solution in the final harmony memory.
    """
    if ctrl is None:
        ctrl = kwargs.pop('ctrl', None)

    id_num = kwargs.pop('id_num', None)

    if ctrl is None:
        ctrl = estimate_ctrl(parameters, algorithm='hs')
        print(f"[HS] Auto-estimated hyperparameters (problem complexity "
              f"= {_problem_size(parameters)['complexity']}):")
    else:
        print(f"[HS] Using provided hyperparameters:")

    print(_describe_ctrl(ctrl, 'hs'))
    print()

    solver = HarmonySearch(parameters, ctrl=ctrl, idnum=id_num)
    existing = [init_sol] if init_sol is not None else None
    solver.run_search(existing_sols=existing)
    solver.close_files()
    best = solver.return_best()
    _print_dashboard(solver, best, algorithm='HS')
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Harmony Search + PBIL
# ─────────────────────────────────────────────────────────────────────────────

def call_harmony_pbil(parameters, init_sol=None, ctrl=None, **kwargs):
    """Run Harmony Search with PBIL-guided pitch adjustment.

    Parameters
    ----------
    parameters : Parameters
        Problem definition.
    init_sol : Solution, optional
        Warm-start solution (passed as existing memory).
    ctrl : tuple, optional
        ``(max_mem, maxiter, max_harm, min_harm, max_pitch, min_pitch)``.
        If omitted the values are estimated from the problem size.
    **kwargs
        ``id_num`` — run identifier.

    Returns
    -------
    Solution
        Best solution in the final harmony memory.
    """
    if ctrl is None:
        ctrl = kwargs.pop('ctrl', None)

    id_num = kwargs.pop('id_num', None)

    if ctrl is None:
        ctrl = estimate_ctrl(parameters, algorithm='hs')
        print(f"[HS+PBIL] Auto-estimated hyperparameters (problem complexity "
              f"= {_problem_size(parameters)['complexity']}):")
    else:
        print(f"[HS+PBIL] Using provided hyperparameters:")

    print(_describe_ctrl(ctrl, 'hs'))
    print()

    solver = HSPBIL(parameters, init_sol, ctrl, idnum=id_num, **kwargs)
    existing = [init_sol] if init_sol is not None else None
    solver.run_search(existing_sols=existing)
    solver.close_files()
    best = solver.return_best()
    _print_dashboard(solver, best, algorithm='HS+PBIL')
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def call_search(parameters, init_sol=None, algorithm='sa', ctrl=None, **kwargs):
    """
    Unified search entry point — choose algorithm at runtime.

    Parameters
    ----------
    parameters : Parameters
        Problem definition.
    init_sol : Solution, optional
        Warm-start solution.
    algorithm : {'sa', 'banditsa', 'hs'}
        ``'sa'``  — Simulated Annealing  (default)
        ``'banditsa'`` — Bandit-guided Simulated Annealing
        ``'hs'``  — Harmony Search
    ctrl : tuple, optional
        Algorithm-specific control tuple.  Auto-estimated if omitted.

        SA  : ``(tI, tF, max_temp_steps, max_iter)``
        HS  : ``(max_mem, maxiter, max_harm, min_harm, max_pitch, min_pitch)``
    **kwargs
        Forwarded to the underlying algorithm function.

    Returns
    -------
    Solution
        Best solution found.

    Examples
    --------
    >>> best = call_search(params)                        # SA, auto ctrl
    >>> best = call_search(params, algorithm='banditsa')  # BanditSA, auto ctrl
    >>> best = call_search(params, algorithm='hs')        # HS, auto ctrl
    >>> best = call_search(params, ctrl=(500,0.001,80,15))# SA, manual ctrl
    >>> best = call_search(params, algorithm='hs',
    ...                    ctrl=(20, 400, 0.9, 0.6, 0.85, 0.3))
    """
    algorithm = algorithm.lower().strip()
    if algorithm in ('sa', 'siman', 'simulated_annealing'):
        return call_siman(parameters, init_sol=init_sol, ctrl=ctrl, **kwargs)
    elif algorithm in ('sapbil', 'sa_pbil', 'sa+pbil', 'pbil'):
        return call_sapbil(parameters, init_sol=init_sol, ctrl=ctrl, **kwargs)
    elif algorithm in ('banditsa', 'bandit_sa', 'bandit-simulated-annealing', 'bsa'):
        return call_banditsa(parameters, init_sol=init_sol, ctrl=ctrl, **kwargs)
    elif algorithm in ('hs', 'harmony', 'harmony_search'):
        return call_harmony(parameters, init_sol=init_sol, ctrl=ctrl, **kwargs)
    elif algorithm in ('hspbil', 'harmony_pbil', 'hs_pbil', 'hs+pbil'):
        return call_harmony_pbil(parameters, init_sol=init_sol, ctrl=ctrl, **kwargs)
    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Choose 'sa', 'sapbil', 'banditsa', 'hs', or 'hspbil'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Parallel SA variants (unchanged interface, improved ctrl handling)
# ─────────────────────────────────────────────────────────────────────────────

def call_parsa(parameters, init_sol=None, nthrds=4, ctrl=None, **kwargs):
    """Run Parallel Simulated Annealing across *nthrds* independent solvers."""
    if ctrl is None:
        ctrl = kwargs.pop('ctrl', None)
    if ctrl is None:
        ctrl = estimate_ctrl(parameters, algorithm='sa')

    parsa = PARSA(parameters, init_sol, ctrl, nthrds=nthrds)
    parsa.run()


def call_parcopsa(parameters, init_sol=None, nthrds=8, ctrl=None):
    """Run Parallel Cooperative SA (solvers share best solution periodically)."""
    if ctrl is None:
        ctrl = estimate_ctrl(parameters, algorithm='sa')
    parcopsa = PARCOPSA(parameters, init_sol, ctrl, nthrds=nthrds)
    parcopsa.run()


def call_threshold(parameters, init_sol=None, ctrl=None):
    """Run Threshold Accepting search."""
    if ctrl is None:
        ctrl = (10, 20, 20)     # (threshold, max_steps, max_iter)
    solver = TA(parameters, init_sol, ctrl)
    solver.run()
    solver.close_files()


# ─────────────────────────────────────────────────────────────────────────────
# Covering-array helper for hyperparameter experiments
# ─────────────────────────────────────────────────────────────────────────────

def covering_arrays(index=0):
    """Return a covering-array row for SA hyperparameter experiments."""
    covering_array = [
        (500,  0.001, 10,  10),
        (500,  0.01,  20,  20),
        (500,  0.1,   30,  50),
        (1000, 0.001, 20,  50),
        (1000, 0.01,  30,  10),
        (1500, 0.001, 30,  20),
        (1500, 0.1,   10,  50),
        (1500, 0.01,  20,  10),
    ]
    if index < len(covering_array):
        return covering_array[index]
    return covering_array[0]

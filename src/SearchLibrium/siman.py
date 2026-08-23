"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
IMPLEMENTATION: SIMULATED ANNEALING
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


"""
BACKGROUND - SIMULATED ANNEALING:

Simulated annealing is a probabilistic optimization algorithm inspired by the annealing process in metallurgy.
In metallurgy, annealing is a heat treatment process where a material (typically metal) is heated to a 
certain temperature and then cooled slowly to remove defects and reduce hardness. Similarly, in simulated annealing, 
the algorithm tries to find the global optimum of a function by iteratively exploring the solution space while 
gradually reducing the probability of accepting worse solutions as it progresses.

Here's how simulated annealing typically works:

1. Initialization: Start with an initial solution to the optimization problem. This could be a randomly generated 
solution or some other method depending on the problem.

2. Temperature Schedule: Simulated annealing uses a temperature parameter that controls the probability of accepting 
worse solutions as the algorithm progresses. Initially, the temperature is set to a high value, allowing the 
algorithm to explore a wide range of solutions. As the algorithm progresses, the temperature is gradually 
decreased according to a predefined cooling schedule.

3. Neighbor Generation: At each iteration, a neighboring solution to the current solution is generated. The neighbor
 could be obtained by making a small change to the current solution, such as flipping a bit in a binary 
 representation or perturbing the current solution in some other way.

4. Acceptance Criterion: The algorithm evaluates the quality of the neighboring solution using an objective 
function (fitness function). If the neighboring solution is better than the current solution, 
it is always accepted. If the neighboring solution is worse, it may still be accepted with a certain 
probability determined by the Metropolis criterion.

Iteration: Repeat steps 3 and 4 for a certain number of iterations or until a stopping criterion is met 
(e.g., reaching a maximum number of iterations, convergence criteria).

Cooling Schedule: The temperature is reduced gradually according to a predefined cooling schedule. 
Common cooling schedules include linear, exponential, or logarithmic cooling.

Simulated annealing is a versatile optimization algorithm that can be applied to various optimization 
problems, including combinatorial optimization, continuous optimization, and machine learning tasks. 
It's particularly useful for problems where the objective function is non-convex or has multiple 
local optima, as it allows the algorithm to escape local optima and explore the solution space more thoroughly.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

''' ---------------------------------------------------------- '''
''' LIBRARIES                                                  '''
''' ---------------------------------------------------------- '''
#from search import*
#try:
#   from .search import*
#    from .latent_class_constrained import LatentClassCoefficients
try:
    from search import *
except ImportError:
    from .search import*

import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set env var SL_QUIET=1 to suppress the per-temperature-step console heartbeat
# ("step number ...", "SA[..] step ../..") while keeping new-best-solution
# prints and the final dashboard. CSV progress logs are unaffected.
_SL_QUIET = bool(os.environ.get("SL_QUIET"))
import threading
from typing import Callable, Tuple
from datetime import datetime
import random
import string
import re


''' ---------------------------------------------------------- '''

overall_best_solution = None  # PARSA: Reference to best solution
lock = threading.Lock()  # PARSA: Mutex - synchronization primitive


import copy, types

def final_safe_deepcopy(obj, _memo=None):
    """Deepcopy that skips modules, classmethods, functions, and abc internals."""
    if _memo is None:
        _memo = {}

    # --- primitives ---
    if isinstance(obj, (int, float, bool, str, bytes, type(None))):
        return obj

    # --- Known unpickleable types ---
    uncopyable_types = (
        types.ModuleType,
        types.FunctionType,
        types.BuiltinFunctionType,
        types.LambdaType,
        types.MethodType,
        classmethod,
        staticmethod,
    )

    # --- Skip any known unpickleable object ---
    if isinstance(obj, uncopyable_types):
        return None

    # --- Skip common ABC and private impl objects ---
    typename = type(obj).__name__
    module = getattr(type(obj), "__module__", "")
    if typename.startswith("_abc") or "abc." in module:
        return None
    if hasattr(obj, "_abc_impl") or hasattr(obj, "_abc_registry"):
        return None

    # --- NumPy arrays ---
    if isinstance(obj, np.ndarray):
        return np.copy(obj)

    # --- Containers ---
    if isinstance(obj, (list, tuple, set, frozenset)):
        copied = [final_safe_deepcopy(v, _memo) for v in obj]
        return type(obj)(copied)

    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, uncopyable_types) or isinstance(v, uncopyable_types):
                continue
            new_dict[final_safe_deepcopy(k, _memo)] = final_safe_deepcopy(v, _memo)
        return new_dict

    # --- Custom objects ---
    if hasattr(obj, "__dict__"):
        # guard circular refs
        if id(obj) in _memo:
            return _memo[id(obj)]
        new_obj = copy.copy(obj)
        _memo[id(obj)] = new_obj
        for key, val in list(vars(new_obj).items()):
            # skip private, callable, or ABC internals
            if (key.startswith("_abc") or isinstance(val, uncopyable_types)
                or (hasattr(val, "_abc_impl") or hasattr(val, "_abc_registry"))
                or type(val).__name__.startswith("_abc")):
                setattr(new_obj, key, None)
            else:
                setattr(new_obj, key, final_safe_deepcopy(val, _memo))
        return new_obj

    # --- Fallback safe copy ---

    return copy.deepcopy(obj, memo=_memo)
def ultra_safe_deepcopy(obj, _memo=None):
    """Deepcopy that skips modules, functions, methods, and classmethods."""
    if _memo is None:
        _memo = {}

    # --- primitives ---
    if isinstance(obj, (int, float, bool, str, bytes, type(None))):
        return obj

    # --- skip dangerous callable/class references ---
    uncopyable_types = (
        types.ModuleType,
        types.FunctionType,
        types.BuiltinFunctionType,
        types.LambdaType,
        types.MethodType,
        classmethod,
        staticmethod,
    )
    if isinstance(obj, uncopyable_types):
        return None

    # --- numpy arrays ---
    if isinstance(obj, np.ndarray):
        return np.copy(obj)

    # --- containers ---
    if isinstance(obj, (list, tuple, set, frozenset)):
        copied = [ultra_safe_deepcopy(v, _memo) for v in obj]
        return type(obj)(copied)

    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, uncopyable_types) or isinstance(v, uncopyable_types):
                continue
            new_dict[ultra_safe_deepcopy(k, _memo)] = ultra_safe_deepcopy(v, _memo)
        return new_dict

    # --- custom objects ---
    if hasattr(obj, "__dict__"):
        # guard circular
        if id(obj) in _memo:
            return _memo[id(obj)]
        new_obj = copy.copy(obj)
        _memo[id(obj)] = new_obj
        for key, val in list(vars(new_obj).items()):
            if isinstance(val, uncopyable_types):
                setattr(new_obj, key, None)
            else:
                setattr(new_obj, key, ultra_safe_deepcopy(val, _memo))
        return new_obj

    # --- fallback ---

    return copy.deepcopy(obj, memo=_memo)
def really_safe_deepcopy(obj, _memo=None):
    """Deep‑copy any object while skipping module and function references."""
    if _memo is None:
        _memo = {}

    # --- primitives ---
    if isinstance(obj, (int, float, bool, str, bytes, type(None))):
        return obj

    # --- skip modules & internal functions outright ---
    if isinstance(obj, types.ModuleType):
        return None
    if isinstance(obj, (types.FunctionType, types.BuiltinFunctionType, types.LambdaType)):
        return None

    # --- containers ---
    if isinstance(obj, (list, tuple, set, frozenset)):
        copied = [really_safe_deepcopy(v, _memo) for v in obj]
        return type(obj)(copied)

    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, types.ModuleType) or isinstance(v, types.ModuleType):
                continue
            new_dict[really_safe_deepcopy(k, _memo)] = really_safe_deepcopy(v, _memo)
        return new_dict

    # --- custom objects ---
    if hasattr(obj, "__dict__"):
        # guard against circular refs
        if id(obj) in _memo:
            return _memo[id(obj)]
        new_obj = copy.copy(obj)
        _memo[id(obj)] = new_obj
        for key, val in list(vars(new_obj).items()):
            # replace any module/function with None
            if isinstance(val, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType)):
                setattr(new_obj, key, None)
            else:
                setattr(new_obj, key, really_safe_deepcopy(val, _memo))
        return new_obj

    # --- everything else ---

    return copy.deepcopy(obj, memo=_memo)

def safe_deepcopy(obj):
    if isinstance(obj, dict):
        return {k: safe_deepcopy(v) for k, v in obj.items() if not isinstance(v, types.ModuleType)}
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(safe_deepcopy(v) for v in obj)
    if hasattr(obj, "__dict__"):
        # Temporarily strip out modules from object
        new_obj = copy.copy(obj)
        for key, val in list(new_obj.__dict__.items()):
            if isinstance(val, types.ModuleType):
                setattr(new_obj, key, None)
        return copy.deepcopy(new_obj)
    return copy.deepcopy(obj)

'''Function for Fancy Printing'''
def star(func):
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)
        print("*" * 15)
        print(result)
        print("*" * 15)
        return result

    return inner


def are_solutions_equivalent(sol1, sol2):
    """
    Check if two solutions are equivalent by comparing their attributes.

    Parameters:
    - sol1: First solution (of type csolution).
    - sol2: Second solution (of type csolution).

    Returns:
    - Boolean: True if all attributes are equivalent, False otherwise.
    """
    # Compare `asvars` (list or dictionary)
    if sol1.asvars != sol2.asvars:
        return False

    # Compare `bcvars` (list or dictionary)
    if sol1.bcvars != sol2.bcvars:
        return False

    # Compare `randvars` (list or dictionary)
    if sol1.randvars != sol2.randvars:
        return False

    # If all attributes are equivalent
    return True


def generate_random_run_name(prefix="run"):
    # Generate a random string of 6 characters
    print('No id applied to SA algorithm, creating random run ID')
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=4))

    # Get the current timestamp
    timestamp = datetime.now().strftime("%m%d_%H%M")

    # Combine the elements to form the run name
    run_name = f"{prefix}_{timestamp}_{random_suffix}"
    return run_name

''' ---------------------------------------------------------- '''
''' CLASS FOR SIMULATED ANNEALING                              '''
''' ---------------------------------------------------------- '''
class SA(Search):
# {
    """ Docstring """

    ''' ---------------------------------------------------------- '''
    ''' Function. Constructor                                      '''
    ''' ---------------------------------------------------------- '''

    #for testing
    verbose = True
    def __init__(self, param:Parameters, init_sol, ctrl, idnum=None, **kwargs):
    # {
        # Generate a FRESH run name per instance (a mutable default evaluated at
        # class-definition time would give every solver the same stale name).
        if idnum is None:
            idnum = generate_random_run_name()
        super().__init__(param, idnum, **kwargs)     # Call base class constructor

        tI, tF, max_temp_steps, max_iter = ctrl  # Extract form 'ctrl'




        self.start_time = time.time()
        # Set parameters:
        self.max_time = kwargs.get('max_time', float('inf'))          # Maximum Allowable Run Time (no limit by default)
        self.max_total_iter = kwargs.get('max_total_iter', 100000)     # Maximum total temperature steps
        self.tI = tI                # Starting temperature
        self.tF = tF                # Final temperature
        self.max_temp_steps = max_temp_steps    # Maximum number of temperature steps
        self.max_iter = max_iter    # Maximum number of iterations at each temperature step
        self.max_no_impr = 100        # Max number of steps permitted without improvements
        # True = let choose_starting_solution recompute tI from delta-E sampling
        # (defaults to False so an explicitly supplied ctrl tuple is respected;
        # callers wanting auto-calibration pass calibrate_tI=True explicitly).
        self.calibrate_tI = kwargs.get('calibrate_tI', False)
        self.terminate = False      # Termination flag
        self.rate = np.exp((1.0 / (self.max_temp_steps-1)) * np.log(self.tF/self.tI)) # Temperature reduction rate

        # Note: tF = tI * power(rate, max_temp_steps-1)
        # Note: Subtract one because the first step at t=tI must be included as a step

        self.no_impr = 0            # Current number of iterations without improvement
        self.step = 0               # Current temperature step
        self.t = tI                 # Current temperature
        self.current_sol = init_sol # Current solution
        self.best_sol = None        # Best solution
        self.archive = []           # Archive of solutions
        self.start = None           # Start time
        self.accepted, self.not_accepted = 0, 0 # Counters
        self.comm_int = 1           # Communication interval for PARSA
        self.idnum = idnum

        self.stlt_coeff_mem = None
        # Where to write output directories (default: 'sa_runs/' beside cwd)
        self._output_dir = kwargs.get('output_dir', 'sa_runs')
        # Outputting results and convergence information
        self.open_files()

        # Define a member function for the acceptance function
        Args = Tuple[np.ndarray, np.ndarray]
        AcceptanceFn = Callable[[Args], bool]
        self.accept_change: AcceptanceFn = self.accept_change_single \
            if self.nb_crit == 1 else self.accept_change_multi

        # Define a member function for the perturbation function
        PerturbFn = Callable[[], None]
        self.perturb_function: PerturbFn = self.perturb_single if self.nb_crit == 1 else self.perturb_multi
        #print(self.nb_crit, 'q')
        #print('n')

    # }
    @classmethod
    def v_print(cls, message):
        if cls.verbose:
            print(message)

    def get_run_time(self):
        '''Gets the current run_time in seconds'''
        end_time = time.time()
        return end_time - self.start_time

    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT FILE SYSTEM
    # ─────────────────────────────────────────────────────────────────────────
    #
    # Every run writes its logs into a dedicated sub-directory:
    #
    #   <output_dir>/
    #   └── sa_<idnum>_<YYYYMMDD_HHMMSS>/
    #       ├── results.txt      Full narrative: initial sol, final sol, objectives
    #       ├── progress.csv     One row per temperature step: step,temp,curr,best,accepted,elapsed_s
    #       ├── perturbations.csv  Per-perturbation: step,obj_values,accepted(T/F)
    #       ├── archive.txt      Pareto-archive or single best (multi-obj runs)
    #       └── best.txt         Machine-readable best specification (copy-paste ready)
    #
    # The output directory defaults to "sa_runs/" relative to the working
    # directory but can be overridden by passing `output_dir=` to the SA
    # constructor or to call_siman().
    #
    # Naming convention rationale
    # ────────────────────────────
    #   sa_<idnum>  → identifies the parallel solver instance (useful for PARSA)
    #   <timestamp> → makes each run unique even with the same idnum
    #
    # ─────────────────────────────────────────────────────────────────────────

    def open_files(self):
    # {
        import os
        ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"sa_{self.idnum}_{ts}"
        base_dir = getattr(self, '_output_dir', 'sa_runs')
        self._run_dir = os.path.join(base_dir, run_name)
        os.makedirs(self._run_dir, exist_ok=True)

        def _open(filename, mode='w'):
            return open(os.path.join(self._run_dir, filename), mode,
                        encoding='utf-8', buffering=1)   # line-buffered

        # ── narrative log: human-readable account of the entire run ──────────
        self.results_file  = _open('results.txt')

        # ── progress log: CSV, one row per temperature step ──────────────────
        # Columns: step, temperature, current_obj, best_obj, n_accepted, elapsed_s
        self.progress_file = _open('progress.csv')
        self.progress_file.write(
            'step,temperature,current_obj,best_obj,n_accepted,elapsed_s\n'
        )

        # ── perturbation log: CSV, one row per perturbation attempt ──────────
        # Columns: step,obj_values,accepted
        # "accepted" is True if the perturbation was accepted into the current solution
        self.debug_file = _open('perturbations.csv')
        self.debug_file.write('step,obj_values,accepted\n')

        # ── Pareto archive / single best (multi-obj) ─────────────────────────
        self.archive_file  = _open('archive.txt')

        # ── best specification: copy-paste ready Python dict ─────────────────
        self.best_file     = _open('best.txt')

        # Write run header to results.txt
        criterions = getattr(self.param, 'criterions', [])
        models     = getattr(self.param, 'models_avail', [])
        print(f"SearchLibrium — Simulated Annealing Run", file=self.results_file)
        print(f"Run ID       : {self.idnum}",              file=self.results_file)
        print(f"Started      : {ts}",                      file=self.results_file)
        print(f"Output dir   : {self._run_dir}",           file=self.results_file)
        print(f"Criterions   : {criterions}",              file=self.results_file)
        print(f"Models       : {models}",                  file=self.results_file)
        print(f"tI={self.tI}, tF={self.tF}, "
              f"steps={self.max_temp_steps}, iter/step={self.max_iter}",
              file=self.results_file)
        print('─' * 60, file=self.results_file)
    # }

    def close_files(self):
    # {
        for f in (self.results_file, self.progress_file,
                  self.debug_file, self.archive_file, self.best_file):
            try:
                f.flush()
                f.close()
            except Exception:
                pass
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    @star
    def curr_score(self, i):
        return self.current_sol.obj(i)

    def best_score(self, i):
        return self.best_sol.obj(i)

    @star
    def return_best(self):
        if self.nb_crit > 1 and self.archive:
            # Multi-objective: hand back the Pareto-archive member that
            # prioritises significance (criterion whose name contains 'sig'),
            # tie-breaking on the first declared criterion (usually bic/AIC).
            crit_names = [c[0].upper() for c in self.param.criterions[:self.nb_crit]]
            sig_prior = [(i, c[0]) for i, c in enumerate(self.param.criterions[:self.nb_crit])
                         if 'sig' in c[0].lower()]
            sig_idx = sig_prior[0][0] if sig_prior else 1
            prime_idx = 0
            best_sol = None
            best_key = None
            for s in self.archive:
                key = (float(s.obj(sig_idx)) if sig_idx < self.nb_crit else 0.0,
                       float(s.obj(prime_idx)))
                if best_key is None or key < best_key:
                    best_key, best_sol = key, s
            if best_sol is not None:
                return best_sol
        return self.best_sol




    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''


    def choose_starting_solution(self, N_trials=20):
        """
        Generate multiple starting solutions, calculate the starting temperature (tI)
        from converged solutions, and return a valid initial solution.

        Parameters:
        - N_trials: Number of independent solutions to generate for temperature estimation.

        Returns:
        - sol: A converged starting solution.
        """
        print(f"SA[{str(self.idnum)}] - Generating {N_trials} independent solutions for temperature calculation")

        temperatures = []  # Store tI values for each generated trial

        # Generate a valid initial starting solution
        base_sol = None
        for attempt in range(1, N_trials + 1):
            sol = self.generate_solution()
            sol = self.repair_solution_for_clarity(sol)
            # track_best=False: this is a disposable calibration trial, not
            # part of the accepted search trajectory, so it must not update
            # (or print) the run's global best solution.
            sol, converged = self.evaluate(sol, track_best=False)
            if converged:
                base_sol = sol
                break
            print(f"Starting solution trial {attempt}: did not converge, retrying.")

        if base_sol is None:
            print(f"No converged initial solution found after {N_trials} attempts; using last generated solution.")
            base_sol = sol

        base_obj = base_sol.obj(0)
        delta_Es = []
        base_sig = self.setup_signature(base_sol)

        def propose_solution(solution):
            if solution is None:
                return None

            choices = []
            l_a = len(self.param.asvarnames) if self.param.asvarnames is not None else 0
            l_b = len(self.param.isvarnames) if self.param.isvarnames is not None else 0

            if self.param.asvarnames is not None:
                choices.extend([self.perturb_asfeature] * l_a)
            if self.param.isvarnames is not None:
                choices.extend([self.perturb_isfeature] * l_b)
            _rand_isvars = getattr(self.param, "allow_random_isvars", False)
            if self.param.allow_random and (l_a > 0 or (_rand_isvars and l_b > 0)):
                choices.extend([self.perturb_randfeature] * max(1, l_a + (l_b if _rand_isvars else 0)))
            if solution['randvars'] and self.param.allow_random:
                choices.extend([self.perturb_distribution] * max(1, l_a))
            if self.param.avail_models is not None and len(self.param.avail_models) > 1:
                choices.append(self.perturb_model_t)
            if self.param.ps_bctrans is not None and self.param.allow_bcvars:
                choices.append(self.perturb_bcfeature)
            if self.param.ps_cor is not None and self.param.allow_corvars:
                choices.append(self.perturb_corfeature)

            if not choices:
                return None

            perturbations = np.random.randint(1, 6)
            max_attempts = 15
            for _ in range(max_attempts):
                candidate = self.copy_solution(solution)
                for _ in range(perturbations):
                    choice = np.random.choice(choices)
                    result = choice(candidate)
                    if result is not None:
                        candidate = result

                candidate = self.apply_constraints(candidate)
                candidate = self.repair_solution_for_clarity(candidate)
                if self.setup_signature(candidate) != base_sig:
                    return candidate
                perturbations = min(perturbations + 1, 8)
            return None

        for trial in range(1, N_trials + 1):
            proposed_sol = propose_solution(base_sol)
            if proposed_sol is None:
                print(f"Trial {trial}: Could not generate a new proposed solution, skipping temperature calculation.")
                continue

            # track_best=False: same reason as above — these proposals are
            # only used to estimate delta_E for the starting temperature and
            # are discarded immediately after, they never become current_sol.
            proposed_sol, converged = self.evaluate(proposed_sol, track_best=False)
            if not converged:
                print(f"Trial {trial}: Proposed solution did not converge, skipping temperature calculation.")
                continue

            delta_E = proposed_sol.obj(0) - base_obj
            if delta_E > 0:
                delta_Es.append(delta_E)

        if getattr(self, 'calibrate_tI', True):
            if delta_Es:
                avg_delta_E = np.mean(delta_Es)
                self.tI = -avg_delta_E / np.log(0.5)
                print(f"Calculated temperature tI = {self.tI}")
            else:
                print("No worse converged proposed solutions found during temperature sampling. Using default starting temperature.")
                self.tI = 1.0

        # Recompute the cooling rate to match the (possibly new) tI
        if self.tI > 0 and self.max_temp_steps > 1:
            self.rate = np.exp((1.0 / (self.max_temp_steps - 1)) * np.log(self.tF / self.tI))

        return base_sol
    # }
    def repair_solution_for_clarity(self, solution):
        '''
        This function repairs a solutions class Membership so similiar variables are not
        place in the class
        For example:
        Class 1: cannot Have Price and Price_2
        #TODO placeholder
        '''
        if solution.data['model_n'] == 'mixed_logit':
            # Try to give the mixed logit at least one random coefficient. If no
            # eligible candidate can be made random (e.g. every variable in the
            # spec is already random, or the pool is empty), DO NOT spin forever
            # printing debug text (this previously produced multi-GB logs and
            # exhausted the walltime) — cap the attempts and fall back to a plain
            # multinomial logit, which is always a valid specification.
            for _ in range(25):
                if len(solution.data['randvars']) > 0:
                    break
                self.perturb_add_randfeature(solution)
            if len(solution.data['randvars']) == 0:
                solution.data['model_n'] = 'multinomial'

        #make sure i is consistent with the asvars and isvars
        if solution.data['model_n'] == 'nested_logit':
            for _ in range(25):
                if len(solution.data['isvars']) + len(solution.data['asvars']) > 0:
                    break
                if random.random() > .5:
                    self.perturb_add_asfeature(solution)
                else:
                    self.perturb_add_isfeature(solution)


                
            

        return solution


    ''' ---------------------------------------------------------- '''
    ''' Function.  Evaluate how good a solution is                 '''
    ''' ---------------------------------------------------------- '''
    def evaluate(self, sol, track_best=True):
    # {
        sol, converged = self.evaluate_solution(sol, track_best=track_best)

        return sol, converged
    # }




    ''' ---------------------------------------------------------- '''
    ''' Function. Prepare to run the algorithm                     '''
    ''' ---------------------------------------------------------- '''
    def copy_solution(self, sol):
    # {
        logging.info('normal copy')
        # Fitted model objects contain module-level references that cannot be
        # pickled, so deepcopy would fail when mixed/random models are used.
        # The fitted model is read-only after estimation, so a shallow reference
        # in the copy is safe.
        _SKIP = ('model',)
        saved = {k: sol.data.pop(k, None) for k in _SKIP}
        try:
            copy_sol = copy.deepcopy(sol)
        finally:
            for k, v in saved.items():  # restore originals regardless of error
                sol.data[k] = v
        for k, v in saved.items():     # shallow-attach to copy
            copy_sol.data[k] = v
        return copy_sol
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Prepare to run the algorithm                     '''
    ''' ---------------------------------------------------------- '''
    def prepare_to_run(self):
    # {
        self.start = time.time()
        self.no_impr, self.step = 0, 0

        if self.current_sol == None:
            self.current_sol = self.choose_starting_solution()
            # choose_starting_solution may have updated self.tI via
            # temperature calibration, so set self.t afterwards.
        self.t = self.tI  # Set current temperature

        # Reset the timer so max_time only governs the SA search
        # iterations, not the calibration phase above.
        self.start_time = time.time()

        # ----------------------------------------------------------
        if self.current_sol is None:
        # {
            print("A feasible starting solution was not generated")
            quit()
        # }
        # _______________________________________________________

        # Add starting solution to the archive
        self.archive.append(self.current_sol)

        # Define best_sol = current_sol
        self.best_sol = self.copy_solution(self.current_sol)

        # Log initial solution and report progress
        print(f"SA[{self.idnum}]. Starting solution: ", self.current_sol.get_obj())
        # report_progress with file=None → console + CSV row to progress_file automatically
        self.report_progress(file=None)
        self.log_solution("Initial Solution", self.current_sol, file=self.results_file)
    # }


    def calculate_starting_temperature(self, init_sol, N_trials=25):
        """
        Dynamically calculate the starting temperature (tI) to achieve ~50% acceptance rate.

        Parameters:
        - init_sol: Initial solution to perturb.
        - N_trials: Number of trial perturbations to perform.

        Returns:
        - tI: Calculated starting temperature.
        """
        delta_Es = []  # Store ΔE values (energy differences)

        for _ in range(N_trials):
            # Generate a random perturbation
            perturbed_sol = self.copy_solution(init_sol)
            perturbed_sol = self.perturb_solution(perturbed_sol)

            # Calculate the objective function difference
            before = [init_sol.obj(i) for i in range(self.nb_crit)]
            after = [perturbed_sol.obj(i) for i in range(self.nb_crit)]

            # Calculate ΔE for the first criterion (can extend for multi-objective)
            delta_E = after[0] - before[0]
            if delta_E > 0:  # Only consider "worse" solutions
                delta_Es.append(delta_E)

        if len(delta_Es) == 0:
            # If no worse solutions were found, fallback to a default temperature
            print("No worse solutions found during trials. Using default starting temperature.")
            return 1.0

        # Calculate the temperature for ~50% acceptance
        avg_delta_E = np.mean(delta_Es)
        tI = -avg_delta_E / np.log(0.5)

        print(f"Calculated starting temperature: tI = {tI}")
        return tI

    ''' ---------------------------------------------------------- '''
    ''' Function. Finish up                                        '''
    ''' ---------------------------------------------------------- '''
    def finalise(self):
    # {
        self.report_exploration_summary()
        print(f"Solver[{str(self.idnum)}]. Finalising")
        if self.nb_crit == 1:
            self.log_solution("Final Solution", self.best_sol, file=self.results_file)
            self.log_decision(self.best_sol, file=self.best_file)
        else:
            self.log_archive("Non Dominated Solutions", file=self.archive_file)

        print(f"#Converged={self.converged}; #Not Converged={self.not_converged}", file=self.results_file)
        print(f"#Accepted={self.accepted}; #Not Accepted={self.not_accepted}", file=self.results_file)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.  Acceptance functions                            '''
    ''' ---------------------------------------------------------- '''
    def accept_change_metrop(self, before, after):
    # {
        """ Evaluate Metropolis function for each objective """
        crits = self.param.criterions

        after = np.array(after, dtype=np.float64)
        before = np.array(before, dtype=np.float64)

        # If the current (before) objectives are NaN or infinite the current
        # solution is considered infinitely bad; always accept a finite
        # replacement so the search is not permanently stuck.
        if not np.all(np.isfinite(before)):
            return bool(np.all(np.isfinite(after)))

        # Note: crit[1] is the sign and equivalent to crits[i][1]
        rn = np.random.rand()

        try:


            accept_i = [np.log(rn) < (crit[1] * (after[i] - before[i]) / self.t)
                    for i, crit in enumerate(crits)]
            #print(f" hjhj {np.log(rn)} vs {[(crit[1] * (after[i] - before[i]) / self.t) for i, crit in enumerate(crits)]} jhj")
            if not accept_i:

                if after - before< 0:
                    raise ValueError('this should have accepts')
                    print('concepttal error')
        except Exception as e:
            print('todo why')
            accept_i = []
            for i, crit in enumerate(crits):
                # Convert crit[1] to NumPy array if it's a list
                crit1 = np.array(crit[1]) if isinstance(crit[1], list) else crit[1]

                # Ensure after[i] and before[i] are NumPy arrays
                after_i = np.array(after[i]) if isinstance(after[i], list) else after[i]
                before_i = np.array(before[i]) if isinstance(before[i], list) else before[i]

                # Perform the comparison
                comparison = np.log(rn) < (crit1 * (after_i - before_i) / self.t)
                accept_i.append(comparison)
        return all(accept_i)
    # }

    def accept_change_relative(self, before, after):
    # {
        """ delta > 0 => improvement, delta < 0 => non improvement """

        crits = self.param.criterions
        delta_i = [(crits[i][1] * (after[i] - before[i])) / before[i] for i in range(len(crits))]
        if all(delta > 0 for delta in delta_i): # If all_positive
            return True
        else:
        # {
            if all(delta < 0 for delta in delta_i): # If all negative
                return False
            else:
                ratio = abs(delta_i[0] / delta_i[1])
                return 0.8 <= ratio <= 1.2
        # }
    # }

    # Note: This acceptance strategy does not use the temperature!
    def accept_change_dom(self, before, after):
    # {
        """ Use dominance conditions to accept/reject """
        crits = self.param.criterions

        if dominates(after, before, crits):
            return True # New solution is strictly better so accept it
        if not dominates(before, after, crits):
            return True # Accept as no dominance relationship exists, i.e. new solution is not worse

        # Solution is dominated - Accept 10% of the time
        return (np.random.rand() < 0.10)
    # }

    def accept_change_single(self, before, after):
    # {
        return self.accept_change_metrop(before, after)
    # }

    def accept_change_multi(self, before, after):
    # {
        #return self.accept_change_metrop(before, after)
        # return self.accept_change_relative(before, after)
        return self.accept_change_dom(before, after)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Restore the best solution                        '''
    ''' ---------------------------------------------------------- '''
    def restore_best(self):
    # {
        self.current_sol = self.copy_solution(self.best_sol)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Solution improvement process                     '''
    ''' ---------------------------------------------------------- '''
    def improve(self, sol):
    # {
        if np.random.rand() < 0.25:
            sol = self.local_search_distribution(sol, 0)
        else:
        # {
            choices = []
            choices.append(self.local_search_asfeature)
            choices.append(self.local_search_isfeature)
            choices.append(self.local_search_randfeature)
            choice = np.random.choice(choices) # Make a choice
            add = np.random.randint(2)  # Choose to add or remove feature
            sol = choice(sol, 0, add)
        # }
        return sol
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Perturb the solution                             '''
    ''' ---------------------------------------------------------- '''
    def perturb_single(self):
    # {
        self.perturb_solution(self.current_sol) # Perturb current solution
    # }

    def perturb_multi(self):
    #{
        num_par = len(self.archive)         # Number of Pareto-optimal solutions
        chosen = np.random.choice(num_par)  # Choose one solution
        chosen_sol = self.archive[chosen]   # Reference to chosen solution
        self.perturb_solution(chosen_sol)   # Perturb chosen solution
    # }

    def perturb_solution(self, sol):
    # {
        # Snapshot signature BEFORE any perturbation
        b = self.setup_signature(sol)
        curr_score = [sol.obj(i) for i in range(self.nb_crit)]
        new_sol = self.copy_solution(sol)

        choices = []
        max_attempts = 15
        # Use 1-5 perturbations; larger budgets risk cancellation
        perturbations = np.random.randint(1, 6)

        # ~~~~~~~~~~~~


        # Calculate lengths of asvarnames and varnames
        l_a = len(self.param.asvarnames) if self.param.asvarnames is not None else 0
        l_b = len(self.param.isvarnames) if self.param.varnames is not None else 0


        total_length = l_a + l_b if l_a + l_b > 0 else 1  # Avoid division by zero

        # Normalize lengths to determine weights







        #The idea is we only want to play with the options of model types. Ie regret, ordered, multinomial.

        # how to weight the choice based on whats available ie because isvars is so small i want to watither it less that
        if self.param.asvarnames is not None:
            for c in range(0, l_a):
                choices.append(self.perturb_asfeature)



        if self.param.isvarnames is not None:
            for c in range(0, l_b):
                choices.append(self.perturb_isfeature)


        if self.param.asvarnames is not None:
            #Not latent so can add
            #if sol['member_params_spec'] is None:
            if self.param.allow_random:
                for c in range(0, l_a):
                    choices.append(self.perturb_randfeature)


        if sol['randvars'] is not None and self.param.allow_random:
            for c in range(0, l_a):
             choices.append(self.perturb_distribution)

        if self.param.avail_models is not None and len(self.param.avail_models)>1:

            choices.append(self.perturb_model_t)
            #raise('does this work')


        if self.param.ps_bctrans is not None and self.param.allow_bcvars:
            choices.append(self.perturb_bcfeature)

        if self.param.ps_cor is not None  and self.param.allow_corvars:
            choices.append(self.perturb_corfeature)


        

        
       
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Call the chosen perturbation strategy

        #print('perturbation choice:', choice.__name__)


        attempts = 0
        a = b  # initialise to "same" so the loop enters
        while attempts < max_attempts:
            attempts += 1

            # Always restart from a fresh copy of the current solution so that
            # successive attempts cannot accumulate partial/cancelling changes.
            new_sol = self.copy_solution(sol)

            for _ in range(perturbations):
                choice = np.random.choice(choices)
                result = choice(new_sol)
                if result is not None:
                    new_sol = result

            if new_sol is None:
                return sol

            new_sol = self.apply_constraints(new_sol)
            new_sol = self.repair_solution_for_clarity(new_sol)

            # Recompute signature after ALL perturbations in this attempt
            a = self.setup_signature(new_sol)
            if a != b:
                # A genuinely different specification was produced
                logging.info('perturbation produced different spec on attempt %d', attempts)
                break

            # Increase the number of perturbations slightly on repeated failure
            # so we escape quickly from regions with few valid neighbours
            perturbations = min(perturbations + 1, 8)

        if a == b:
            # Could not produce a different specification – keep current
            return sol

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Evaluate the new specification
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        try:
            new_sol, converged = self.evaluate(new_sol)
        except Exception as exc:
            logging.warning("SA: evaluation failed — %s", exc)
            self.not_converged += 1
            return self.current_sol

        # IMPORTANT: only accept solutions that have converged
        if not converged:
            self.not_converged += 1
            return self.current_sol

        new_score = [new_sol.obj(i) for i in range(self.nb_crit)]
        args = (curr_score, new_score)
        if self.accept_change(*args):
        # {
            accd = True
            self.no_impr = 0
            self.accepted += 1
            self.current_sol = new_sol
            self.update_best(new_sol)
        # }
        else:
            accd = False
            self.not_accepted += 1
        # }
        self.log_kpi(new_sol, self.debug_file, accd)
        return self.current_sol
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def update_best(self, sol):
    # {


        if self.best_sol is None:
            # Initialize best_sol if it is None
            self.best_sol = self.copy_solution(sol)
            print("Initialized best_sol with the first solution")
            return

        if self.nb_crit == 1:
        # {
            if is_better(sol.obj(0), self.best_sol.obj(0), self.param.sign_crit(0)):
                self.best_sol = self.copy_solution(sol)
                self.no_impr = 0
                print('new best')
        # }
        else:
            self.archive = self.update_archive(sol)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Store non-dominated solutions only               '''
    ''' ---------------------------------------------------------- '''
    def update_archive(self, add_sol):
    # {
        _dominated = []
        for sol in self.archive:
        # {
            if dominates(sol.get_obj(), add_sol.get_obj(), self.param.criterions):
            # {
                # The new solution is dominated by an archive solution
                return self.archive # Return the archive - no need to continue
            # }
            elif dominates(add_sol.get_obj(), sol.get_obj(), self.param.criterions):
            # {

                if not any(np.array_equal(sol, d) for d in _dominated):
                    _dominated.append(sol)
                #if sol not in _dominated: _dominated.append(sol)
            # }
        # }

        # Remove all solutions 'add_sol' dominates and add 'add_sol'
        self.archive = [
                           s for s in self.archive
                           if not any(np.array_equal(s, d) for d in _dominated)
                       ] + [add_sol]

        return self.archive
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' -----------------------------------------------------------'''
    def reset_current_solution(self, size=None):
    # {
        if self.nb_crit == 1:
            self.handle_non_improvement()
        else:
            self.handle_static_archive(size)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' -----------------------------------------------------------'''
    def handle_non_improvement(self):
    # {
        if is_worse(self.current_sol.obj(0), self.best_sol.obj(0), self.param.sign_crit(0)):
            self.no_impr += 1  # Increment non improvement counter

        if self.no_impr > self.max_no_impr:  # Key step enabling performance
        # {
            print("NO IMPROVEMENT FOR A WHILE. RESTORE BEST SOLUTION.")
            self.restore_best()  # Reinstate the best solution
            self.no_impr = 0  # Reset non improvement counter
        # }
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' -----------------------------------------------------------'''
    def handle_static_archive(self, size):
    # {
        if len(self.archive) == size:
            self.no_impr += 1 # Increment non improvement counter

        if self.no_impr > self.max_no_impr:  # Key step enabling performance
        # {
            print("ARCHIVE STATIC. RESTORE A NON DOMINATED SOLUTION.")
            choice = np.random.randint(len(self.archive))
            self.current_sol = self.copy_solution(self.archive[choice])  # Deep copy to avoid aliasing archive entry
            self.no_impr = 0
        # }
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def synchronize(self):
    # {
        global overall_best_solution
        with lock:
        # {
            if overall_best_solution is None or \
                is_better(self.best_sol.obj(0), overall_best_solution.obj(0), self.param.sign_crit(0)):
                overall_best_solution = self.copy_solution(self.best_sol)  # Update overall best solution (deep copy to prevent overwriting)
            elif overall_best_solution is not None and \
                is_worse(self.best_sol.obj(0), overall_best_solution.obj(0), self.param.sign_crit(0)):
                self.update_best(overall_best_solution)  # Revise best solution of current SA solver
        # }
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Inner loop of Simulated Annealing algorithm      '''
    ''' ---------------------------------------------------------- '''
    def evaluate_state_changes(self):
    # {
        self.step += 1  # Increment the step variable
        count, size = 0, len(self.archive)
        while (True):
        # {
            self.perturb_function()
            count = count + 1
            if count % 10000 == 0:
                print(f'Iteration at {count}')
            if count >= self.max_iter:
                break
        # }
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # TURNED OFF. UNSATISFACTORY PERFORMANCE SO FAR!
        if not _SL_QUIET:
            print(f'step number {self.step}')
        #if (self.step) % 2:
        #    self.best_sol = self.improve(self.best_sol)  # Apply local improvement
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.report_progress(self.results_file)  # text narrative → results.txt
        # CSV row → progress.csv always; console heartbeat only when not quiet.
        self.report_progress(to_console=not _SL_QUIET)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t = self.rate * self.t  # Reduce the temperature accordingly

        # Pass the pre-loop archive size so handle_static_archive can detect a
        # static archive (previously called with no args, so the check never fired).
        self.reset_current_solution(size)
    # }

    def frozen(self):
        #if any true return true,
        #else return false
        return (self.get_run_time() > self.max_time or
        self.step > self.max_total_iter or
        self.t < self.tF)
        #return (self.t < self.tF)

    def iterate(self, synch=False):
    # {
        if not self.frozen(): # i.e. t > tF
            self.evaluate_state_changes()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Optional - Synchronize with other parallel SA
            #if synch and (self.step % self.comm_int == 0):
            #    self.synchronize()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.terminate = False
        else:
            if self.nb_crit == 1:
                self.restore_best() # Reinstate the best solution
            print(f"Solver[{str(self.idnum)}]. Search complete")
            self.terminate = True
        # }
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def report_progress(self, file=None, to_console=True):
    # {
        now     = time.time()
        elapsed = round(now - self.start, 1)
        temp    = f"{self.t:.4g}"
        step    = f"{self.step}/{self.max_temp_steps}"
        curr    = self.current_sol.concatenate_obj()

        if self.nb_crit == 1:
            best_obj = self.best_sol.concatenate_obj()
            # Human-readable console / text-file line
            text = (f"SA[{self.idnum}] step {step:>9s} | T={temp:>8s} | "
                    f"curr={curr} | best={best_obj} | "
                    f"acc={self.accepted} | t={elapsed}s")
            # CSV row → progress.csv
            # Columns: step, temperature, current_obj, best_obj, n_accepted, elapsed_s
            csv_row = (f"{self.step},{self.t:.6g},"
                       f"{self.current_sol.obj(0):.6g},"
                       f"{self.best_sol.obj(0):.6g},"
                       f"{self.accepted},{elapsed}")
        else:
            arch    = len(self.archive)
            text    = (f"SA[{self.idnum}] step {step:>9s} | T={temp:>8s} | "
                       f"curr={curr} | archive={arch} | "
                       f"acc={self.accepted} | t={elapsed}s")
            # For multi-objective runs the CSV keeps the same column layout as
            # the header (step,temperature,current_obj,best_obj,n_accepted,elapsed_s)
            # using the primary criterion for current/best so the values stay
            # strictly numeric (no trailing "archive_size=..." tokens).
            best0 = None
            if self.archive:
                best0 = min(float(s.obj(0)) for s in self.archive)
            if best0 is None:
                best0 = float(self.current_sol.obj(0))
            csv_row = (f"{self.step},{self.t:.6g},"
                       f"{self.current_sol.obj(0):.6g},"
                       f"{best0:.6g},"
                       f"{self.accepted},{elapsed}")

        # Write human-readable line (console or results_file). The console
        # heartbeat can be suppressed (to_console=False) while the results_file
        # narrative and the CSV progress log are always kept.
        if file is not None or to_console:
            print(text, file=file)

        # Always write CSV to progress_file (separate from the text log)
        if file is None and hasattr(self, 'progress_file'):
            print(csv_row, file=self.progress_file)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. obj[1], obj[2], t/f                                 '''
    ''' ---------------------------------------------------------- '''
    def log_kpi(self, sol, file=None, accept=True):
    # {
        # perturbations.csv row: step, obj_values (pipe-separated), accepted
        # Example: 3,523.17,true
        obj_str  = '|'.join(str(round(sol.obj(i), 6)) for i in range(self.nb_crit))
        accepted = 'true' if accept else 'false'
        print(f"{self.step},{obj_str},{accepted}", file=file)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def log_solution(self, descr, sol, file=None):
    # {
        # ── Section header ────────────────────────────────────────────────────
        sep = '═' * 60
        print(sep, file=file)
        print(f"  {descr}", file=file)
        print(sep, file=file)

        # ── Objective values ─────────────────────────────────────────────────
        print("Objectives:", file=file)
        for i in range(self.nb_crit):
            opt  = "Maximise" if self.param.sign_crit(i) == 1 else "Minimise"
            text = f"  [{i}] ({opt}) {self.param.crit(i)} = {round(sol.obj(i), 4)}"
            print(text, file=file)

        # ── Model fit statistics (LL, BIC, etc.) ─────────────────────────────
        print("", file=file)
        print("Model Statistics:", file=file)
        try:
            report_model_statistics(sol['model'], file)
        except Exception as e:
            print(f"  (statistics unavailable: {e})", file=file)

        print("", file=file)

        # ── Specification (copy-paste ready) ─────────────────────────────────
        print("Specification:", file=file)
        self.log_decision(sol, file=file)
        print("", file=file)
    # }

    def log_decision(self, sol, file=None):
    # {
        def _clean(v):
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
            if isinstance(v, dict):
                return {str(k): str(val) for k, val in v.items()}
            return v

        print("asvars   = ", _clean(sol['asvars']),   file=file)
        print("isvars   = ", _clean(sol['isvars']),   file=file)
        print("randvars = ", _clean(sol['randvars']), file=file)
        print("bcvars   = ", _clean(sol['bcvars']),   file=file)
        print("corvars  = ", _clean(sol['corvars']),  file=file)
        print("bctrans  = ", sol['bctrans'],           file=file)
        print("asc_ind  = ", sol['asc_ind'],           file=file)
        print("model    = ", sol['model_n'],            file=file)
        model = sol.get('model')
        if model is not None and hasattr(model, 'descr') and model.descr:
            print("model descr =", model.descr, file=file)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def log_archive(self, descr, file=None):
    # {
        for i, sol in enumerate(self.archive):
        # {
            descr = "Non-Dominated #" + str(i)
            self.log_solution(descr, sol, file=file)
        # }
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Outer loop of Simulated Annealing Algorithm      '''
    ''' ---------------------------------------------------------- '''
    def run_search(self):
    # {
        self.prepare_to_run()
        while (True):
        #
            self.iterate()
            if self.terminate: break
        # }
        self.finalise()
    # }


    ''' ---------------------------------------------------------- '''
    ''' Function. Sequential latent class approach                 '''
    ''' ---------------------------------------------------------- '''
    def search_latent_update_single(self, overall_best):
    # {
        sign = self.param.sign_crit(0)     # Shortcut to optimisation sign

        if overall_best is None or is_better(self.best_sol.obj(0), overall_best.obj(0), sign):
            overall_best = self.copy_solution(self.best_sol)
            #update the coefficients
            

        # Current best solution is worse, so terminate:
        terminate = is_worse(self.best_sol.obj(0), overall_best.obj(0), sign)
        return overall_best, terminate
    # }

    # Terminate if no solution in the archive has a 'class_num' = q
    def search_latent_update_multi(self, q):
    # {
        pareto_class_nums = [sol['class_num'] for sol in self.archive]
        terminate = (max(pareto_class_nums) != q)
        return terminate
    # }

    """def run_search_latent(self, max_classes=5):
    # {
        overall_best_sol = None
        for q in range(1, self.max_classes):
        # {
            print('RunSearchLatent. #classes=', q)
            self.param.latent_class = False if q==1 else True
            self.param.num_classes = q
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # FORCE GENERATION OF A NEW STARTING SOLUTION
            # WITH LATENT CLASS COMPONENTS
            if q == 2:
                del self.current_sol # Delete current solution
                self.current_sol = None
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.run_search()
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.current_sol['class_num'] = q
            self.best_sol['class_num'] = q
            if self.nb_crit == 1:
                overall_best_sol, terminate = self.search_latent_update_single(overall_best_sol)
            else:
                terminate = self.search_latent_update_multi(q)
            if terminate: break

            print(f"q={q},terminate={terminate}")
        # }
        print(f"Best solution has {overall_best_sol['class_num']} latent classes")
        self.finalise()
    # }"""

    ####################################################################################
    # THE SEQUENTIAL APPROACH DESCRIBED ABOVE IS NOT BEST PRACTICE.
    # IT MAKES MORE SENSE TO RUN EACH NUMBER OF LATENT CLASSES SEPARATELY, IN PARALLEL
    # AND TO IDENTIFY THE BEST APPROACH AT THE END
    ####################################################################################
    # New & Untested!!!!
    def run_search_latent(self, max_classes=5):
    # {
        init_sol = None
        ctrl = (self.tI, self.tF, self.max_temp_steps, self.max_iter)

        # Define and setup independent solvers.
        # SA.__init__ signature is (param, init_sol, ctrl, ...): the previous
        # call passed ctrl/init_sol in reverse order, which crashed when
        # unpacking ctrl. Class counts start at 1 (0 classes is meaningless).
        self.solvers = []
        for q in range(1, max_classes + 1):
        # {
            self.param.latent_class = q > 1
            self.param.num_classes = q
            solver_q = SA(self.param, init_sol, ctrl, idnum=q)
            self.solvers.append(solver_q)

        # }

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(solver.run) for solver in self.solvers]

        for future in as_completed(futures):
            result = future.result()  # This will wait until each task completes

        for q, solver in enumerate(self.solvers):
        # {
            solver.current_sol['class_num'] = q
            solver.best_sol['class_num'] = q
            solver.finalise()
        # }

    # }

    ####################################################################################


    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def revise_tI(self, new_tI):
    # {
        self.tI = new_tI
        if self.max_temp_steps > 1:
            self.rate = np.exp((1.0 / (self.max_temp_steps - 1)) * np.log(self.tF / self.tI))
        else:
            self.rate = 1.0
    # }
 # }


''' ----------------------------------------------------------- '''
''' PARALLEL SIMULATED ANNEALING                                '''
''' ----------------------------------------------------------- '''
class PARSA():
# {
    """ Docstring """

    ''' ---------------------------------------------------------- '''
    ''' Function. Constructor                                      '''
    ''' ---------------------------------------------------------- '''
    def __init__(self, param: Parameters, init_sol, ctrl, nthrds=1):
    # {
        self.nthrds = nthrds

        # Define and setup independant solvers
        self.solvers = [SA(param, init_sol, ctrl, idnum=i) for i in range(nthrds)]

        self.choose_custom_tI()  # Optional

        self.comm_int = 1
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def wait(self, futures):
    # {
        for future in as_completed(futures):
            result = future.result()  # This will wait until each task completes
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def run(self):
    # {
        for i in range(self.nthrds):
            self.solvers[i].comm_int = self.comm_int

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.solvers[i].run) for i in range(self.nthrds)]

        self.wait(futures)
        print("PARSA FINISHED!")
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def choose_custom_tI(self, options=None):
    # {
        if options == None or len(options) == 0:
            for _, solver in enumerate(self.solvers):
                solver.revise_tI(np.random.randint(1, 10000))
        else:
            for _, solver in enumerate(self.solvers):
                solver.revise_tI(np.random.choice(options))
    # }
# }


''' ----------------------------------------------------------- '''
''' PARALLEL COOPERATIVE SIMULATED ANNEALING                    '''
''' ----------------------------------------------------------- '''
class PARCOPSA(PARSA):
# {
    """ Docstring """

    ''' ---------------------------------------------------------- '''
    ''' Function. Constructor                                      '''
    ''' ---------------------------------------------------------- '''
    def __init__(self, param: Parameters, init_sol, ctrl, nthrds=1):
    # {
        super().__init__(param, init_sol, ctrl, nthrds)  # Call base class constructor
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def get_best(self):
    # {
        best_at, sign = 0, self.solvers[0].param.sign_crit(0)
        for i in range(1, self.nthrds):
        # {
            obj_i = self.solvers[i].best_sol.get_obj()
            obj_best = self.solvers[best_at].best_sol.get_obj()
            best_at = i if is_better(obj_i, obj_best, sign) else best_at
        # }
        return best_at
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def communicate(self, best_at):
    # {
        best_sol = self.solvers[best_at].best_sol
        for idx, solver in enumerate(self.solvers):
            if idx != best_at:
                solver.update_best(best_sol)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def run(self):
    # {
        with ThreadPoolExecutor(max_workers=self.nthrds) as executor:
        # {
            futures = [executor.submit(self.solvers[i].prepare_to_run) for i in range(self.nthrds)]
            self.wait(futures)
            # ~~~~~~~~~~~~~~~~~~~~~~
            cont, step = True, 0
            while cont:
            # {
                step += 1
                print(f"PARCOPSA. Step {step}")
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                futures = [executor.submit(self.solvers[i].iterate) for i in range(self.nthrds)]
                self.wait(futures)
                best_sol = self.get_best()
                self.communicate(best_sol)
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                cont = all(not self.solvers[i].terminate for i in range(self.nthrds))
            # }
        # }
        for i in range(self.nthrds):
            self.solvers[i].finalise()
    # }
# }

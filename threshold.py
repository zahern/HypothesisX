"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
IMPLEMENTATION: THRESHOLD ACCEPTANCE
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""
BACKGROUND - THRESHOLD ACCEPTANCE

TA is a meta-heuristic very similar in structure to Simulated Anealling 

Key Features:
Acceptance Criterion: The algorithm maintains a threshold value, often denoted as 
Δ. It generates candidate solutions and accepts those that are within a certain margin of the current
 best solution, considering the threshold Δ.

Perturbation: TA often includes a perturbation mechanism where the current solution is modified
 slightly to explore neighboring solutions in the search space. This perturbation can help in 
 escaping local optima and exploring the solution space more extensively.

Iteration: The algorithm iterates through a series of steps where it generates a candidate solution, 
evaluates it using the objective function, applies the acceptance criterion, and updates the current 
solution accordingly.

Workflow:
Initialization: Start with an initial solution or generate one randomly.
Iterative Improvement: Repeat until a stopping criterion is met (e.g., number of iterations, convergence criteria).
Generate a new candidate solution by perturbing the current solution.
Evaluate the candidate solution using the objective function.
Decide whether to accept the candidate solution based on the acceptance criterion (typically comparing 
it to the current best solution).
Update the current solution if the candidate is accepted.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


''' ---------------------------------------------------------- '''
''' LIBRARIES                                                  '''
''' ---------------------------------------------------------- '''
try:
    from search import*
except ImportError:
    from .search import*
import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Callable, Tuple

''' ---------------------------------------------------------- '''

overall_best_solution = None  # PARSA: Reference to best solution
lock = threading.Lock()  # PARSA: Mutex - synchronization primitive

''' ---------------------------------------------------------- '''
''' CLASS FOR THRESHOLD ACCEPTANCE                             '''
''' ---------------------------------------------------------- '''

class TA(Search):
# {
    def __init__(self, param:Parameters, init_sol, ctrl, idnum=0):
    # {
        super().__init__(param, idnum)     # Call base class constructor

        self.max_threshold, self.max_steps, self.max_iter = ctrl
        self.grad = - self.max_threshold / (self.max_steps-1)

        # Note: Subtract one because the first step at t=tI must be included as a step?

        self.max_no_impr = 3  # Max number of steps permitted without improvements
        self.terminate = False  # Termination flag

        self.step = 0  # Current step
        self.current_sol = init_sol  # Current solution
        self.best_sol = None  # Best solution
        self.archive = []  # Archive of solutions
        self.start = None  # Start time
        self.accepted, self.not_accepted = 0, 0  # Counters
        self.threshold = self.max_threshold
        self.comm_int = 1  # Communication interval for PARTA
        self.idnum = idnum

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
    # }

    def open_files(self):
    # {
        str_idnum = str(self.idnum)
        self.results_file = open("ta_results[" + str_idnum + "].txt", "w")
        self.progress_file = open("ta_progress[" + str_idnum + "].txt", "w")
        self.archive_file = open("ta_archive[" + str_idnum + "].txt", "w")
        self.debug_file = open("ta_pert[" + str_idnum + "].txt", "w")
        self.best_file = open("ta_best[" + str_idnum + "].txt", "w")
    # }

    def close_files(self):
    # {
        self.results_file.close()
        self.progress_file.close()
        self.archive_file.close()
        self.debug_file.close()
        self.best_file.close()
    # }
    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def reduce_threshold(self):
    # {
        self.threshold = self.threshold + self.grad

        # Note: threshold(t) = grad * t + max_threshold
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def curr_score(self, i):
        return self.current_sol.obj(i)

    def best_score(self, i):
        return self.best_sol.obj(i)

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def choose_starting_solution(self):
    # {
        print(f"TA[{str(self.idnum)}] - Generating a starting solution")
        attempts = 0
        while attempts < 100:
        # {
            print(f"TA[{str(self.idnum)}]. Attempt={str(attempts)}")
            sol = self.generate_solution()  # Generate a solution
            sol, converged = self.evaluate(sol) # Evaluate the solution
            if converged:
                return sol  # Exit the procedure with a valid solution
            attempts += 1   # Update attempts
        # }
        return None
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.  Evaluate how good a solution is                 '''
    ''' ---------------------------------------------------------- '''
    def evaluate(self, sol):
    # {
        sol, converged = self.evaluate_solution(sol)
        return sol, converged
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Prepare to run the algorithm                     '''
    ''' ---------------------------------------------------------- '''
    def copy_solution(self, sol):
    # {
        copy_sol = copy.deepcopy(sol)
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
        print(f"TA[{self.idnum}]. Starting solution: ", self.current_sol.get_obj())
        self.report_progress(self.progress_file)
        self.log_solution("Initial Solution", self.best_sol, file=self.results_file)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Finish up                                        '''
    ''' ---------------------------------------------------------- '''
    def finalise(self):
    # {
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
    def accept_change_threshold(self, before, after):
    # {
        """ Docstring """
        crits = self.param.criterions

        # Accept if after is strictly better or if it worse but within tolerance
        # i.e., if (be - af) sign <= thr
        # When sign=1:  be - af <= thr =>  af >= be - thr
        # When sign=-1: af - be <= thr => af <= be  + thr
        accept_i = [(before[i] - after[i]) * crit[1] <= self.threshold
                    for i, crit in enumerate(crits)]
        return all(accept_i)
    # }

    # Note: This acceptance strategy does ot use the temperature!
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
        return self.accept_change_threshold(before, after)
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
        self.perturb_solution(self.current_sol)
    # }

    def perturb_multi(self):
    #{
        num_par = len(self.archive)
        chosen = np.random.choice(num_par)
        chosen_sol = self.archive[chosen]
        self.perturb_solution(chosen_sol)
    # }

    def perturb_solution(self, sol):
    # {
        curr_score = [sol.obj(i) for i in range(self.nb_crit)]
        new_sol = self.copy_solution(sol)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        choices = []
        choices.append(self.perturb_asfeature)

        if self.param.isvarnames is not None:
            choices.append(self.perturb_isfeature)

        if self.param.asvarnames is not None:
            choices.append(self.perturb_randfeature)

        if sol['randvars'] is not None:
            choices.append(self.perturb_distribution)

        if self.param.ps_bctrans is None or self.param.ps_bctrans:
            choices.append(self.perturb_bcfeature)

        if self.param.ps_cor is None or self.param.ps_cor:
            choices.append(self.perturb_corfeature)

        if self.optimise_class:
            if sol['class_params_spec'] is not None:
                choices.append(self.perturb_class_paramfeature)

        if self.optimise_membership:
            if sol['member_params_spec'] is not None:
                choices.append(self.perturb_member_paramfeature)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Call the chosen perturbation strategy
        choice = np.random.choice(choices)
        new_sol = choice(new_sol)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        new_sol, converged = self.evaluate(new_sol)
        if converged:
        # {
            new_score = [new_sol.obj(i) for i in range(self.nb_crit)]
            args = (curr_score, new_score)
            if self.accept_change(*args):
            # {
                accd = True
                self.accepted += 1  # Tracking acceptances - Increment counter
                self.current_sol = new_sol  # Set new current solution
                self.update_best(new_sol)   # Update the best solution if necessary
                # Optional: self.log_solution("Perturbation", self.current_sol, file=self.results_file)
            # }
            else:
                accd = False
                self.not_accepted += 1 # Tracking non-acceptances - Increment counter
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.log_kpi(new_sol, self.debug_file, accd) # Report to file
        # }
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def update_best(self, sol):
    # {
        if self.nb_crit == 1:
        # {
            if is_better(sol.obj(0), self.best_sol.obj(0), self.param.sign_crit(0)):
                self.best_sol.copy_solution(sol)
                self.no_impr = 0
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
                if sol not in _dominated: _dominated.append(sol)
            # }
        # }

        # Remove all solutions 'add_sol' dominates and add 'add_sol'
        self.archive = [sol for sol in self.archive if sol not in _dominated] + [add_sol]
        #DEBUG: print("The new solution is non-dominated. It dominates ",len(_dominated)," existing solution(s)")

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

    def handle_static_archive(self, size):
    # {
        if len(self.archive) == size:
            self.no_impr += 1 # Increment non improvement counter

        if self.no_impr > self.max_no_impr:  # Key step enabling performance
        # {
            print("ARCHIVE STATIC. RESTORE A NON DOMINATED SOLUTION.")
            choice = np.random.randint(len(self.archive))
            self.current_sol = self.archive[choice]
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
                overall_best_solution = self.best_sol  # Update overall best solution
            elif overall_best_solution is not None and \
                is_worse(self.best_sol.obj(0), overall_best_solution.obj(0), self.param.sign_crit(0)):
                self.update_best(overall_best_solution)  # Revise best solution of current TA solver
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
            if count >= self.max_iter:
                break
        # }
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # TURNED OFF. UNSATISFACTORY PERFORMANCE SO FAR!
        #if (self.step) % 2:
        #    self.best_sol = self.improve(self.best_sol)  # Apply local improvement
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.report_progress(self.progress_file)  # Report current state
        self.report_progress()  # Report to the screen
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.reduce_threshold()  # Reduce the threshold accordingly
        self.reset_current_solution()  # Reset current solution conditionally
    # }

    def complete(self):
        return (self.threshold <= 0) # or self.step > self.max_steps

    def iterate(self, synch=False):
    # {
        if not self.complete():
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
    def report_progress(self, file=None):
    # {
        text = f"TA[{self.idnum}]. Progress @ Step={self.step}; Curr=[{self.current_sol.concatenate_obj()}]; "

        if self.nb_crit == 1:
            text += f"Best=[{self.best_sol.concatenate_obj()}]; "
        else:
            text += f"Archive Size={len(self.archive)}; "

        now = time.time()
        elapsed = now - self.start
        text += f"Elapsed Time={round(elapsed,2)}"

        print(text, file=file)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def log_kpi(self, sol, file=None, accept=True):
    # {
        text = sol.concatenate_obj()
        text += ",true," if accept else ",false,"
        text += str(self.step)
        print(text, file=file)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def log_solution(self, descr, sol, file=None):
    # {
        header = f"Solution: {descr}"
        nchar = len(header)
        line = "_" * nchar
        print(line, file=file)
        print(header, file=file)
        print(line, file=file)
        print("Objectives:", file=file)
        for i in range(self.nb_crit):
        # {
            opt = "Maximise" if self.param.sign_crit(i) == 1 else "Minimise"
            text = f"[{i}]. ({opt}) {self.param.crit(i)} = {round(sol.obj(i), 4)}"
            print(text, file=file)
        # }
        report_model_statistics(sol['model'], file)
        print("", file=file)  # Empty line
        self.log_decision(sol, file=file)
        print("", file=file)  # Empty line
    # }

    def log_decision(self, sol, file=None):
    # {
        print("asvars=", sol['asvars'], file=file)
        print("isvars=", sol['isvars'], file=file)
        print("randvars=", sol['randvars'], file=file)
        print("bcvars=", sol['bcvars'], file=file)
        print("corvars=", sol['corvars'], file=file)
        print("bctrans=", sol['bctrans'], file=file)
        print("asc_ind=", sol['asc_ind'], file=file)
        print("class_param_spec=", sol['class_params_spec'], file=file)
        print("member_params_spec=", sol['member_params_spec'], file=file)
        print("model=", sol['model'].descr, file=file)                                              
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
        # {
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
        sign = self.param.sign_crit(0)  # Shortcut to optimisation sign

        if overall_best is None or is_better(self.best_sol.obj(0), overall_best.obj(0), sign):
            overall_best = self.copy_solution(self.best_sol)

        # Current best solution is worse, so terminate:
        terminate = is_worse(self.best_sol.obj(0), overall_best.obj(0), sign)
        return overall_best, terminate
    # }

    def run_search_latent(self):
    # {
        overall_best_sol = None
        for q in range(1, self.max_classes):
            # {
            print('RunSearchLatent. #classes=', q)
            self.param.latent_class = False if q == 1 else True
            self.param.num_classes = q
            self.run_search()
            self.current_sol['class_num'] = q
            self.best_sol['class_num'] = q
            if self.nb_crit == 1:
                overall_best_sol, terminate = self.search_latent_update_single(overall_best_sol)
            else:
                terminate = self.search_latent_update_multi(q)
            if terminate: break
        # }
        print(f"Best solution has {overall_best_sol['class_num']} latent classes")
        self.finalise()
        # }
    # }

    def revise_threshold(self, new_threshold):
    # {
        self.max_threshold = new_threshold
        self.grad = - self.max_threshold / self.max_steps
    # }

# }


''' ----------------------------------------------------------- '''
''' PARALLEL THRESHOLD ACCEPTANCE                               '''
''' ----------------------------------------------------------- '''
class PARTA():
# {
    """ Docstring """

    ''' ---------------------------------------------------------- '''
    ''' Function. Constructor                                      '''
    ''' ---------------------------------------------------------- '''
    def __init__(self, param: Parameters, ctrl, nthrds=1):
    # {
        self.nthrds = nthrds
        init_sol = None

        # Define and setup independant solvers
        self.solvers = [TA(param, ctrl, init_sol, idnum=i) for i in range(nthrds)]

        self.choose_custom_threshold()  # Optional

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
    def choose_custom_threshold(self, options=None):
    # {
        if len(options) > 0:
            for _, solver in enumerate(self.solvers):
                solver.revise_threshold(np.random.choice(options))
        else:
            for _, solver in enumerate(self.solvers):
                solver.revise_threshold(np.random.randint(100, 10000))
    # }
# }


''' ----------------------------------------------------------- '''
''' PARALLEL COOPERATIVE SIMULATED ANNEALING                    '''
''' ----------------------------------------------------------- '''
class PARCOPTA(PARTA):
# {
    """ Docstring """

    ''' ---------------------------------------------------------- '''
    ''' Function. Constructor                                      '''
    ''' ---------------------------------------------------------- '''
    def __init__(self, param: Parameters, ctrl, nthrds=1):
    # {
        super().__init__(param, ctrl, nthrds)  # Call base class constructor
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



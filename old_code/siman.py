"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
IMPLEMENTATION: SIMULATED ANNEALING
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#from akshay_test import member_params_spec

#from scipy.special import kwargs

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
try:
    from .search import*
    from .latent_class_constrained import LatentClassCoefficients
except ImportError:
    from search import*
    from latent_class_constrained import LatentClassConstrained
import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import Callable, Tuple
from datetime import datetime
import random
import string
import re


''' ---------------------------------------------------------- '''

overall_best_solution = None  # PARSA: Reference to best solution
lock = threading.Lock()  # PARSA: Mutex - synchronization primitive


def generate_random_run_name(prefix="run"):
    # Generate a random string of 6 characters
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
    def __init__(self, param:Parameters, init_sol, ctrl, idnum=generate_random_run_name(), **kwargs):
    # {
        super().__init__(param, idnum, **kwargs)     # Call base class constructor

        tI, tF, max_temp_steps, max_iter = ctrl  # Extract form 'ctrl'


        self.start_time = time.time()
        # Set parameters:
        self.max_time = kwargs.get('max_time', 3600*12)    # Maximum Allowable Run Time (Terminate
        self.max_total_iter = kwargs.get('max_total_iter', 10000)
        self.tI = tI                # Starting temperature
        self.tF = tF                # Final temperature
        self.max_temp_steps = max_temp_steps    # Maximum number of temperature steps
        self.max_iter = max_iter    # Maximum number of iterations at each temperature step
        self.max_no_impr = 3        # Max number of steps permitted without improvements
        self.terminate = False      # Termination flag
        self.rate = np.exp((1.0 / (self.max_temp_steps-1)) * np.log(self.tF/self.tI)) # Temperature reduction rate

        # Note: tF = tI * power(rate, max_temp_steps-1)
        # Note: Subtract one because the first step at t=tI must be included as a step

        self.no_impr = 0            # Current number of iterations without improvement
        self.step = 0               # Current temperature step
        self.t = 0                  # Current temperature
        self.current_sol = init_sol # Current solution
        self.best_sol = None        # Best solution
        self.archive = []           # Archive of solutions
        self.start = None           # Start time
        self.accepted, self.not_accepted = 0, 0 # Counters
        self.comm_int = 1           # Communication interval for PARSA
        self.idnum = idnum

        self.stlt_coeff_mem = None 
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

    def get_run_time(self):
        '''Gets the current run_time in seconds'''
        end_time = time.time()
        return end_time - self.start_time

    def open_files(self):
    # {
        str_idnum = str(self.idnum)
        self.results_file = open("siman_results[" + str_idnum + "].txt", "w")
        self.progress_file = open("siman_progress[" + str_idnum + "].txt", "w")
        self.archive_file = open("siman_archive[" + str_idnum + "].txt", "w")
        self.debug_file = open("siman_pert[" + str_idnum + "].txt", "w")
        self.best_file = open("siman_best[" + str_idnum + "].txt", "w")
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
    def curr_score(self, i):
        return self.current_sol.obj(i)

    def best_score(self, i):
        return self.best_sol.obj(i)

    def return_best(self):
        return self.best_sol

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def choose_starting_solution(self):
    # {
        print(f"SA[{str(self.idnum)}] - Generating a starting solution")
        attempts = 0
        while attempts < 100:
        # {
            print(f"SA[{str(self.idnum)}]. Attempt={str(attempts)}")
            sol = self.generate_solution()  # Generate a solution
            if self.param.latent_class:
                #print('repair the solutution')
                sol = self.repair_solution_for_clarity(sol)
            sol, converged = self.evaluate(sol) # Evaluate the solution

            if converged:
                return sol  # Exit the procedure with a valid solution
            attempts += 1   # Update attempts
            print('attempt number', attempts)
        # }
        return None
    # }
    def repair_solution_for_clarity(self, solution):
        '''
        This function repairs a solutions class Membership so similiar variables are not
        place in the class
        For example:
        Class 1: cannot Have Price and Price_2

        '''

        #make sure i is consistent with the asvars and isvars
        if self.param.latent_class is False:
            #if any in solution ['isvars'] is in solution['asvars'] remove the isvars items
            solution['isvars'] = [i for i in solution['isvars'] if i not in solution['asvars']]
            #solution['isvars'] #needs to be unique from the asvars 
            solution['randvars'] = {k: v for k, v in solution['randvars'].items() if k not in solution['isvars']}
            return solution



        member_part = solution.get('member_params_spec')
        for ii, arr in enumerate(member_part):
            # Create a set to track seen variable names
            seen = set()
            # Filter the array to remove similar variables
            member_part[ii] = [
                item for item in arr
                if not (re.sub(r'_\d+', '', item) in seen or seen.add(re.sub(r'_\d+', '', item)) ) # Add the base name to seen
            ]
           # re.sub(r'_\d+', '', item)
        solution['member_parms_spec'] = member_part

        if self.param.LCR is not None:
            class_i = 0
            for i in self.param.LCR.classes.values():
                

                #solution['asvar'] = list(set(solution['asvar']).union(i.get('req_asvar')))
                if i.get('req_asvar') is not None or len(i.get('req_asvar')) > 0:
                     solution['class_params_spec'][class_i] = list(set(solution['class_params_spec'][class_i]).union(i.get('req_asvar')))
                else:
                    solution['class_params_spec'][class_i] = list(set(solution['class_params_spec'][class_i]))
                if i.get('req_isvars') is not None or len(i.get('req_isvars')) > 0:
                    solution['class_params_spec_is'][class_i] = list(set(solution['class_params_spec_is'][class_i]).union(i.get('req_isvars')))
                else:
                    solution['class_params_spec_is'][class_i] = list(set(solution['class_params_spec_is'][class_i]))
                #solution['isvars'] = list(set(solution['isvars']).union(i.get('req_isvars')))
                #solution['class_params_spec_is'][class_i] = list(set(solution['class_params_spec_is'][class_i]).union(i.get('req_isvars')))
                class_i +=1

                
            

        return solution


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
        self.t = self.tI  # Set current temperature

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
        print(f"SA[{self.idnum}]. Starting solution: ", self.current_sol.get_obj())
        self.report_progress(self.progress_file)
        self.log_solution("Initial Solution", self.current_sol, file=self.results_file)
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
    def accept_change_metrop(self, before, after):
    # {
        """ Evaluate Metropolis function for each objective """
        crits = self.param.criterions

        # Note: crit[1] is the sign and equivalent to crits[i][1]
        rn = np.random.rand()
        accept_i = [np.log(rn) < (crit[1] * (after[i] - before[i]) / self.t)
                    for i, crit in enumerate(crits)]
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
        curr_score = [sol.obj(i) for i in range(self.nb_crit)]
        new_sol = self.copy_solution(sol)

        # ~~~~~~~~~~~~
        
        choices = []


        #The idea is we only want to play with the membership feautures first
        if self.optimise_membership is not True:

            choices.append(self.perturb_asfeature)

            if self.param.isvarnames is not None:
                choices.append(self.perturb_isfeature)

            if self.param.asvarnames is not None:
                #Not latent so can add
                #if sol['member_params_spec'] is None:
                if self.param.allow_random:
                    choices.append(self.perturb_randfeature)


            if sol['randvars'] is not None and self.param.allow_random:
                choices.append(self.perturb_distribution)

            if self.param.ps_bctrans is None or self.param.ps_bctrans and self.param.allow_bcvars:
                choices.append(self.perturb_bcfeature)

            if self.param.ps_cor is None or self.param.ps_cor and self.param.allow_corvars:
                choices.append(self.perturb_corfeature)

        if self.optimise_membership:
        # {
            if sol['member_params_spec'] is not None:
                choices.append(self.perturb_member_paramfeature)
        # }
        
        if self.optimise_class:
        # {
            if sol['class_params_spec'] is not None:
                choices.append(self.perturb_class_paramfeature)
            if sol['class_params_spec_is'] is not None:
                print('to do perturb is feature')
                choices.append(self.perturb_class_paramfeature)

        # }
        
       
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Call the chosen perturbation strategy
        choice = np.random.choice(choices)
        #print('perturbation choice:', choice.__name__)
        new_sol = choice(new_sol)
        
        if new_sol is None:
           # print('perturbation choice is none:', choice.__name__)
            return sol
        #print('perturbation choice is fine:', choice.__name__)
        new_sol = self.repair_solution_for_clarity(new_sol)
        
        #TODO add some contraint to ensure membership is cool beans.

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Solution evaluation and acceptance handling
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
                self.update_best(new_sol)  # Update the best solution if necessary
                # Optional: self.log_solution("Perturbation", self.current_sol, file=self.results_file)
            # }
            else:
                accd = False
                self.not_accepted += 1  # Tracking non-acceptances - Increment counter
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.log_kpi(new_sol, self.debug_file, accd)  # Report to file
        # }
        else:
            new_sol =sol
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
                print('return the coefficients')
                sol.get('coeff_est')
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

                if not any(np.array_equal(sol, d) for d in _dominated):
                    _dominated.append(sol)
                #if sol not in _dominated: _dominated.append(sol)
            # }
        # }

        # Remove all solutions 'add_sol' dominates and add 'add_sol'
        print('before archvie: ', len(self.archive))
        self.archive = [
                           s for s in self.archive
                           if not any(np.array_equal(s, d) for d in _dominated)
                       ] + [add_sol]
        print('after archive:' , len(self.archive))
        #self.archive = [sol for sol in self.archive if sol not in _dominated] + [add_sol]

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
            if count % 1000 == 0:
                print(f'Iteration at {count}')
            if count >= self.max_iter:
                break
        # }
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # TURNED OFF. UNSATISFACTORY PERFORMANCE SO FAR!
        print(f'step number {self.step}')
        #if (self.step) % 2:
        #    self.best_sol = self.improve(self.best_sol)  # Apply local improvement
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.report_progress(self.progress_file)  # Report current state
        self.report_progress()  # Report to the screen
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t = self.rate * self.t  # Reduce the temperature accordingly

        self.reset_current_solution()  # Reset current solution conditionally
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
    def report_progress(self, file=None):
    # {
        text = f"SA[{self.idnum}]. Progress @ Step={self.step}; Curr=[{self.current_sol.concatenate_obj()}]; "

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
    ''' Function. obj[1], obj[2], t/f                                 '''
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
        print("asvars = ", sol['asvars'], file=file)
        print("isvars = ", sol['isvars'], file=file)
        print("randvars = ", sol['randvars'], file=file)
        print("bcvars = ", sol['bcvars'], file=file)
        print("corvars = ", sol['corvars'], file=file)
        print("bctrans = ", sol['bctrans'], file=file)
        print("asc_ind = ", sol['asc_ind'], file=file)
        print("class_param_spec_is = ", sol['class_params_spec_is'], file=file)
        print("class_param_spec = ", sol['class_params_spec'], file=file)
        print("member_params_spec = ", sol['member_params_spec'], file=file)
        #print("member_params_spec_is = ", sol['member_params_spec_is'], file=file)
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

        # Define and setup independent solvers
        self.solvers = []
        for q in range(max_classes):
        # {
            self.param.latent_class = False if q == 1 else True
            self.param.num_classes = q
            solver_q = SA(self.param, ctrl, init_sol, idnum=q)
            self.solvers.append(solver_q)

        # }

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.solvers[q].run) for q in range(self.max_classes)]

        for future in as_completed(futures):
            result = future.result()  # This will wait until each task completes

        for q in range(max_classes):
        # {
            self.solvers[q].current_sol['class_num'] = q
            self.solvers[q].best_sol['class_num'] = q
            self.solvers[q].finalise()
        # }

    # }

    ####################################################################################


    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def revise_tI(self, new_tI):
    # {
        self.tI = new_tI
        self.rate = np.exp((1.0 / self.max_temp_steps) * np.log(self.tF/self.tI))
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

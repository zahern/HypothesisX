"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
IMPLEMENTATION: HARMONY SEARCH
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
BACKGROUND - HARMONY SEARCH

Initialization: The algorithm starts by initializing a population of candidate solutions, called "harmonies." 
Each harmony represents a potential solution to the optimization problem.

Improvisation: Similar to musicians trying out different combinations of notes, Harmony Search generates new 
solutions by combining elements from existing harmonies. This is done through a process called "harmony memory 
consideration."

Evaluation: Each newly generated harmony is evaluated based on a fitness function, which measures how 
good the solution is in terms of solving the optimization problem.

Updating Harmony Memory: The best harmonies are selected to update the harmony memory, replacing the worst 
harmonies if they are better.

Memory Consideration: During the improvisation process, the algorithm considers both the current harmonies 
and the memory of the best solutions found so far to guide the search towards better solutions.

Pitch Adjustment: Similar to how musicians adjust their notes to achieve harmony, Harmony Search introduces 
randomness by adjusting certain elements (parameters) of the harmonies. This helps in exploring different
regions of the search space.

Termination: The process continues for a certain number of iterations or until a termination criterion is 
met (e.g., a satisfactory solution is found).

"""

''' ---------------------------------------------------------- '''
''' LIBRARIES                                                  '''
''' ---------------------------------------------------------- '''
#from search import*
import matplotlib.pyplot as plt
import datetime
import math
import pandas as pd
try:
    from .search import*
except ImportError:
    from search import*

''' ---------------------------------------------------------- '''
''' CONSTANTS                                                  '''
''' ---------------------------------------------------------- '''

sol_keys = ['asvars', 'isvars', 'randvars', 'bcvars', 'corvars', 'bctrans', 'cor']

''' ---------------------------------------------------------- '''
''' CLASS FOR HARMONY SEARCH (HS)                              '''
''' ---------------------------------------------------------- '''
class HarmonySearch(Search):
# {
    """ Docstring """

    # ===================
    # CLASS PARAMETERS
    # ===================

    """
    int mem_size: Harmony memory size / Defaults to 10.
    int min_classes: Minimum number of latent classes. Defaults to 1
    int max_classes: Maximum number of latent classes. Defaults to 5
    float min_harm: Minimum harmony memory consideration rate / Defaults to 0.6.
    float max_harm: Maximum harmony memory consideration rate / Defaults to 0.9.
    float max_pitch: Maximum pitch adjustment rate / Defaults to 0.85
    float min_pitch: Minimum pitch adjustment / Defaults to 0.3
    int maxiter: Maximum iteratioms / Defaults to 30.
    float prop_local: Proportion of iterations without local search /  Defaults to 0.8.
    int threshold: Threshold to compare new solution with worst solution / Defaults to 15

    bool termination_override: termination flag that overrides the default / Defaults to False.
        If true, the search will run for each number of latent classes
        between min_classes and max_classes

    iter_prop: Proportion of maxiter after which local search is initiated / float
    """

    # =======================
    # CLASS FUNCTIONS
    # =======================

    """
    1. set_control_parameters()
    2. create_opposite_solution(self, sol);
    3. initialize_memory(self, nb_sols);
    4. build_solution(self, memory, prop);
    5. pitch_adjustment(self, sol, pitch);
    6. get_best_features(self, memory);
    7. local_search(self, improved_harmony, iter, pitch);
    8. improvise(self);
    9. run(self, latent=False);
    10. sort_memory(self, mem);
    11. insert_solution(self, solution):
    """

    ''' ---------------------------------------------------------- '''
    ''' Function. Constructor                                      '''
    ''' ---------------------------------------------------------- '''
    def set_control_parameters(self, max_harm=0.9, min_harm=0.6, max_pitch=0.85, min_pitch=0.3,
        max_mem=10, maxiter=30, threshold=15, prop_local=0.8, generate_plots=True):
    # {
        self.max_harm = max_harm  # Maximum Harmony Memory Considering Rate / float
        self.min_harm = min_harm  # Minimum Harmony Memory Considering Rate / float
        self.max_pitch = max_pitch  # Maximum Pitch Adjusting Rate / float
        self.min_pitch = min_pitch  # Minimum Pitch Adjusting Rate / float
        self.max_mem = max_mem  # Harmony memory size / int
        self.maxiter = maxiter  # Maximum number of iterations / int
        self.threshold = threshold  # Convergence threshold /float
        self.prop_local = prop_local  # Proportion of maxiter
        self.perform_local = int(self.prop_local * self.maxiter)  # When to apply local search
        self.generate_plots = generate_plots
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Constructor                                      '''
    ''' ---------------------------------------------------------- '''
    def __init__(self, param: Parameters):
    # {
        super().__init__(param) # Call base class constructor
        self.set_control_parameters() # Assume default options - hence no inputs here
        self.pitch = self.max_pitch
        self.memory = []  # List of solutions is empty
        self.all_solutions = [] # Avoid generating same solution twice

        self.results_file = open("results.txt", "w")  # File to output results
        self.progress_file = open("progress.txt", "w")  # File to output convergence information
# }

    ''' ---------------------------------------------------------------- '''
    ''' Function                                                         '''
    ''' ---------------------------------------------------------------- '''
    def sort_memory(self, mem):
    # {
        if self.param.nb_crit > 1:
            mem = self.non_dominant_sorting(mem)
        else:
            mem = sorted(mem, key=lambda sol: sol.obj[0])
        return mem
    # }

    ''' ---------------------------------------------------------------- '''
    ''' Function                                                         '''
    ''' ---------------------------------------------------------------- '''
    def create_opposite_solution(self, sol):
    # {
        opp_sol = self.generate_solution()
        for key in sol_keys:  # Iterate through variable types
        # {
            skip = (not opp_sol[key] or isinstance(opp_sol[key], bool) or getattr(self.param, 'ps_' + key))
            if skip:
                continue  # Skip current loop
            else:
            # {
                opp_sol[key] = [v for v in opp_sol[key] if v not in sol[key]]  # Filter out elements in sol[key]
                if self.param.ps_intercept is None:
                    opp_sol['asc_ind'] = not sol['asc_ind']
            # }
        # }
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.param.avail_rvars:
            opp_sol['randvars'] = {k: self.param.generator.choice(self.param.distr)
                                   for k in opp_sol['randvars'] if k in opp_sol['asvars']}
        else:
            opp_sol['randvars'] = {}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        opp_sol['corvars'] = [corvar for corvar in opp_sol['corvars'] if corvar in opp_sol['randvars']]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.param.avail_bcvars:
            opp_sol['bcvars'] = [bcvar for bcvar in opp_sol['bcvars']
                                 if bcvar in opp_sol['asvars'] and bcvar not in opp_sol['corvars']]
        else:
            opp_sol['bcvars'] = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.revise_solution('class_params_spec', opp_sol, sol)
        self.revise_solution('member_params_spec', opp_sol, sol)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # }

    ''' ---------------------------------------------------------------- '''
    ''' Function. Initialization of harmony search memory                '''
    ''' ---------------------------------------------------------------- '''
    def initialize_memory(self, nb_sols):
    # {
        """ This function initializes the harmony memory and opposite
        harmony memory with unique random solutions. The harmony memory
        stores initial solutions, while the opposite harmony memory
        stores solutions that include variables not included in the
        harmony memory. If the generated solution converges, it's added
        to the harmony memory. Otherwise, the function generates an
        "opposite" solution and, if it converges, adds it to the
        opposite harmony memory.
        """

        mem, opp_mem = [], []
        for counter in range(30000):
        # {
            sol = self.generate_solution()  # Generated solution
            sol, converged = self.evaluate_solution(sol)
            if converged: mem.append(sol) # Add new solution to list

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Create opposite solution that has non_included variables
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            opp_sol = self.create_opposite_solution(sol)
            opp_sol, converged = self.evaluate_solution(opp_sol)
            if converged: opp_mem.append(opp_sol)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            mem += opp_mem    # Aggregate solutions
            mem = get_unique(mem, 0)  # Keep unique solutions only - Compare by first objective

            # QUERY: WHY NOT FILTER BY sol.obj[1] AS WELL?

            mem = [sol for sol in mem if abs(sol.obj[0]) < BOUND] # Filter out poor solutions
            if len(mem) >= nb_sols:
                return mem[:nb_sols]  # Exit and return list of solutions
        # }
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return mem # Failed to generate required number of solutions
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Build new solution using Harmony Memory          '''
    ''' A new solution, could either be built from an existing one '''
    ''' or constructed randomly.                                   '''
    ''' ---------------------------------------------------------- '''
    def build_solution(self, memory, prop):
    # {
        """ This function decides whether to build a new solution from an existing solution
        in the harmony memory or to generate a completely new solution, based on a random number and the
        Harmony Memory Consideration Rate (HMCR). If the random number is less than or equal to prop,
        it selects a proportion of the features from a randomly chosen existing solution to build the new solution.
        Otherwise, it generates a completely new solution """


        bin = [0,1] # Binary values
        prob = [1-prop, prop]  # Range
        new_sol = Solution(nb_crit=self.nb_crit)    # Create a new solution object

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # IS THIS NECESSARY?
        '''fronts, pareto = None, None
        if nb_crit > 1: # {
            memory = self.non_dominant_sorting(memory)
            fronts = self.get_fronts(memory)
            pareto = self.get_pareto(fronts, memory)
        # }'''
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.param.generator.rand() > prop:
            new_sol = self.generate_solution()  # Generate a new solution
        else:
        # {
            choice = self.param.generator.choice(len(memory)) # Choose one of the member solutions
            chosen_sol = memory[choice] # Define reference to the chosen member solution

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # OPTIONAL CODE.
            # size = len(chosen_sol['asvars'])
            # new_asvars_index = self.param.generator.choice(bin, size=size, p=prob)
            # new_asvars = [i for (i, v) in zip(chosen_sol['asvars'], new_asvars_index) if v]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # Randomly select a subset of the variables from the chosen solution
            size = int((len(chosen_sol['asvars'])) * prop)
            new_asvars = list(self.param.generator.choice(chosen_sol['asvars'], size=size, replace=False))
            n_asvars = sorted(list(set().union(new_asvars, self.param.ps_asvars)))
            new_asvars = self.remove_redundant_asvars(n_asvars, self.param.trans_asvars, self.param.asvarnames)
            new_sol['asvars'] = new_asvars

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Randomly select a subset of the variables from the chosen solution
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            size = int((len(chosen_sol['isvars'])) * prop)
            new_isvars = list(self.param.generator.choice(chosen_sol['isvars'], size=size, replace=False))
            new_isvars = sorted(list(set().union(new_isvars, self.param.ps_isvars)))
            new_sol['isvars'] = new_isvars

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Include variables in new solution based on the chosen solution
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            new_randvars = {k: v for k, v in chosen_sol['randvars'].items() if k in new_asvars}
            new_sol['randvars'] = new_randvars

            new_bcvars = [var for var in chosen_sol['bcvars']
                            if var in new_asvars and var not in self.param.ps_corvars]
            new_sol['bcvars'] = new_bcvars

            new_corvars = chosen_sol['corvars']
            if new_corvars:
                new_corvars = [var for var in chosen_sol['corvars']
                                if var in new_randvars.keys() and var not in new_bcvars]
            new_sol['corvars'] = new_corvars

            # Take fit_intercept from chosen solution
            new_sol['asc_ind'] = chosen_sol['asc_ind']

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if chosen_sol['class_params_spec'] is not None:
            # {
                class_params_spec = copy.deepcopy(chosen_sol['class_params_spec'])
                for ii, class_params in enumerate(class_params_spec):
                # {
                    class_params_index = self.param.generator.choice(bin, size=len(class_params), p=prob)
                    class_params_spec[ii] = np.array([i for (i, v) in zip(class_params, class_params_index) if v],
                                                     dtype=class_params.dtype)
                # }
                new_sol['class_params_spec'] = class_params_spec
            # }
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if chosen_sol['member_params_spec'] is not None:
            # {
                member_params_spec = copy.deepcopy(chosen_sol['member_params_spec'])
                for ii, member_params in enumerate(member_params_spec):
                # {
                    member_params_index = self.param.generator.choice(bin, size=len(member_params), p=prob)
                    member_params_spec[ii] = np.array([i for (i, v) in zip(member_params, member_params_index) if v],
                                                      dtype=member_params.dtype)
                # }
                new_sol['member_params_spec'] = member_params_spec
            # }
        # }

        return new_sol
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def remove_non_unique_solutions(self):
    # {
        seen_tuple = set()  # Create an empty set
        new_memory = list()  # Create an empty list
        crit = self.param.criterions[:self.nb_crit]
        for sol in self.memory:
        # {
            sol_tuple = tuple([sol[crit[0]], sol[crit[1]]])
            if sol_tuple not in seen_tuple:
            # {
                seen_tuple.add(sol_tuple)   # Revise what has been seen
                new_memory.append(sol)      # Update list of unique solutions
            # }
        # }
        self.memory = new_memory
    # }

    ''' ------------------------------------------------------------ '''
    ''' Function. Insert solution and filter out non-unique solutions'''
    ''' ------------------------------------------------------------ '''
    def insert_solution(self, solution):
    # {
        self.memory.append(copy.deepcopy(solution))
        self.remove_non_unique_solutions()
        self.memory = self.sort_memory(self.memory)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Performs the pitch adjustment operation to       '''
    ''' fine-tune a given solution. The process includes adding    '''
    ''' new features or removing existing ones based on a binary   '''
    ''' indicator. The resulting solution is evaluated and inserted'''
    ''' The solutions in memory are then filtered                  '''
    ''' ---------------------------------------------------------- '''
    def pitch_adjustment(self, sol, pitch):
    # {
        # QUERY. IS DEEP COPY REQUIRED?
        adj = copy.deepcopy(sol)  # Adjusted solution

        # pitch adjustment: add/remove as variables
        if self.param.generator.rand() <= pitch: adj = self.perturb_asfeature(sol)
        # pitch adjustment: add|remove is variables
        if self.param.generator.rand() <= pitch: adj = self.perturb_isfeature(adj)
        # pitch adjustment: add|remove random variable
        if self.param.generator.rand() <= pitch: adj = self.perturb_randfeature(adj)

        if self.param.generator.rand() <= pitch: adj = self.change_distribution(adj)

        # pitch adjustment: add|remove bc variables
        if self.param.generator.rand() <= pitch: adj = self.perturb_bcfeature(adj, pitch)
        # Pitch adjustment: add|remove cor variables
        if self.param.generator.rand() <= pitch: adj = self.perturb_corfeature(adj)
        # Pitch adjustment: add|remove class param variables
        if self.param.generator.rand() <= pitch: adj = self.perturb_class_paramfeature(adj)
        # Pitch adjustment: add|remove member param variables
        if self.param.generator.rand() <= pitch: adj = self.perturb_member_paramfeature(adj)

        adj, converged = self.evaluate_solution(adj)
        return adj, converged
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Extracts the best features                       '''
    ''' ---------------------------------------------------------- '''
    def get_best_features(self, memory):
    # {
        soln = self.find_best_sol(memory)

        # Copy necessary values from soln dictionary
        best_asvars = soln['asvars'].copy()
        best_isvars = soln['isvars'].copy()
        best_randvars = soln['randvars'].copy()
        best_bcvars = soln['bcvars'].copy()
        best_corvars = soln['corvars'].copy()
        asc_ind = soln['asc_ind']
        best_class_params_spec = soln['class_params_spec'].copy() if soln['class_params_spec'] is not None else None
        best_member_params_spec = soln['member_params_spec'].copy() if soln['member_params_spec'] is not None else None

        # Return a tuple containing eight things
        return (best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind, best_class_params_spec,
                best_member_params_spec)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Used in local_search_1 to local_search_10        '''
    ''' ---------------------------------------------------------- '''
    def make_evaluate_insert(self, _asvars, _isvars, _randvars, _bcvars, _corvars, _asc_ind,
                             _class_params_spec, _member_params_spec):
    # {
        solution = Solution(nb_crit=self.nb_crit, asvars=_asvars, isvars=_isvars, randvars=_randvars, bcvars=_bcvars,
                    corvars=_corvars, asc_ind=_asc_ind, class_params_spec=_class_params_spec,
                    member_params_spec=_member_params_spec)

        revised_solution, converged = self.evaluate_solution(solution)
        if converged:
            self.insert_solution(revised_solution)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Apply local search to improve a solution         '''
    ''' ---------------------------------------------------------- '''
    # Check whether changing a coefficient distribution improves the kpi
    def make_change_1(self, candidate):
    # {
        # Identify the best solution features
        best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, \
            asc_ind, best_class_params_spec, best_member_params_spec = self.get_best_features(candidate)

        # Make changes
        solution = Solution(nb_crit=self.nb_crit, asvars=best_asvars, isvars=best_isvars, randvars=best_randvars, bcvars=best_bcvars,
                            corvars=best_corvars, asc_ind=asc_ind, class_params_spec=best_class_params_spec,
                            member_params_spec=best_member_params_spec)

        revised_solution = self.change_distribution(solution)  # Make a change

        # Revise the following dictionary and lists
        best_randvars = {key: val for key, val in best_randvars.items() if key in best_asvars and val != 'f'}
        best_bcvars = [var for var in best_bcvars if var in best_asvars and var not in self.param.ps_corvars]
        best_corvars = [var for var in best_randvars.keys() if var not in best_bcvars]

        # Make a solution, evaluate it, and insert if converged
        self.make_evaluate_insert(best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind,
                            best_class_params_spec, best_member_params_spec)
    # }

    # Check if having a full covariance matrix leads to an improved BIC
    def make_change_2(self, candidate):
    # {
        # Identify the best solution features
        best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, \
            asc_ind, best_class_params_spec, best_member_params_spec = self.get_best_features(candidate)

        # Make changes
        best_bcvars = [var for var in best_asvars if var in self.param.ps_bcvars]
        if self.param.ps_cor is None or self.param.ps_cor:
            best_corvars = [var for var in best_randvars if var not in best_bcvars]
        else:
            best_corvars.clear()

        # Make a solution, evaluate it, and insert if converged
        self.make_evaluate_insert(best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind,
                          best_class_params_spec, best_member_params_spec)
    # }

    # Check if having all the variables transformed leads to an improvement in BIC
    def make_change_3(self, candidate):
    # {
        # Identify the best solution features
        best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, \
            asc_ind, best_class_params_spec, best_member_params_spec = self.get_best_features(candidate)

        # Make changes
        if self.param.ps_bctrans is None or self.param.ps_bctrans:
            best_bcvars = [var for var in best_asvars if var not in self.param.ps_corvars]
        else:
            best_bcvars.clear()

        best_corvars = [var for var in best_randvars.keys() if var not in best_bcvars]

        # Make a solution, evaluate it, and insert if converged
        self.make_evaluate_insert(best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, asc_ind,
                          best_class_params_spec, best_member_params_spec)
    # }

    def make_change_4(self, candidate):
    # {
        # Identify the best solution features
        best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, \
            asc_ind, best_class_params_spec, best_member_params_spec = self.get_best_features(candidate)

        if len(best_asvars) < len(self.param.asvarnames):
        # {
            solution = Solution(nb_crit=self.nb_crit, asvars=best_asvars, isvars=best_isvars,
                                randvars=best_randvars, bcvars=best_bcvars,
                                corvars=best_corvars, asc_ind=asc_ind,
                                class_params_spec=best_class_params_spec,
                                member_params_spec=best_member_params_spec)

            solution = self.add_asfeature(solution)
            revised_solution, converged = self.evaluate_solution(solution)
            if converged: self.insert_solution(revised_solution)
        # }
    # }


    def make_change_5(self, candidate):
    # {
        # Identify the best solution features
        best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, \
            asc_ind, best_class_params_spec, best_member_params_spec = self.get_best_features(candidate)

        solution = Solution(nb_crit=self.nb_crit, asvars=best_asvars, isvars=best_isvars,
                            randvars=best_randvars, bcvars=best_bcvars, corvars=best_corvars, asc_ind=asc_ind,
                            class_params_spec=best_class_params_spec, member_params_spec=best_member_params_spec)

        solution = self.add_isfeature(solution)
        revised_solution, converged = self.evaluate_solution(solution)
        if converged: self.insert_solution(revised_solution)
    # }

    def make_change_6(self, candidate):
    # {
        # Identify the best solution features
        best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, \
            asc_ind, best_class_params_spec, best_member_params_spec = self.get_best_features(candidate)

        solution = Solution(nb_crit=self.nb_crit, asvars=best_asvars, isvars=best_isvars, randvars=best_randvars,
                            bcvars=best_bcvars, corvars=best_corvars, asc_ind=asc_ind,
                            class_params_spec=best_class_params_spec, member_params_spec=best_member_params_spec)

        if self.param.avail_bcvars: # {
            solution = self.perturb_bcfeature(solution, self.pitch)
            revised_solution, converged = self.evaluate_solution(solution)
            if converged: self.insert_solution(revised_solution)
        # }
    # }

    def make_change_7(self, candidate):
    # {
        # Identify the best solution features
        best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, \
            asc_ind, best_class_params_spec, best_member_params_spec = self.get_best_features(candidate)

        solution = Solution(nb_crit=self.nb_crit, asvars=best_asvars, isvars=best_isvars, randvars=best_randvars, bcvars=best_bcvars,
                            corvars=best_corvars, asc_ind=asc_ind, class_params_spec=best_class_params_spec,
                            member_params_spec=best_member_params_spec)

        if self.param.avail_rvars:
        # {
            solution = self.perturb_corfeature(solution)
            revised_solution, converged = self.evaluate_solution(solution)
            if converged:
                self.insert_solution(revised_solution)

            revised_solution = self.perturb_randfeature(solution)
            revised_solution, converged = self.evaluate_solution(revised_solution)
            if converged:
                self.insert_solution(revised_solution)
        # }
    # }

    # Check if changing coefficient distributions improves the solution BIC
    def make_change_8(self, candidate):
    # {
        # Identify the best solution features
        best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, \
            asc_ind, best_class_params_spec, best_member_params_spec = self.get_best_features(candidate)

        for var in best_randvars: # {
            if var not in self.param.ps_randvars:
                rm_dist = [distr for distr in self.param.distr if distr != best_randvars[var]]
                best_randvars[var] = self.param.generator.choice(rm_dist)
        # }
        best_randvars = {key: val for key, val in best_randvars.items() if key in best_asvars and val != 'f'}
        best_bcvars = [var for var in best_bcvars if var in best_asvars and var not in self.param.ps_corvars]

        if self.param.ps_cor is None or self.param.ps_cor:
            best_corvars = [var for var in best_randvars.keys() if var not in best_bcvars]
        else:
            best_corvars = []

        if len(best_corvars) < 2:
            best_corvars = []

        solution = Solution(nb_crit=self.nb_crit, asvars=best_asvars, isvars=best_isvars,
                            randvars=best_randvars, bcvars=best_bcvars, corvars=best_corvars,
                            asc_ind=asc_ind, class_params_spec=best_class_params_spec,
                            member_params_spec=best_member_params_spec)
        revised_solution, converged = self.evaluate_solution(solution)

        if converged:
            self.insert_solution(revised_solution)
    # }

    def make_change_9(self, candidate):
    # {
        # Identify the best solution features
        best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, \
            asc_ind, best_class_params_spec, best_member_params_spec = self.get_best_features(candidate)


        solution = Solution(nb_crit=self.nb_crit, asvars=best_asvars, isvars=best_isvars,
                randvars=best_randvars, bcvars=best_bcvars, corvars=best_corvars, asc_ind=asc_ind,
                class_params_spec=best_class_params_spec, member_params_spec=best_member_params_spec)

        if solution['class_params_spec'] is not None:
        # {
            revised_solution = self.perturb_class_paramfeature(solution)
            revised_solution, converged = self.evaluate_solution(revised_solution)
            if converged:
                self.insert_solution(revised_solution)
        # }
    # }

    def make_change_10(self, candidate):
    # {
        # Identify the best solution features
        best_asvars, best_isvars, best_randvars, best_bcvars, best_corvars, \
            asc_ind, best_class_params_spec, best_member_params_spec = self.get_best_features(candidate)

        solution = Solution(nb_crit=self.nb_crit, asvars=best_asvars, isvars=best_isvars,
                    randvars=best_randvars, bcvars=best_bcvars, corvars=best_corvars, asc_ind=asc_ind,
                    class_params_spec=best_class_params_spec, member_params_spec=best_member_params_spec)

        if solution['member_params_spec'] is not None:
        # {
            revised_solution = self.perturb_member_paramfeature(solution)
            revised_solution, converged = self.evaluate_solution(revised_solution)
            if converged: self.insert_solution(revised_solution)
        # }
    # }

    def local_search(self):
    # {
        # Identify candidate solutions
        candidate = [sol for sol in self.memory if abs(sol.obj[0]) < BOUND]

        # TODO
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def log_convergence(self, memory):
    # {
        crit = self.param.criterions[:self.nb_crit]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Filter out poor solutions:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        filtered_memory = [sol for sol in memory if abs(sol[crit[0]]) < BOUND and abs(sol[crit[1]]) < BOUND]
        # OR,
        # filtered_memory = [sol for sol in memory if abs(sol.obj[0]) < BOUND and abs(sol.obj[1]) < BOUND]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sort the new list of solutions by 'sol_num':
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        filtered_memory = sorted(filtered_memory, key=lambda sol: sol['sol_num'])

        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        # Record the best obj val
        # ~~~~~~~~~~~~~~~~~~~~~~~~~
        best_val = self.get_best_val(self.param.criterions, filtered_memory)
        all_val = self.get_all_val(self.param.criterions, filtered_memory)
        for i in range(self.nb_crit):
            logger.debug(f"Best points (obj {i}): {best_val[i]}") # Log the ith value

        return all_val, best_val
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function  Conduct harmony memory consideration             '''
    ''' pitch adjustment, and local search                         '''
    ''' This function tracks the progress of the optimization      '''
    ''' process by recording the score of the best and current     '''
    ''' solutions at each iteration                                '''
    ''' ---------------------------------------------------------- '''
    def improvise(self):
    # {
        best, current = [], []
        for iter in range(self.maxiter):
        # {
            # Compute consideration rate and pitch value
            # This code introduces oscillations (a.k.a., variations) based on the iteration number.
            # The result is scaled by the sine function only when its value is non-negative.
            sine_iter = max(0, np.sign(math.sin(iter)))
            self.harm_rate = (self.min_harm + ((self.max_harm - self.min_harm) / self.maxiter) * iter) * sine_iter
            self.pitch = (self.min_pitch + ((self.max_pitch - self.min_pitch) / self.maxiter) * iter) * sine_iter

            new_sol = self.build_solution(self.memory, self.harm_rate) # Create a single new solution and perform an adjustment
            curr_sol, converged = self.pitch_adjustment(new_sol, self.pitch)   # Perform additional perturbations
            if converged:
            # {
                self.insert_solution(curr_sol)

                #if iter > int(self.prop_local * self.maxiter):
                # {
                    # Run local search
                    #best_sol = self.memory[0]
                    #best.append(best_sol.obj[0])
                    #current.append(curr_sol.obj[0])
                # }
            # }
        # }

        all_val, obj_val = self.log_convergence(self.memory)
        if self.generate_plots:
            self.plot_results(self.memory, all_val, obj_val)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.  Discard non-convergent solutions                '''
    ''' ---------------------------------------------------------- '''
    def screen_solutions(self, solutions):
    # {
        feasible_solutions = []
        if solutions is not None:
        # {
            for sol in solutions:
            # {
                new_sol = copy.deepcopy(sol)
                new_sol = self.increase_sol_by_one_class(new_sol)
                new_sol.pop('class_num')    # Remove 'class_num'
                new_sol, converged = self.evaluate_solution(new_sol)
                if converged:
                    feasible_solutions.append(new_sol)
            # }
        # }
        return feasible_solutions
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function. Code used in "self.run_search"                   '''
    ''' ---------------------------------------------------------- '''
    def extract_parameter(self):
    # {
        avail, avail_latent = self.param.avail, self.param.avail_latent
        weights = self.param.weights
        alt_var = self.param.alt_var
        choice_id = self.param.choice_id
        ind_id = self.param.ind_id

        if self.nb_crit > 1:
        # {
            if self.param.avail is not None:
                avail = np.row_stack((self.param.avail, self.param.test_avail))

            if self.param.avail_latent is not None:
            # {
                avail_latent = make_list(None, self.param.num_classes)  # i.e., [None] * self.param.num_classes
                for ii, avail_latent_ii in enumerate(self.param.avail_latent):
                    # {
                    if avail_latent_ii is not None:
                        avail_latent[ii] = np.row_stack((avail_latent_ii, self.param.test_avail_latent[ii]))
                # }
            # }

            if self.param.weights is not None:
                weights = np.concatenate((self.param.weights, self.param.test_weight_var))

            if self.param.alt_var is not None:
                alt_var = np.concatenate((self.param.alt_var, self.param.test_alt_var))

            if self.param.choice_id is not None:
                choice_id = np.concatenate((self.param.choice_id, self.param.test_choice_id))

            if self.param.ind_id is not None:
                ind_id = np.concatenate((self.param.ind_id, self.param.test_ind_id))
        # }
        return  avail, weights, alt_var, choice_id, ind_id, avail_latent
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def extract_from_sol(self, sol):
    # {
        asvarnames, isvarnames, randvars, bcvars, corvars, intercept, \
            class_params_spec, member_params_spec = self.get_components(sol)

        # Revise varnames
        if self.param.latent_class:
            varnames = np.concatenate(class_params_spec + member_params_spec + [isvarnames])
            varnames = np.unique(varnames)
        else:
            varnames = asvarnames + isvarnames

        # Delete '_inter' bug fix
        if '_inter' in varnames:
            varnames = np.delete(varnames, np.argwhere(varnames == '_inter'))

        return varnames, asvarnames, isvarnames, randvars, bcvars, corvars, \
            intercept, class_params_spec, member_params_spec
    # }

    # call if self.param.multi_objective?
    def test_best_solution(self, best_sol):
    # {
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sort memory and extract features
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        best_varnames, best_asvarnames, best_isvarnames, best_randvars, best_bcvars, best_corvars, \
            best_intercept, best_class_params_spec, best_member_params_spec = self.extract_from_sol(best_sol)

        avail_all, weights_all, alt_var_all, choice_id_all, ind_id_all, avail_latent_all = self.extract_parameter()

        df_all = pd.concat([self.param.df, self.param.df_test], ignore_index=True)
        y = self.param.choices + self.param.test_choices
        X = df_all[best_varnames]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Define appropriate model and fit coefficients
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if bool(best_randvars):
        # {
            if self.param.latent_class:
                """
                Note template: search::fit_lcmm(X, y, varnames, isvars, class_params_spec, member_params_spec, num_classes,
                    alts, ids, panels, bcvars, randvars, corvars, maxiter, gtol, avail, weights)
                """
                model = self.fit_lcmm(X, y, best_varnames, best_isvarnames, best_class_params_spec,
                    best_member_params_spec, self.param.num_classes, alt_var_all, choice_id_all,
                    ind_id_all, best_bcvars, best_randvars, best_corvars,
                    self.param.maxiter, self.param.gtol, avail_all, weights_all)
            else:
                """
                Note template: search::fit_mxl(X, y, varnames, alts, isvars, transvars, ids, panels, randvars, corvars,
                    fit_intercept, n_draws, weights, avail, base_alt,  maxiter, seed, ftol, gtol, save_fitted_params)
                """
                model = self.fit_mxl(X, y, best_varnames, alt_var_all, best_isvarnames, best_bcvars,
                    choice_id_all, ind_id_all, best_randvars, best_corvars, best_intercept,
                    self.param.n_draws, None, None, None, 2000, None, 1e-6, 1e-6, False)
            # }
        # }
        else:
        # {
            if self.param.latent_class:
                """ 
                Note template: search::fit_lcm(X, y, varnames, class_params_spec, member_params_spec, num_classes, ids,
                transvars, maxiter, gtol, gtol_membership_func, avail, avail_latent, intercept_opts, weights, seed,
                alts, ftol_lccm, base_alt)
                """
                seed = self.param.generator.randint(2 ** 31 - 1)
                model = self.fit_lcm(X, y, best_varnames, best_class_params_spec, best_member_params_spec,
                    self.param.num_classes, choice_id_all, best_bcvars, self.param.maxiter,
                    self.param.gtol, self.param.gtol_membership_func, avail_all, avail_latent_all,
                    self.param.intercept_opts, weights_all, seed, None, 1e-6, None)

            else:
                """
                Note template: search::fit_mnl(X, y, varnames, isvars, alts, ids, transvars, fit_intercept,
                    weights, avail, base_alt, maxiter, ftol, gtol, seed)
                """
                model = self.fit_mnl(X, y, best_varnames, best_isvarnames, alt_var_all,
                    choice_id_all, best_bcvars, best_intercept, None, None, None, 2000, 1e-6, 1e-6, None)
        # }

        report_model_statistics(self.results_file, model)  # Output the model statistics
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def log_solutions(self, solutions):
    # {
        if self.nb_crit == 1:
        # {
            all_solutions = sorted(solutions, key=lambda sol: sol.obj[0])
            best_sols = all_solutions[:self.max_mem]
            best_sol = best_sols[0]
            logger.info("Model with best score had {} classes".format(self.max_classes))
            logger.info("Best solution")
            for k, v in best_sol.items():
                logger.info(f"{k}: {v}")
        # }
        else:
        # {
            fronts = self.get_fronts(solutions)
            pareto = self.get_pareto(fronts, solutions)
            all_solutions = self.non_dominant_sorting(solutions)
            logger.info("Models in Pareto front had at most {} classes".format(self.max_classes))
            logger.info("Best models in Pareto front")
            for i, sol in enumerate(pareto):
            # {
                logger.info(f'Best solution - {i}')
                for k, v in sol.items():
                    logger.info(f"{k}: {v}")
            # }
            best_sol = all_solutions[0]
            logger.info(f"Best solution with {best_sol['class_num']} classes")
            for k, v in best_sol.items():
                logger.info(f"{k}: {v}")
        # }
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def run_search(self, existing_sols=None):
    # {
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Combine solutions into one list
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        existing_memory = self.screen_solutions(existing_sols)  # Screen out non convergent solutions
        generated_memory = self.initialize_memory(self.max_mem)  # Generate some solutions

        # OPTIONAL: CREATE max(max_mem - existing, 0) NEW SOLUTIONS?

        init_memory = generated_memory + existing_memory    # Aggregate solution lists
        unique_memory = get_unique(init_memory, 0)  # Remove duplicate solutions if present
        for sol in unique_memory:
            sol.data['is_initial_sol'] = True  # Set solution status

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sort memory by first objective or by Pareto ranking and retain specified number
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        memory_sorted = self.sort_memory(unique_memory)
        memory = memory_sorted[:self.max_mem]
        self.memory = memory.copy()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate new solutions by combining elements from existing ones
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.improvise()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Copy, Sort & Test
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        memory = self.memory.copy()
        improved_memory = self.sort_memory(memory)
        best_sol = improved_memory[0]
        self.test_best_solution(best_sol)

        # ~~~~~~~~~~~~~~~~
        # Perform logging
        # ~~~~~~~~~~~~~~~~
        logger.info("Improved harmony: {}".format(improved_memory))
        logger.info("Search ended at: {}".format(str(time.ctime())))

        return improved_memory
    # }


    def run_search_latent(self, override=False):
    # {
        prev, best_model_idx = infinity, 0
        all_solutions, solutions = [], []

        for q in range(self.min_classes, self.max_classes + 1):
        # {
            self.param.num_classes = q
            self.param.latent_class = False if q == 1 else True
            solutions = self.run_search(existing_sols=all_solutions)

            # This code iterates over each dictionary sol in the solutions list and updates
            # the value associated with the key 'class_num' to q. The use of a list
            # comprehension avoids the need for an explicit loop.
            [sol.update({'class_num': q}) for sol in solutions]

            # Aggregate solutions
            all_solutions = all_solutions + solutions

            if self.param.nb_crit > 1:
            # {
                all_solutions = self.non_dominant_sorting(all_solutions)
                fronts = self.get_fronts(all_solutions)
                pareto = self.get_pareto(fronts, all_solutions)
                self.pareto_front = pareto

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Check if a solution with q classes is in the Pareto front
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                pareto_class_nums = [sol['class_num'] for sol in pareto]
                stop_run = max(pareto_class_nums) != q
                if stop_run and not override:
                    logger.info(f"Stopping search at {q} classes")
                    break  # Exit the loop immediately

                best_model_idx += 1
            # }
            else:
            # {
                all_solutions = sorted(solutions, key=lambda sol: sol.obj[0])
                best_solution = all_solutions[0]  # assume already sorted
                if best_solution.obj[0] < prev or override:
                # {
                    best_model_idx += 1
                    prev = best_solution.obj[0]
                # }
                else:  # {
                    break  # Exit the loop immediately
                # }
            # }
        # }
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.log_solutions(all_solutions)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        best_val, all_val, all_val_classes = self.post_process(all_solutions)

        if self.generate_plots:
            self.plot_results_latent(all_solutions, best_val, all_val, all_val_classes)

        return all_solutions
    # }


    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def plot_multi(self, solutions, all_val):
    # {
        crit = self.param.criterions[:self.nb_crit]
        fig, ax = plt.subplots()

        # ~~~~~~~~~~~
        # LINE 1
        # ~~~~~~~~~~~
        scaled_val = np.log(all_val[1]) if crit[1] == 'MAE' else all_val[1]
        line_1 = ax.scatter(all_val[0], scaled_val, label="All solutions", marker='o')

        # ~~~~~~~~~~~
        # LINE 2
        # ~~~~~~~~~~~
        init_solns = [sol for sol in solutions if abs(sol.obj[0]) < BOUND]
        init_0 = [sol[crit[0]] for sol in init_solns]
        init_1 = [sol[crit[1]] for sol in init_solns]
        line_2 = ax.scatter(init_0, init_1, label="Initial solutions", marker='x')

        # ~~~~~~~~~~~~~~~~~~~~
        # PARETO CALCULATIONS
        # ~~~~~~~~~~~~~~~~~~~~
        fronts = self.get_fronts(solutions)
        pareto = self.get_pareto(fronts, solutions)
        self.pareto_front = [sol for sol in pareto if abs(sol.obj[0]) < BOUND]  # Store filtered
        pareto_0 = np.array([sol.obj[0] for sol in pareto])
        pareto_1 = np.array([sol.obj[1] for sol in pareto])
        pareto_1 = np.log(pareto_1) if crit[1] == 'MAE' else pareto_1

        # ~~~~~~~~~~~
        # LINE 3
        # ~~~~~~~~~~~
        line_3 = ax.scatter(pareto_0, pareto_1, label="Pareto Front", marker='o')

        # ~~~~~~~~~~~
        # LINE 4
        # ~~~~~~~~~~~
        # Determine the indices that would sort the pareto_0 array in ascending order.
        # These indices represent the positions of elements in the original array.
        pareto_idx = np.argsort(pareto_0)
        line_4 = ax.plot(pareto_0[pareto_idx], pareto_1[pareto_idx], color="r", label="Pareto Front")

        # ~~~~~~~~~~~
        # ALL LINES
        # ~~~~~~~~~~~
        all_lines = (line_1, line_2, line_4[0])

        labels = [line.get_label() for line in all_lines]
        log_str = 'log' if crit[1] == 'MAE' else ''
        ax.set_xlabel(f"{crit[0]} - Training dataset")
        ax.set_ylabel(f"{crit[1]} - Testing dataset")
        lgd = ax.legend(all_lines, labels, loc='upper right', bbox_to_anchor=(0.5, -0.1))
        current_time = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
        latent_info = "_" + str(self.param.num_classes) + "_classes_" if (self.param.num_classes > 1) else "_"
        plot_filename = self.code_name + "_" + latent_info + current_time + "_MOOF.png"
        plt.savefig(plot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    # }

    def plot_multi_latent(self, solutions, all_val_classes):
    #
        crit = self.param.criterions[:self.nb_crit]

        fronts = self.get_fronts(solutions)
        pareto = self.get_pareto(fronts, solutions)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        fig, ax = plt.subplots()
        lns_all = []
        for q in range(self.min_classes,self.max_classes):
        # {
            class_label = "All solutions - " + str(q) + " classes"
            if q == 1:
                class_label = "All solutions - " + str(q) + " class"
            lns = ax.scatter(all_val_classes[0][q], all_val_classes[1][q], label=class_label, marker='o')
            lns_all.append(lns)
        # }
        # ~~~~~~~~~~~
        # LINE 2
        # ~~~~~~~~~~~
        init_val = [[] for _ in range(self.nb_crit)]
        init_sols = [sol for sol in solutions if sol.obj[0] < BOUND and sol['is_initial_sol']]
        init_val[0] = [sol.obj[0] for sol in init_sols]
        if crit[0] == 'MAE': init_val[1] = np.log(init_val[1])
        init_val[1] = [sol.obj[1] for sol in init_sols]
        if crit[1] == 'MAE': init_val[1] = np.log(init_val[1])
        lns2 = ax.scatter(init_val[0], init_val[1], label="Initial solutions", marker='x', color='black')

        # ~~~~~~~~~~~
        # LINE 4
        # ~~~~~~~~~~~
        pareto = [pareto for _, pareto in enumerate(pareto) if np.abs(pareto.obj[0]) < BOUND]
        logger.info('Final Pareto: {}'.format(str(pareto)))
        pareto_0 = np.array([par.obj[0] for par in pareto])
        pareto_1 = np.array([par.obj[1] for par in pareto])
        log_str = ''
        if crit[1] == 'MAE':
            pareto_1 = np.log(pareto_1)
            log_str = 'log'

        pareto_idx = np.argsort(pareto_0)
        lns4 = ax.plot(pareto_0[pareto_idx], pareto_1[pareto_idx], color="r", label="Pareto Front")

        # ~~~~~~~~~~~
        # ALL LINES
        # ~~~~~~~~~~~
        lns = (*lns_all, lns2, lns4[0])

        labs = [l_pot.get_label() for l_pot in lns]
        ax.set_xlabel(f"{crit[0]} - Training dataset")
        ax.set_ylabel(f"{log_str} {crit[1]} - Testing dataset")
        lgd = ax.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        current_time = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
        plot_filename = self.code_name + "_" + current_time + "_MOOF.png"
        plt.savefig(plot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def plot_single(self, all_val, best_val):
    # {
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.xaxis.get_major_locator().set_params(integer=True)

        crit = self.param.criterions[:self.nb_crit]
        label ="Solution estimated at current iteration (" + crit + ")"
        line_1 = ax1.plot(np.arange(len(all_val[0])), all_val[0], label=label)

        label = "Best solution in memory at current iteration (" + crit + ")"
        line_2 = ax1.plot(np.arange(len(best_val[0])), best_val[0], label=label, linestyle="dotted")

        label = "In-sample LL of best solution in memory at current iteration"
        line_3 = ax2.plot(np.arange(len(best_val[1])), best_val[1], label=label, linestyle="dashed")

        all_lines = line_1 + line_2 + line_3

        labels = [line.get_label() for line in all_lines]
        handles, _ = ax1.get_legend_handles_labels()
        lgd = ax1.legend(all_lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel(crit[0])
        ax2.set_ylabel(crit[1])
        current_time = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
        latent_info = "_" + str(self.param.num_classes) + "_classes_" if (self.param.num_classes > 1) else "_"
        plot_filename = self.code_name + "_" + latent_info + current_time + "_SOOF.png"
        plt.savefig(plot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    # }

    def plot_single_latent(self, best_val, all_val, all_val_classes):
    # {
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.xaxis.get_major_locator().set_params(integer=True)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        counter = 0
        max_0 = np.max(all_val[0])
        for q in range(self.min_classes, self.max_classes):
        # {
            num_sols_in_class = len(all_val_classes[0][q])
            ax1.axvline(x=counter, color='r', linestyle='--')
            if q == 1: line_text = '1 class'
            else:  line_text = str(q) + ' classes'
            ax1.text(counter, max_0, line_text)
            counter += num_sols_in_class
        # }
        all_val[0] = np.concatenate(np.array(all_val_classes[0]))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        crit = self.param.criterions[:self.nb_crit]
        label ="Solution estimated at current iteration (" + crit + ")"
        line_1 = ax1.plot(np.arange(len(all_val[0])), all_val[0], label=label)

        label = "Best solution in memory at current iteration (" + crit + ")"
        line_2 = ax1.plot(np.arange(len(best_val[0])), best_val[0], label=label, linestyle="dotted")

        label = "In-sample LL of best solution in memory at current iteration"
        line_3 = ax2.plot(np.arange(len(best_val[1])), best_val[1], label=label, linestyle="dashed")

        all_lines = line_1 + line_2 + line_3

        labels = [line.get_label() for line in all_lines]
        handles, _ = ax1.get_legend_handles_labels()
        lgd = ax1.legend(all_lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel(crit[0])
        ax2.set_ylabel(crit[1])
        current_time = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
        latent_info = "_" + str(self.param.num_classes) + "_classes_" if (self.param.num_classes > 1) else "_"
        plot_filename = self.code_name + "_" + latent_info + current_time + "_SOOF.png"
        plt.savefig(plot_filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def post_process(self, solutions):
    # {
        valid_solutions = [sol for sol in solutions if sol.obj[0] < BOUND]
        valid_solutions = sorted(valid_solutions, key=lambda sol: sol['sol_num'])

        all_val_classes, all_val = [[] for _ in range(self.nb_crit)]
        for q in range(self.min_classes, self.max_classes + 1):
        # {
            for i in range(self.nb_crit):
            # {
                all_val[i] = [sol.obj[i] for sol in valid_solutions if sol['class_num'] == q]
                crit = self.param.crit(i)
                if crit == 'MAE': all_val[i] = np.log(all_val[i])
                all_val_classes[i].append(all_val[i])
            # }
        # }

        best_val = [[] for _ in range(self.nb_crit)]
        for i in range(self.nb_crit):
            best_val[i] = self.get_best_val(self.param.criterions, all_val_classes[i])

        return best_val, all_val, all_val_classes
    # }
    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def plot_results(self, solutions, best_val, all_val):
    # {
        if self.nb_crit == 1:
            self.plot_single(all_val[0], best_val)
        else:
            self.plot_multi(solutions, all_val)
    # }

    def plot_results_latent(self, solutions, best_val, all_val, all_val_classes):
    # {
        if self.nb_crit == 1:
            self.plot_single_latent(best_val, all_val, all_val_classes)
        else:
            self.plot_multi_latent(solutions, all_val_classes)
    # }
# }


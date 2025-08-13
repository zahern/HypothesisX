# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# META HEURISTIC OPTIMISATION APPROACH
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

try:
    from harmony import*
    from siman import*
    from threshold import*
except ImportError:
    from .harmony import*
    from .siman import*
    from .threshold import*

def call_harmony(parameters, init_sol=None):
# {
    solver = HarmonySearch(parameters)
    #todo inject init_sol
    solver.max_mem = 25
    solver.maxiter = 5000
    solver.run()
# }

def call_siman(parameters, init_sol=None,  **kwargs):
    '''
    Runs the Simulated Annealing (SA) algorithm with provided parameters and initial solution.

    Args:
        parameters (dict): The parameters required for the SA algorithm. Should include:
            - 'objective': The objective function to optimize.
            - 'constraints': Any constraints for the optimization problem.
            - (Other domain-specific parameters can be added.)
        init_sol (optional): The initial solution for the algorithm. Defaults to None.
        **kwargs: Additional optional arguments. These can include:
            - ctrl (tuple): Control parameters for SA as (tI, tF, max_temp_steps, max_iter):
                - tI: Initial temperature.
                - tF: Final temperature.
                - max_temp_steps: Maximum number of temperature steps.
                - max_iter: Maximum iterations per temperature step.
            - id_num (optional): An identifier for the current run.
    '''
    # Step 2: Extract control parameters or use defaults
    ctrl = kwargs.get('ctrl',
                      (1000, 0.001, 250, 10))  # Default: (Initial Temp, Final Temp, Max Temp Steps, Max Iterations)
    print(f"[INFO] Control parameters (ctrl): {ctrl}")
    print(
        "\n[INFO] To change the control parameters, pass the 'ctrl' argument as a tuple:\n"
        "  Example: ctrl=(200, 0.01, 10, 50)\n"
        "  - tI: Initial temperature (higher = more exploration).\n"
        "  - tF: Final temperature (lower = more exploitation).\n"
        "  - max_temp_steps: Number of temperature steps.\n"
        "  - max_iter: Number of iterations per temperature step.\n"
    )

    if 'ctrl' in kwargs:
        # Need to delete the 'ctrl' key from kwargs
        # This is because the function has a parameter named 'ctrl'
        # and the 'ctrl' key in kwargs would be a duplicate parameter
        del kwargs['ctrl']
    # ctrl = (1000, 0.001, 20, 20)  # i.e. (tI,tF,max_temp_steps,max_iter)
    id_num = kwargs.get('id_num', None)
    solver = SA(parameters, init_sol, ctrl, id_num, **kwargs)
    solver.run()
    solver.close_files()
    return solver.return_best()
# }

def call_parsa(parameters, init_sol=None, nthrds=4, **kwargs):
# {
   # ctrl = (10, 0.001, 10, 10)  # i.e. (tI, tF, max_temp_steps, max_iter)


    ctrl = kwargs.get('ctrl',(10, 0.001, 10, 10))

    if 'ctrl' in kwargs:
        # Need to delete the 'ctrl' key from kwargs
        # This is because the function has a parameter named 'ctrl'
        # and the 'ctrl' key in kwargs would be a duplicate parameter
        del kwargs['ctrl']
    parsa = PARSA(parameters, init_sol, ctrl, nthrds=nthrds)
    parsa.run()
# }

def call_parcopsa(parameters, init_sol=None, nthrds=8):
# {
    ctrl = (10, 0.001, 10, 10)  # i.e. (tI, tF, max_temp_steps, max_iter)
    parcopsa = PARCOPSA(parameters, init_sol, ctrl, nthrds=nthrds)

    # Optional. Set a different behaviour for each solver
    #tI = [1, 10, 100, 1000, np.random.randint(1, 10000), np.random.randint(1, 10000),
    #np.random.randint(1, 10000), np.random.randint(1, 10000)]
    #for i in range(8):
    #    parcopsa.solvers[i].revise_tI(tI[i])

    parcopsa.run()
# }

def call_threshold(parameters, init_sol=None, hm=False):
# {
    ctrl = (10, 20, 20) # i.e., threshold, max_steps, max_iter
    #ctrl = (10, 10, 1)  # i.e., threshold, max_steps, max_iter
    solver = TA(parameters, init_sol, ctrl)
    solver.run()
    solver.close_files()
# }

def covering_arrays(index = 0):
        # Define parameter ranges
    tI_values = [500, 1000, 1500]
    tF_values = [0.001, 0.01, 0.1]
    max_temp_steps_values = [10, 20, 30]
    max_iter_values = [10, 20, 50]

    # Generate a full factorial design for illustration (use a library for pairwise if needed)
    import itertools
    all_combinations = list(itertools.product(tI_values, tF_values, max_temp_steps_values, max_iter_values))

    # If you want pairwise, you may need a library like `allpairspy` or a manual covering array generator
    # Example of a manually reduced covering array for simplicity:
    covering_array = [
        (500, 0.001, 10, 10),
        (500, 0.01, 20, 20),
        (500, 0.1, 30, 50),
        (1000, 0.001, 20, 50),
        (1000, 0.01, 30, 10),
        (1500, 0.001, 30, 20),
        (1500, 0.1, 10, 50),
        (1500, 0.01, 20, 10),
    ]
    print("Covering Array:")
    for row in covering_array:
        print(row)
    if index < len(covering_array):
        return covering_array[index]


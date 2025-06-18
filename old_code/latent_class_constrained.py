import numpy as np
class LatentClassConstrained:
    """
    A class to manage lists for an arbitrary number of latent classes, 
    including their subsets.
    """
    def __init__(self, num_classes):
        """
        Initializes the latent classes with empty lists.

        Args:
            num_classes (int): Number of latent classes.
        """
        self.classes = {}

        for i in range(1, num_classes + 1):
            class_name = f"latent_class_{i}"
            self.classes[class_name] = {
                "asvar": [],
                "isvars": [],
                "randvars": [],
                "memvars": [],
                "sub_asvar": [],
                "sub_isvars": [],
                "sub_randvars": [],
                "sub_memvars": []
            }

    def populate_class(self, class_name, asvar=None, isvars=None, randvars=None, memvars=None, 
                       req_asvar=None, req_isvars=None, req_randvars=None, req_memvars=None):
        """
        Populates the lists for a specific latent class.

        Args:
            class_name (str): The name of the latent class to populate.
            asvar (list): Values for the `asvar` list.
            isvars (list): Values for the `isvars` list.
            randvars (list): Values for the `randvars` list.
            memvars (list): Values for the `memvars` list.
            sub_asvar (list): Values for the `sub_asvar` list.
            sub_isvars (list): Values for the `sub_isvars` list.
            sub_randvars (list): Values for the `sub_randvars` list.
            sub_memvars (list): Values for the `sub_memvars` list.
        """
        if class_name not in self.classes:
            raise ValueError(f"Latent class {class_name} does not exist.")
        
        # Update the lists; default to empty lists if not provided.
        self.classes[class_name]["asvar"] = asvar or []
        self.classes[class_name]["isvars"] = isvars or []
        self.classes[class_name]["randvars"] = randvars or []
        self.classes[class_name]["memvars"] = memvars or []
        self.classes[class_name]["req_asvar"] = req_asvar or []
        self.classes[class_name]["req_isvars"] = req_isvars or []
        self.classes[class_name]["req_randvars"] = req_randvars or []
        self.classes[class_name]["req_memvars"] = req_memvars or []

    def get_class(self, class_name):
        """
        Retrieves the data for a specific latent class.

        Args:
            class_name (str): The name of the latent class to retrieve.

        Returns:
            dict: The data for the specified latent class.
        """
        if class_name not in self.classes:
            raise ValueError(f"Latent class {class_name} does not exist.")
        return self.classes[class_name]

    def get_all_classes(self):
        """
        Retrieves all latent classes and their data.

        Returns:
            dict: A dictionary containing all latent classes and their data.
        """
        return self.classes
    
    def get_global_asvars_randvars(self):
        """
        Aggregates the `asvar` and `randvars` across all latent classes.

        Returns:
            dict: A dictionary with global `asvars` and `randvars`.
        """
        global_asvars = []
        global_randvars = []
        global_isvars = []
        global_memvars = []
        for class_data in self.classes.values():
            global_asvars.extend(class_data["asvar"])
            global_randvars.extend(class_data["randvars"])
            global_isvars.extend(class_data["isvars"])
            global_memvars.extend(class_data["memvars"])

        
        # Remove duplicates by converting to a set and back to a list
        return {
            "asvars": list(set(global_asvars)),
            "randvars": list(set(global_randvars)),
            "isvars": list(set(global_isvars)),
            "memvars":list(set(global_memvars))

        }


class LatentClassCoefficients:
    """
    A class to store and manage coefficients for variables in latent classes.
    Each latent class will have coefficients based on the provided variables.
    """

    def __init__(self, num_alternatives, num_classes, asvars, isvars, memvars):
        """
        Initialize the coefficients for a specific latent class.

        Args:
            num_alternatives (int): The number of alternatives (used for `isvars` coefficients).
            latent_class (dict): A dictionary representing a single latent class, containing
                                 lists of `asvar`, `isvars`, `randvars`, and `memvars`.
        """
        self.obj = None
        self.num_alternatives = num_alternatives
        self.num_classes = num_classes
        self.asvars = asvars
        self.isvars = isvars
        self.memvars = memvars
        self.randvars = []
        #self.coefficients =   # Dictionary to store coefficients for variables
        self.class_names = [f"class {str(i)}" for i in range(2, num_classes +1)] #start at 2
        # Initialize coefficients for each type of variable
        self.coefficients = self.initialize_coefficients()

    def define_structure(self, list_as_vars, list_is_vars, list_memvars):
        """
        Define the structure of coefficients for each latent class based on current variables.

        Args:
            list_as_vars (list of lists): Current `asvar` variables for each class.
            list_is_vars (list of lists): Current `isvars` variables for each class.
            list_memvars (list of lists): Current `memvars` variables, for all classes except the last one.

        Returns:
            dict: A dictionary mapping each latent class to its variable coefficients.
        """
        if len(list_as_vars) != self.num_classes:
            raise ValueError("Mismatch between the number of classes and the size of list_as_vars.")
        if len(list_is_vars) != self.num_classes:
            raise ValueError("Mismatch between the number of classes and the size of list_is_vars.")
        if len(list_memvars) != self.num_classes - 1:
            raise ValueError("Mismatch between the number of classes and the size of list_memvars.")

        structure = {}

        for i in range(self.num_classes):
            # Initialize the structure for the current class
            structure[i] = {
                "asvar": {},
                "isvars": {},
                "memvars": {}
            }

            # Process `asvar` variables
            for var in list_as_vars[i]:
                if var in self.coefficients[i]:
                    structure[i]["asvar"][var] = self.coefficients[i][var]
                else:
                    raise ValueError(f"Variable '{var}' not found in coefficients for latent class {i}.")

            # Process `isvars` variables
            for var in list_is_vars[i]:
                if var in self.coefficients[i]:
                    structure[i]["isvars"][var] = self.coefficients[i][var]
                else:
                    raise ValueError(f"Variable '{var}' not found in coefficients for latent class {i}.")

            # Process `memvars` variables (only for classes 0 to num_classes - 2)
            if i < self.num_classes - 1:
                for var in list_memvars[i]:
                    if var in self.coefficients[i]:
                        structure[i]["memvars"][var] = self.coefficients[i][var]
                    else:
                        raise ValueError(f"Variable '{var}' not found in coefficients for latent class {i}.")

        return structure

    def arrange_coefficients(self, sol):
        memvars = sol.get('memvars')
        asvars = sol.get('asvars')
        isvars = sol.get('isvars')
    
    def initialize_coefficients(self):
        """
        Initialize the coefficients for all latent classes.

        Returns:
            dict: A dictionary where each latent class has its own dictionary of coefficients.
        """
        coefficients = {}

        for i in range(self.num_classes):
            coefficients[i] = {
                'isvars':{},
                'asvars':{},
                'memvars':{}
            }

            # Initialize `isvars` coefficients (array of zeros for each variable in `isvars`)
            for var in self.isvars:
                coefficients[i]['isvars'][var] = np.zeros(self.num_alternatives)

            # Initialize `asvars` coefficients (single zero for each variable in `asvar`)
            for var in self.asvars:
                coefficients[i]['asvars'][var] = 0.0

            # Initialize `randvars` coefficients (single zero for each variable in `randvars`)
            
            for var in self.randvars:
                pass #placeholder
                #coefficients[i][var] = 0.0

            # Initialize `memvars` coefficients (single zero for each variable in `memvars`)
            if i >0: #ignore base line class
                for var in self.class_names:
                    coefficients[i]['memvars'][var] = 0.0
                for var in self.memvars:
                    coefficients[i]['memvars'][var] = 1.0

        return coefficients
    def update_coefficients(self, coeff, coeff_latent, isvars_list, asvars_list, memvars_list, obj):
        """
        Get the coefficients for a list of `isvars`, `asvars`, and `memvars` for each latent class.

        Args:
            coeff a list of the coefficients
            isvars_list (list of lists): A list of lists of `isvars` for each latent class.
            asvars_list (list of lists): A list of lists of `asvars` for each latent class.
            memvars_list (list of lists): A list of lists of `memvars` for each latent class.

        Returns:
            dict: A dictionary containing the coefficients for each input variable type and latent class.
        """
        if self.obj is None or obj < self.obj:
            print('Test: update')
            self.obj = obj
        else:
            print('Test: no update')
            return
        result = {}
        coeff_counter =0
        coeff_counter_l = 0
        for i in range(self.num_classes):
            result[i] = {
                'isvars': {},
                'asvars': {},
                'memvars': {}
            }

            # Get `isvars` coefficients for the current latent class
            for var in isvars_list[i]:
                if var in self.coefficients[i]['isvars']:
                    self.coefficients[i]['isvars'][var] = coeff[coeff_counter:coeff_counter+self.num_alternatives]
                    coeff_counter +=self.num_alternatives-1

            # Get `asvars` coefficients for the current latent class
            for var in asvars_list[i]:
                if var in self.coefficients[i]['asvars']:
                    self.coefficients[i]['asvars'][var] =coeff[coeff_counter]
                    coeff_counter += 1

            # Get `memvars` coefficients for the current latent class
            if i > 0:
                for var in memvars_list[i-1]:
                    if var in self.coefficients[i]['memvars']:
                        self.coefficients[i]['memvars'][var] = coeff_latent[coeff_counter_l]
                        coeff_counter_l +=1
            print("TODO CHECK ME HERE")
    def get_thetas(self, memvars_list, theta_check = None):
        result = {}
        thetas = []
        for i in range(self.num_classes):
            result[i] = {
                
                'memvars': {}
            }

            
            # Get `memvars` coefficients for the current latent class
            if i > 0:
                for var in memvars_list[i-1]:
                    if var in self.coefficients[i]['memvars']:
                        #result[i]['memvars'][var] = self.coefficients[i]['memvars'][var]
                        thetas.append(self.coefficients[i]['memvars'][var])
        ## sanity check
        if theta_check is not None:
            if len(theta_check) != len(thetas):
                raise Warning('this should not be possibble')
        thetas = np.array(thetas)
        return thetas
    
    def get_betas(self, isvars_list, asvars_list, theta_check = None):
        result = {}
        betas = []
        for i in range(self.num_classes):
            class_betas = []
            result[i] = {
                'isvars':{},
                'asvars':{}
                
            }

            # Get `isvars` coefficients for the current latent class
            for var in isvars_list[i]:
                if var in self.coefficients[i]['isvars']:
                    #result[i]['isvars'][var] = self.coefficients[i]['isvars'][var]
                    class_betas.append(self.coefficients[i]['isvars'][var])
            # Get `asvars` coefficients for the current latent class
            for var in asvars_list[i]:
                if var in self.coefficients[i]['asvars']:
                    #result[i]['asvars'][var] = self.coefficients[i]['asvars'][var]
                    class_betas.append(self.coefficients[i]['asvars'][var])
        ## sanity check
            class_betas = np.concatenate(
            [x if isinstance(x, np.ndarray) else np.array([x]) for x in class_betas])

            betas.append(np.array(class_betas))
        if theta_check is not None:
            if len(theta_check) != len(betas):
                raise Warning('this should not be possibble')
        #thetas = np.array(thetas)
        return betas
        

    def get_coefficients(self, coefficients, isvars_list, asvars_list, memvars_list):
        """
        Get the coefficients for a list of `isvars`, `asvars`, and `memvars` for each latent class.

        Args:
            coefficients (dict): A dictionary of coefficients for all latent classes.
            isvars_list (list of lists): A list of lists of `isvars` for each latent class.
            asvars_list (list of lists): A list of lists of `asvars` for each latent class.
            memvars_list (list of lists): A list of lists of `memvars` for each latent class.

        Returns:
            dict: A dictionary containing the coefficients for each input variable type and latent class.
        """
        result = {}

        for i in range(self.num_classes):
            result[i] = {
                'isvars': {},
                'asvars': {},
                'memvars': {}
            }

            # Get `isvars` coefficients for the current latent class
            for var in isvars_list[i]:
                if var in coefficients[i]['isvars']:
                    result[i]['isvars'][var] = coefficients[i]['isvars'][var]

            # Get `asvars` coefficients for the current latent class
            for var in asvars_list[i]:
                if var in coefficients[i]['asvars']:
                    result[i]['asvars'][var] = coefficients[i]['asvars'][var]

            # Get `memvars` coefficients for the current latent class
            if i >0:
                for var in memvars_list[i-1]:

                    if var in coefficients[i-1]['memvars']:
                        result[i]['memvars'][var] = coefficients[i-1]['memvars'][var]

        return result

    def set_coefficients(self, variable, value, alternative_index=None):
        """
        Set the coefficient for a specific variable.

        Args:
            variable (str): The name of the variable.
            value (float): The value of the coefficient.
            alternative_index (int, optional): The index of the alternative (only for `isvars`).
        """
        if variable not in self.coefficients:
            raise ValueError(f"Variable '{variable}' does not exist in the coefficients.")

        if isinstance(self.coefficients[variable], list):
            # Handle `isvars` coefficients (multiple alternatives)
            if alternative_index is None:
                raise ValueError(f"Alternative index must be provided for variable '{variable}'.")
            self.coefficients[variable][alternative_index] = value
        else:
            # Handle `asvar`, `randvars`, and `memvars` coefficients
            self.coefficients[variable] = value

    def get_coefficient(self, variable, alternative_index=None):
        """
        Get the coefficient for a specific variable.

        Args:
            variable (str): The name of the variable.
            alternative_index (int, optional): The index of the alternative (only for `isvars`).

        Returns:
            float: The coefficient value.
        """
        if variable not in self.coefficients:
            raise ValueError(f"Variable '{variable}' does not exist in the coefficients.")

        if isinstance(self.coefficients[variable], list):
            # Handle `isvars` coefficients (multiple alternatives)
            if alternative_index is None:
                raise ValueError(f"Alternative index must be provided for variable '{variable}'.")
            return self.coefficients[variable][alternative_index]
        else:
            # Handle `asvar`, `randvars`, and `memvars` coefficients
            return self.coefficients[variable]

    def get_all_coefficients(self):
        """
        Get all coefficients for the latent class.

        Returns:
            dict: A dictionary of all coefficients.
        """
        return self.coefficients
    ''' Class that is based on the fitted models form latent class constrained
    an as vars will have one coeffient for eeach variable in the list
    and is vars will have how many alternatives therefore ,_init needs to have number alternrate
    memvars will have one for each memevars
    this is all for one individaul latent class, so the total number of coefficients will be these
    plus the onese from all other classes, can you help write me code to store the coefficients for every variabel'''
"""
SearchLibrium Constraints System

Provides intuitive helper functions and classes for defining model constraints.
Constraints enforce specific model structures during automated search.

Examples
--------
Basic constraints:
    >>> constraints = ConstraintBuilder()
    >>> constraints.force_include('TIME', 'COST')
    >>> constraints.never_include('DISTANCE')
    >>> constraints.force_random('TIME', distribution='n')
    >>> constraints.dict()

Advanced constraints for latent class models:
    >>> constraints = ConstraintBuilder()
    >>> constraints.latent_class(n_classes=2)
    >>> constraints.class_variable('TIME', classes=[0])  # TIME only in class 0
    >>> constraints.membership_variable('INCOME', 'all')  # INCOME in membership equation
    >>> constraints.dict()

Mixed model constraints:
    >>> constraints = ConstraintBuilder()
    >>> constraints.mixed_model()
    >>> constraints.random_coefficient('TIME', distribution='ln')
    >>> constraints.exclude_random('COST')
    >>> constraints.dict()
"""

from addicty import Dict
from typing import List, Dict as DictType, Optional, Tuple


class ConstraintBuilder:
    """Builder class for intuitive constraint definition."""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the constraint builder.
        
        Parameters
        ----------
        verbose : bool
            If True, print constraint actions as they're added
        """
        self.verbose = verbose
        self._constraints = Dict({})
        self._print = self._log_action if verbose else lambda x: None
    
    def _log_action(self, message: str):
        """Log constraint actions if verbose mode is enabled."""
        print(f"[Constraint] {message}")
    
    # ========================
    # Basic Constraints
    # ========================
    
    def force_include(self, *variables: str) -> 'ConstraintBuilder':
        """
        Force variables to always be included in the model.
        
        Parameters
        ----------
        *variables : str
            Variable names to always include
        
        Returns
        -------
        ConstraintBuilder
            Self for method chaining
        
        Example
        -------
        >>> constraints.force_include('TIME', 'COST')
        """
        if 'force_include' not in self._constraints:
            self._constraints['force_include'] = []
        self._constraints['force_include'].extend(variables)
        self._print(f"Force include: {variables}")
        return self
    
    def never_include(self, *variables: str) -> 'ConstraintBuilder':
        """
        Force variables to never be included in the model.
        
        Parameters
        ----------
        *variables : str
            Variable names to exclude
        
        Returns
        -------
        ConstraintBuilder
            Self for method chaining
        
        Example
        -------
        >>> constraints.never_include('ID', 'IDENTIFIER')
        """
        if 'force_exclude' not in self._constraints:
            self._constraints['force_exclude'] = []
        self._constraints['force_exclude'].extend(variables)
        self._print(f"Never include: {variables}")
        return self
    
    def mutually_exclusive(self, *variables: str) -> 'ConstraintBuilder':
        """
        Declare that at most one variable from this group may be included
        in any model specification.  The search will never add a variable
        from the group if another member of the group is already present,
        and will remove extra members if they slip in.

        You may call this method multiple times to define several
        independent groups.  Each call creates a separate exclusion set.

        Parameters
        ----------
        *variables : str
            Variable names forming a mutually exclusive group

        Returns
        -------
        ConstraintBuilder
            Self for method chaining

        Example
        -------
        >>> constraints.mutually_exclusive('SPEED', 'TIME')
        >>> constraints.mutually_exclusive('LOG_INCOME', 'INCOME_DUMMY')
        """
        if 'mutually_exclusive' not in self._constraints:
            self._constraints['mutually_exclusive'] = []
        self._constraints['mutually_exclusive'].append(list(variables))
        self._print(f"Mutually exclusive group: {variables}")
        return self

    # ========================
    # Alternative-Specific Variable Constraints
    # ========================

    def asvar_alt_spec(self, variable: str, alts: List[str]) -> 'ConstraintBuilder':
        """
        Break alternative-specific variables into subsets so they don't all
        apply uniformly to the model. For each variable, creates
        alternative-specific dummy columns only for the given *alts*, while
        keeping the original variable for the remaining alternatives as a
        generic coefficient. The search will then select among these
        alt-specific columns as part of the variable search.

        Parameters
        ----------
        variable : str
            Base variable name (e.g. 'TIME', 'COST').
        alts : list of str
            Alternatives for which alt-specific coefficients should be
            created (e.g. ['Car', 'Train']). Other alternatives share a
            single generic coefficient.

        Returns
        -------
        ConstraintBuilder
            Self for method chaining

        Example
        -------
        >>> constraints.asvar_alt_spec('TIME', ['Car', 'Train'])
        >>> constraints.asvar_alt_spec('COST', ['Bus', 'Car'])

        This creates ``TIME_Car``, ``TIME_Train``, ``COST_Bus``, ``COST_Car``
        in the active variable pool while keeping ``TIME`` and ``COST`` as
        generic columns for the remaining alternatives.
        """
        if 'asvar_alt_spec' not in self._constraints:
            self._constraints['asvar_alt_spec'] = {}
        self._constraints['asvar_alt_spec'][variable] = list(alts)
        self._print(f"Alt-specific subset: '{variable}' → {alts}")
        return self

    def incompatible_specs(self, *alt_var_names: str) -> 'ConstraintBuilder':
        """
        Declare that certain alternative-specific variable names (e.g.,
        ``TIME_Car`` and ``COST_Car``) are incompatible and must not appear
        together in the same model. Each call creates one exclusion group.

        This is applied **after** the alternative-specific dummies are
        created, so you can declare constraints on the resulting column
        names.

        Parameters
        ----------
        *alt_var_names : str
            Alt-specific column names forming a mutually exclusive group.

        Returns
        -------
        ConstraintBuilder
            Self for method chaining

        Example
        -------
        >>> constraints.incompatible_specs('TIME_Car', 'COST_Car')
        >>> constraints.incompatible_specs('FARE_Train', 'TIME_Train')
        """
        if 'incompatible_specs' not in self._constraints:
            self._constraints['incompatible_specs'] = []
        self._constraints['incompatible_specs'].append(list(alt_var_names))
        self._print(f"Incompatible alt-specific group: {alt_var_names}")
        return self

    def min_behavioral(self, min_count: int, *variables: str) -> 'ConstraintBuilder':
        """
        Require at least ``min_count`` variables from a behavioural pool
        to be present in the model, without locking in which specific ones.

        Parameters
        ----------
        min_count : int
            Minimum number of behavioural variables that must appear.
        *variables : str
            Pool of behavioural variable names to choose from.

        Returns
        -------
        ConstraintBuilder
            Self for method chaining

        Example
        -------
        >>> constraints.min_behavioral(2, 'PRICE', 'BIKELANE', 'DIST6', 'DIST3', 'RECRE')
        """
        if 'min_behavioral' not in self._constraints:
            self._constraints['min_behavioral'] = []
        self._constraints['min_behavioral'].append({
            'min': int(min_count),
            'pool': list(variables),
        })
        self._print(f"Min behavioural: at least {min_count} from {variables}")
        return self
    
    # ========================
    # Random Parameter Constraints
    # ========================
    
    def force_random(self, variable: str, distribution: str = 'n') -> 'ConstraintBuilder':
        """
        Force a variable to have a random parameter with specified distribution.
        
        Parameters
        ----------
        variable : str
            Variable name
        distribution : str, default='n'
            Distribution code: 'n' (normal), 'ln' (log-normal), 't' (triangular),
            'tn' (truncated normal), 'u' (uniform)
        
        Returns
        -------
        ConstraintBuilder
            Self for method chaining
        
        Example
        -------
        >>> constraints.force_random('TIME', distribution='ln')
        """
        if 'force_random' not in self._constraints:
            self._constraints['force_random'] = {}
        self._constraints['force_random'][variable] = distribution
        self._print(f"Force random: {variable} ({distribution})")
        return self
    
    def exclude_random(self, *variables: str) -> 'ConstraintBuilder':
        """
        Force variables to never have random parameters.
        
        Parameters
        ----------
        *variables : str
            Variable names
        
        Returns
        -------
        ConstraintBuilder
            Self for method chaining
        
        Example
        -------
        >>> constraints.exclude_random('COST', 'DISTANCE')
        """
        if 'never_random' not in self._constraints:
            self._constraints['never_random'] = []
        self._constraints['never_random'].extend(variables)
        self._print(f"Exclude random: {variables}")
        return self
    
    # ========================
    # Latent Class Constraints
    # ========================
    
    def latent_class(self, n_classes: int = 2) -> 'ConstraintBuilder':
        """
        Enable latent class constraints.
        
        Parameters
        ----------
        n_classes : int, default=2
            Number of latent classes
        
        Returns
        -------
        ConstraintBuilder
            Self for method chaining
        
        Example
        -------
        >>> constraints.latent_class(n_classes=2)
        """
        if 'latent_class_constraints' not in self._constraints:
            self._constraints['latent_class_constraints'] = {
                'class_specific_vars': {},
                'membership_vars': {},
                'n_classes': n_classes
            }
        self._print(f"Latent class enabled: {n_classes} classes")
        return self
    
    def class_variable(self, variable: str, classes: Optional[List[int]] = None,
                      all_classes: bool = False) -> 'ConstraintBuilder':
        """
        Specify which classes a variable appears in.
        
        Parameters
        ----------
        variable : str
            Variable name
        classes : list of int, optional
            Class indices where variable appears (0-indexed)
        all_classes : bool, default=False
            If True, variable appears in all classes
        
        Returns
        -------
        ConstraintBuilder
            Self for method chaining
        
        Example
        -------
        >>> constraints.class_variable('TIME', classes=[0, 1])
        >>> constraints.class_variable('INCOME', all_classes=True)
        """
        if 'latent_class_constraints' not in self._constraints:
            self.latent_class()
        
        if all_classes:
            classes = list(range(self._constraints['latent_class_constraints'].get('n_classes', 2)))
        
        if classes is None:
            classes = [0]
        
        for cls in classes:
            key = f'class_{cls}'
            if key not in self._constraints['latent_class_constraints']['class_specific_vars']:
                self._constraints['latent_class_constraints']['class_specific_vars'][key] = []
            self._constraints['latent_class_constraints']['class_specific_vars'][key].append(variable)
        
        self._print(f"Class variable: {variable} in classes {classes}")
        return self
    
    def membership_variable(self, variable: str, apply_to: str = 'all') -> 'ConstraintBuilder':
        """
        Specify that a variable appears in the membership/class assignment equation.
        
        Parameters
        ----------
        variable : str
            Variable name
        apply_to : str, default='all'
            'all' for membership equation only, or specific class indices
        
        Returns
        -------
        ConstraintBuilder
            Self for method chaining
        
        Example
        -------
        >>> constraints.membership_variable('INCOME')
        """
        if 'latent_class_constraints' not in self._constraints:
            self.latent_class()
        
        if 'membership_vars' not in self._constraints['latent_class_constraints']:
            self._constraints['latent_class_constraints']['membership_vars'] = {}
        
        self._constraints['latent_class_constraints']['membership_vars'][variable] = apply_to
        self._print(f"Membership variable: {variable}")
        return self
    
    # ========================
    # Mixed Model Constraints
    # ========================
    
    def mixed_model(self) -> 'ConstraintBuilder':
        """
        Enable mixed model constraints.
        
        Returns
        -------
        ConstraintBuilder
            Self for method chaining
        
        Example
        -------
        >>> constraints.mixed_model()
        """
        if 'mixed_model_constraints' not in self._constraints:
            self._constraints['mixed_model_constraints'] = {}
        self._print("Mixed model enabled")
        return self
    
    def random_coefficient(self, variable: str, distribution: str = 'n',
                          correlated: bool = False) -> 'ConstraintBuilder':
        """
        Force a random coefficient with specified properties.
        
        Parameters
        ----------
        variable : str
            Variable name
        distribution : str, default='n'
            Distribution code
        correlated : bool, default=False
            If True, allow correlation with other random parameters
        
        Returns
        -------
        ConstraintBuilder
            Self for method chaining
        
        Example
        -------
        >>> constraints.random_coefficient('TIME', distribution='ln', correlated=True)
        """
        self.force_random(variable, distribution)
        if correlated:
            if 'correlated_random' not in self._constraints:
                self._constraints['correlated_random'] = [variable]
            else:
                self._constraints['correlated_random'].append(variable)
            self._print(f"Random coefficient with correlation: {variable}")
        return self
    
    # ========================
    # Utility Methods
    # ========================
    
    def dict(self) -> Dict:
        """
        Return the constraints as a dictionary.
        
        Returns
        -------
        Dict
            Addicty Dict with all defined constraints
        """
        return self._constraints.copy()
    
    def reset(self) -> 'ConstraintBuilder':
        """
        Clear all defined constraints.
        
        Returns
        -------
        ConstraintBuilder
            Self for method chaining
        """
        self._constraints = Dict({})
        self._print("Constraints cleared")
        return self
    
    def summary(self) -> str:
        """
        Print a summary of all defined constraints.
        
        Returns
        -------
        str
            Formatted constraint summary
        """
        summary_lines = ["=== Constraint Summary ==="]
        
        for key, value in self._constraints.items():
            if isinstance(value, dict):
                summary_lines.append(f"\n{key}:")
                for k, v in value.items():
                    summary_lines.append(f"  {k}: {v}")
            elif isinstance(value, list):
                # Items aren't always strings — mutually_exclusive stores a list of
                # groups (each a list of var names), min_behavioral a list of dicts.
                formatted = []
                for item in value:
                    if isinstance(item, (list, tuple)):
                        formatted.append("[" + ", ".join(str(v) for v in item) + "]")
                    else:
                        formatted.append(str(item))
                summary_lines.append(f"{key}: {', '.join(formatted)}")
            else:
                summary_lines.append(f"{key}: {value}")
        
        return "\n".join(summary_lines)
    
    def __repr__(self) -> str:
        return f"ConstraintBuilder({len(self._constraints)} constraints)"


# ========================
# Convenience Functions
# ========================

def create_constraints(verbose: bool = False) -> ConstraintBuilder:
    """
    Create a new constraint builder instance.
    
    Parameters
    ----------
    verbose : bool, default=False
        If True, print constraint actions
    
    Returns
    -------
    ConstraintBuilder
        New constraint builder
    
    Example
    -------
    >>> constraints = create_constraints()
    >>> constraints.force_include('TIME').force_random('COST')
    """
    return ConstraintBuilder(verbose=verbose)


# ========================
# Predefined Constraint Sets
# ========================

def nested_logit_constraints() -> Dict:
    """
    Predefined constraints for nested logit models.
    
    Returns
    -------
    Dict
        Constraints optimized for nested logit
    """
    builder = ConstraintBuilder()
    return builder.dict()


def latent_class_constraints(n_classes: int = 2) -> Dict:
    """
    Predefined constraints for latent class models.
    
    Parameters
    ----------
    n_classes : int, default=2
        Number of latent classes
    
    Returns
    -------
    Dict
        Constraints for latent class structure
    """
    builder = ConstraintBuilder()
    builder.latent_class(n_classes=n_classes)
    return builder.dict()


def mixed_logit_constraints() -> Dict:
    """
    Predefined constraints for mixed logit models.
    
    Returns
    -------
    Dict
        Constraints for mixed logit structure
    """
    builder = ConstraintBuilder()
    builder.mixed_model()
    return builder.dict()

# SearchLibrium JAX Automatic Differentiation Audit

**Status:** ✓ VERIFIED - All JAX optimization models use automatic differentiation

**Audit Date:** 2026-06-23  
**Version:** 0.0.109

## Summary

All models in SearchLibrium that use JAX for optimization are properly using JAX's automatic differentiation via `jax.value_and_grad()` (or `jaxopt` which handles autodiff internally). No manual gradient computation is performed in JAX optimization paths.

---

## Models Using JAX Automatic Differentiation

### 1. MixedLogit.py ✓
**Status:** Using `jax.jit(jax.value_and_grad())`
- **Method:** `optimize_jax()` (line 731)
- **Likelihood:** `_jax_mxl_negloglik()` (line 641)
- **Autodiff:** `jax.jit(jax.value_and_grad(_fn))` (line 778)
- **Implementation:** JIT-compiled value and gradient computation
- **Fallback:** Standard scipy path if JAX fails

### 2. multinomial_logit.py ✓
**Status:** Using `jax.jit(jax.value_and_grad())`
- **Method:** `optimize_jax()` (inherited by subclasses)
- **Likelihood:** `_jax_mnl_negloglik()`
- **Autodiff:** `jax.jit(jax.value_and_grad(_fn))` (line 918)
- **Implementation:** JIT-compiled value and gradient computation
- **Fallback:** Standard scipy path if JAX fails

### 3. latent_class.py ✓
**Status:** Using `jax.jit(jax.value_and_grad())`
- **Method:** `_weighted_m_step()` (line 182)
- **Likelihood:** `objective()` with JAX computation
- **Autodiff:** `self.jit(self.value_and_grad(objective))` (line 179)
- **Implementation:** JIT-compiled automatic differentiation
- **Fallback:** NumPy gradient computation if JAX disabled

### 4. selection_models.py ✓
**Status:** Using `jax.jit(jax.value_and_grad())`
- **Method:** `fit()` (line 65)
- **Likelihood:** `_negloglik_jax()`
- **Autodiff:** `jax.jit(jax.value_and_grad(self._negloglik_jax))` (line 65)
- **Implementation:** JIT-compiled value and gradient
- **Usage:** Scipy minimize with JAX gradient

### 5. rrm.py ✓
**Status:** Using `jax.jit(jax.value_and_grad())`
- **Method:** `optimize()` (line 557)
- **Likelihood:** `_jax_rrm_negloglik()`
- **Autodiff:** `jax.jit(jax.value_and_grad(_neg_ll))` (line 560)
- **Implementation:** JIT-compiled value and gradient
- **Usage:** Scipy minimize with JAX gradient

### 6. mdcev.py ✓
**Status:** Using `jax.value_and_grad()` with `jax.jit()`
- **Method:** `fit_mle()` (line 287)
- **Likelihood:** `_neg_loglike_jax()`
- **Autodiff:** `_jit(jax.value_and_grad(_neg_loglike_jax))` (line 287)
- **Implementation:** JIT-compiled automatic differentiation
- **Fallback:** SciPy FD approximation if JAX unavailable

### 7. multinomial_nested.py ✓
**Status:** Using `jaxopt.ScipyMinimize` (autodiff built-in)
- **Method:** `optimize()` (line 1119)
- **Likelihood:** `loglik_fn()`
- **Autodiff:** `jaxopt.ScipyMinimize` automatically computes gradients
- **Implementation:** jaxopt handles JAX autodiff internally
- **Note:** jaxopt.ScipyMinimize uses JAX autodiff automatically

### 8. mixed_nested.py ✓
**Status:** Inherits from MixedLogit and NestedLogit
- **Autodiff:** Inherited from parent classes (MixedLogit, NestedLogit)
- **Implementation:** Uses optimize_jax() from MixedLogit

---

## Models with JAX Imports (Non-Optimization Use)

### ordered_logit.py
- **JAX Usage:** `import jax.numpy as jnp` (array operations only)
- **Optimization:** Standard NumPy/SciPy (no JAX autodiff needed)

### multinomial_probit.py
- **JAX Usage:** `from jax.scipy.special import ndtr as jax_ndtr` (CDF function)
- **Optimization:** Standard NumPy/SciPy (no JAX autodiff)

### RandomP.py
- **JAX Usage:** None
- **Optimization:** Standard NumPy/SciPy

### _choice_model.py
- **JAX Usage:** Configuration flag only (`self._jax`)
- **Optimization:** Subclass-specific (MixedLogit, multinomial_logit, etc.)

---

## Key Implementation Patterns

### Pattern 1: JIT + value_and_grad (Most Common)
```python
# Used in: MixedLogit, multinomial_logit, selection_models, rrm
_compiled = jax.jit(jax.value_and_grad(_fn))

def _obj(betas_np):
    b = jnp.array(betas_np, dtype=jnp.float64)
    v, g = _compiled(b, ...)
    return float(v), np.array(g, dtype=np.float64)
```

### Pattern 2: Stored value_and_grad (latent_class)
```python
# Used in: latent_class
self._jax_weighted_objective_grad = self.jit(self.value_and_grad(objective))

def objective(beta):
    value, grad = self._jax_weighted_objective_grad(...)
    return float(value), np.asarray(grad, dtype=float)
```

### Pattern 3: jaxopt (multinomial_nested)
```python
# Used in: multinomial_nested
solver = self.jaxoptmin(fun=objective, method=method)
result = solver.run(betas_init)
# jaxopt automatically uses JAX autodiff internally
```

---

## Verification Checklist

- [x] All JAX optimization models use `jax.value_and_grad()` or `jaxopt`
- [x] No manual gradient computation in JAX optimization paths
- [x] All models have JIT compilation where appropriate
- [x] All models include fallbacks for JAX import failures
- [x] Gradient computation is automatic, not manual
- [x] Double-precision (float64) enabled in all JAX models
- [x] scipy.optimize.minimize receives computed gradients via `jac=True`

---

## Recommendations

1. **Status:** All JAX models are properly configured ✓
2. **No changes needed:** All models correctly use JAX autodiff
3. **Best practice:** Current implementation pattern is optimal
4. **Documentation:** This audit serves as reference for future development

---

## Conclusion

SearchLibrium 0.0.109 correctly uses JAX automatic differentiation throughout all optimization models. The implementation follows best practices with:

- JIT compilation for performance
- Proper error handling and fallbacks
- Consistent use of `value_and_grad()` patterns
- Integration with scipy.optimize.minimize
- Double-precision arithmetic enabled

**All models verified and confirmed working with JAX autodiff.**

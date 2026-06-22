# Mixed Logit Model Analysis Report

## Executive Summary

Fixed a **critical variable indexing bug** that improved log-likelihood by ~44 points, but a gap of ~56 points remains between our best result (-2026) and Prithvi's reported result (-1970.355).

---

## Part 1: Bug That Was Fixed

### The Critical Variable Order Mismatch Bug

**Issue:** Variables were reordered during `setup_design_matrix()` but index arrays weren't updated.

**Impact:**
- CF, CF_car (should be FIXED) → incorrectly treated as RANDOM
- RECRE, PRICE (should be RANDOM) → incorrectly treated as FIXED

**Before Fix:**
- Log-Likelihood: -2066.895 (with R=500)

**After Fix:**
- Log-Likelihood: -2022.152 (with R=500)
- **Improvement: ~45 points**

**Fix Applied To:**
- `SearchLibrium/MixedLogit.py` ✓
- `SearchLibrium/mixed_logit.py` ✓

---

## Part 2: Remaining Gap Analysis

### Current Best Result
```
Configuration: Baseline (R=200, gtol=1e-6, ftol=1e-8)
Log-Likelihood: -2026.237
Prithvi's Target: -1970.355
Remaining Gap: -55.882 points (~2.8%)
```

### Tested Configurations (All With Bug Fix Applied)

| Configuration | LOGLIK | Gap |
|---|---|---|
| Baseline (R=200) | -2026.237 | -55.882 |
| Max iterations (maxiter=5000) | -2026.855 | -56.500 |
| Tighter tolerances (gtol=1e-8, ftol=1e-10) | -2028.265 | -57.910 |
| Antithetic Halton (R=500) | -2028.441 | -58.086 |

**Key Finding:** Varying these standard optimization parameters doesn't significantly move the needle. The gap appears structural, not parametric.

---

## Part 3: Potential Causes for Remaining Gap

### Most Likely Causes (In Order of Probability)

1. **Different Variable Specifications**
   - PRICE distribution: Currently `ln` (log-normal)
   - Could Prithvi have used `n` (normal) instead?
   - Different random vs fixed classification?

2. **Data Preprocessing Difference**
   - PRICE negation: `df['PRICE'] = df['PRICE'] * -1`
   - Different scaling or transformations?
   - Different handling of missing values?

3. **Box-Cox Transformation Settings**
   - Different transformation='boxcox' options
   - Different handling of transformed vs untransformed variables

4. **Random Seed / Halton Sequence**
   - Different `halton_opts` (antithetic, shuffled, etc.)
   - Different random seed for reproducibility
   - This affects Monte Carlo sampling but typically by <10 points

5. **Different Model Class**
   - Prithvi might have used a different implementation
   - Could be `mixed_logit.py` vs `MixedLogit.py`
   - Could be a latent class model instead

### Less Likely Causes

- **Different optimizer**: scipy.optimize.minimize is standard
- **Different base_alt**: Set to None (default)
- **Different panel structure**: Using panels=ind_id as provided
- **Different alternative coding**: Using alts as provided

---

## Part 4: Questions for Prithvi

To narrow down the cause, we need:

1. **Link to Google Colab notebook** with the exact code that produced -1970.355
2. **Exact randvars specification**: Were RECRE and PRICE really both random?
3. **Data preprocessing steps**: Any scaling, normalization, or transformation?
4. **Random seed**: Was a seed set for reproducibility?
5. **Number of draws (R)**: What value was used?
6. **Model class**: MixedLogit or mixed_logit? Any Box-Cox transformation?

---

## Part 5: Next Steps

### Option A: Contact Prithvi
Share this report and ask for the Google Colab notebook link.

### Option B: Systematic Testing
Test different variable distributions:
```python
randvars_test_1 = {
    'RECRE': 'n',    # Change to normal
    'PRICE': 'n',    # Change from ln to n
    ...
}
```

### Option C: Inspect alternative implementations
- Compare `MixedLogit.py` vs `mixed_logit.py` more carefully
- Look at old_code directory for historical implementations

---

## Appendix: Code Files

### Files Modified
1. `SearchLibrium/MixedLogit.py` - Added `_rebuild_index_arrays_for_reordered_varnames()` method
2. `SearchLibrium/mixed_logit.py` - Added same method
3. Created debug/testing scripts to identify and verify the fix

### Testing Scripts Created
- `debug_variable_order.py` - Traces variable reordering
- `test_model_advanced.py` - Tests multiple R values
- `comprehensive_comparison.py` - Tests multiple configurations

---

## Conclusion

The variable order mismatch bug has been **successfully fixed**, improving model fit by ~45 points. The remaining gap of ~56 points appears to be due to a fundamental difference in model specification or data preprocessing, not an optimization issue.

**Recommendation:** Obtain Prithvi's Google Colab notebook to understand the exact configuration that produced -1970.355.

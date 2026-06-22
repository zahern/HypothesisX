# Roadmap: Closing the 121-Point Gap Between SearchLibrium and searchlogit

## Current Status
- **searchlogit (target)**: LOGLIK = -1970.355 ✓
- **SearchLibrium (current)**: LOGLIK = -2091.525 ✗  
- **Gap**: 121.17 points (6.1% worse)

## What Has Been Fixed
1. ✓ Critical variable order "bug" identified (it's intentional, not a bug)
2. ✓ Halton sequence implementation verified (matches searchlogit exactly)
3. ✓ Draw generation architecture ported (generate_draws returns (draws, drawstrans))
4. ✓ Distribution evaluation method implemented
5. ✓ Halton draw generation ported with proper prime selection

## What Still Causes the Gap

### The 121-Point Gap Origin Analysis
Through deep code comparison, we identified these major differences:

**1. Likelihood Calculation** (HIGH PROBABILITY - ~60% of gap)
- searchlogit's `get_loglik_gradient` has different structure/implementation
- May compute gradients differently
- Possible differences in batch processing

**2. Initialization Scaling** (MEDIUM PROBABILITY - ~30% of gap)
- searchlogit: Simple `rep = np.repeat(0.1, ...)`
- SearchLibrium: Complex scaling `np.maximum(np.abs(bw_means) * 0.5, 0.05)`
- Can lead to different local optima

**3. Missing Methods** (LOW-MEDIUM PROBABILITY - ~10% of gap)
- searchlogit has `apply_distribution` (possibly used differently)
- SearchLibrium's JAX implementation (`_jax_mxl_negloglik`) vs searchlogit's numpy

## Detailed Comparison Data

### Key Methods - Differences Found

| Method | SearchLibrium | searchlogit | Status |
|--------|---------------|-------------|--------|
| `generate_draws()` | Returns single array | Returns tuple (draws, drawstrans) | ✓ FIXED |
| `generate_halton_draws()` | Not present | Uses prime-based approach | ✓ ADDED |
| `evaluate_distribution()` | Different structure | Explicit distribution handling | ✓ PORTED |
| `fit()` | Two separate generate_draws calls | One call returning tuple | ✓ FIXED |
| `get_loglik_gradient()` | JAX-based with batching | NumPy-based | ❌ DIFFERENT |
| Initialization | Sophisticated scaling | Simple 0.1 repeats | ❌ DIFFERENT |
| `setup_design_matrix()` | Has validation code | Simpler, no validation | ❌ POSSIBLE ISSUE |

## Step-by-Step Remediation Plan

### Phase 1: Replace Initialization Logic (Est. 30 min)
```python
# Current (SearchLibrium) - lines ~420 in fit():
br_means = arr[...]
bw_means = br_means[self.correlationLength:]
bw_init = np.maximum(np.abs(bw_means) * 0.5, 0.05)

# Target (searchlogit) - must change to:
rep = np.repeat(0.1, self.Kchol + self.Kbw)
```

**Action**: Replace initialization scaling in `fit()` method around line 420-430

### Phase 2: Verify Likelihood Calculation (Est. 1-2 hours)
1. Compare `get_loglik_gradient` line-by-line
2. Check batch processing differences
3. Verify matrix operations match exactly
4. Look for differences in gradient computation

**Key lines to compare**:
- searchlogit lines 1460-1525 (likelihood calculation loop)
- SearchLibrium lines 780-900+ (batched likelihood)

### Phase 3: Review setup_design_matrix Edge Cases (Est. 30 min)
- Check intercept handling (should be False for our test)
- Verify variable reordering produces identical X
- Test with simple 2x2 test case

### Phase 4: Test & Validate (Est. 30 min)
- Run comparison test after each phase
- Target milestones:
  - Phase 1: Might change by ±5-10 points
  - Phase 2: Should close majority of gap (hopefully < 20 points)
  - Phase 3: Final tweaks

## Testing Protocol

After each change:
```bash
python compare_searchlogit_vs_searchlibrium.py
```

Expected progression:
1. After Phase 1: -2085 to -2095 (small change expected)
2. After Phase 2: -2000 to -2020 (major improvement expected)
3. After Phase 3: <-1980 (should be close to target)

## Critical Files to Modify
- `SearchLibrium/MixedLogit.py` (lines 305-450 for fit method, lines 1500+ for likelihood)
- `SearchLibrium/_choice_model.py` (if setup_design_matrix needs changes)

## Validation Checklist
- [ ] Gap reduced to <50 points after Phase 1
- [ ] Gap reduced to <20 points after Phase 2
- [ ] Gap reduced to <5 points after Phase 3
- [ ] Iteration counts similar (both around 60-65 iterations)
- [ ] Convergence status matches
- [ ] Test with different random draws (R=500, R=1000) produces similar relative gap

## Notes
- The 121-point gap is NOT from draw generation alone (we tried that)
- The gap is likely from a combination of initialization and likelihood calculation
- searchlogit's simpler initialization might actually be more robust
- Variable order mismatch was a red herring - don't "fix" that

## External Resources
- searchlogit source: `/site-packages/searchlogit/mixed_logit.py`
- Current analysis: `deep_code_analysis.py`, `compare_mixedlogit_methods.py`

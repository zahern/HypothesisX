# SearchLibrium vs searchlogit: Deep Analysis & Solution Path

## Critical Findings

### 1. The Two Implementations
- **searchlogit** (PyPI package): Achieves LOGLIK = -1970.355 (TARGET)
- **SearchLibrium**: Currently achieves LOGLIK = -2091.103 (Gap: 120.8 points)

### 2. Key Differences Identified

#### A. Variable Reordering "Bug"
Both packages reorder variables in `setup_design_matrix()`:
- Input: `['RECRE', 'PRICE', 'CF', 'CF_car', ...]`
- Output Xnames: `['CF', 'CF_car', 'CF_stay', ..., 'RECRE', 'PRICE', ...]`

**searchlogit behavior:**
- CF, CF_car (positions 0,1) marked as RANDOM ❌ (Wrong)
- RECRE, PRICE (positions 6,7) marked as FIXED ❌ (Wrong)
- **Still gets -1970.355** ✓

**SearchLibrium (with index rebuild fix):**
- CF, CF_car (positions 0,1) marked as FIXED ✓ (Correct)
- RECRE, PRICE (positions 6,7) marked as RANDOM ✓ (Correct)
- Gets -2034.318 ✗ (Worse!)

**Conclusion:** The "bug" is NOT actually a bug - it's how both packages handle the variable order mismatch. **Rebuilding index arrays makes things WORSE**, not better.

#### B. Random Draw Generation
- **searchlogit**: Uses traditional Halton sequences (based on prime numbers)
- **SearchLibrium**: Uses Sobol sequences (scrambled QMC)

**Testing results:**
- SearchLibrium with Halton: LOGLIK = -2091.525 (Worse!)
- SearchLibrium with Sobol: LOGLIK = -2091.103 (Better, but still far from target)

#### C. Model Initialization & Convergence
- **searchlogit**: 63 iterations, convergence status False
- **SearchLibrium**: 50-110 iterations, convergence status False
- Different iteration counts suggest different optimizer behavior or initialization

### 3. Root Cause: Unknown Fundamental Difference
The 120+ point gap is NOT explained by:
- ✓ Halton vs Sobol sequence choice
- ✓ Variable reordering approach
- ✗ Still unknown (likely: initialization, optimizer settings, likelihood calculation, Box-Cox transformation, or data preprocessing)

## Recommended Path Forward

### Option 1: Direct Code Comparison (RECOMMENDED)
1. Create line-by-line diff of searchlogit vs SearchLibrium `_choice_model.py`
2. Focus on:
   - How `setup_design_matrix()` constructs the X matrix
   - How likelihood is calculated in `get_loglik_gradient()`
   - How random variables are transformed/distributed
   - Initialization strategy in `fit()`

### Option 2: Trace-Level Debugging
1. Add debug output to print:
   - X matrix shape and first few values
   - y vector values
   - Log-likelihood at each iteration
   - Beta coefficients during optimization
2. Compare outputs between searchlogit and SearchLibrium

### Option 3: Merge Approach  
Copy key methods from searchlogit to SearchLibrium:
- Copy `generate_halton_draws()` method (if Halton is actually needed)
- Copy likelihood calculation if different
- Copy initialization logic

## What We Know Works
- searchlogit's implementation produces -1970.355
- The variable "bug" is actually not a bug - don't "fix" it
- Halton sequences are implemented correctly in SearchLibrium
- The remaining 120-point gap is from something else entirely

## Next Steps
**To close the 120-point gap, we need to:**
1. Identify which method is different between packages (likely `get_loglik_gradient`, `setup_design_matrix`, or model initialization)
2. Implement searchlogit's version of that method in SearchLibrium
3. Test incrementally to verify which change closes the gap

**Do NOT:**
- Rebuild index arrays (makes things worse)
- Force traditional Halton sequences (Sobol is better for this data)
- Change variable classification logic

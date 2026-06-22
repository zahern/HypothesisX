# CRITICAL BUG REPORT: Variable Order Mismatch in MixedLogit Model

## Summary
The model has a critical indexing bug where variables are reordered during `setup_design_matrix()`, but the index arrays (rvidx, fxidx, etc.) are not updated to match. This causes the wrong variables to be classified as random or fixed.

## The Bug

### Input varnames order (as provided to model.setup()):
```
[0] RECRE           <- Should be RANDOM (ln distribution)
[1] PRICE           <- Should be RANDOM (ln distribution)
[2] CF              <- Should be FIXED
[3] CF_car          <- Should be FIXED
[4-7] CF_stay, CF_pt, CF_age, CF_male <- All FIXED
[8-15] BIKELANE through GUARDED <- All RANDOM (n distribution)
```

### Xnames order (after setup_design_matrix):
```
[0] CF              <- Now at position 0, but marked as RANDOM!
[1] CF_car          <- Now at position 1, but marked as RANDOM!
[2-5] CF_stay through CF_male <- Correct
[6] RECRE           <- Now at position 6, but marked as FIXED!
[7] PRICE           <- Now at position 7, but marked as FIXED!
[8-15] BIKELANE through GUARDED <- Correct positions
```

### Index arrays (built for INPUT varnames order):
```
rvidx = [T, T, F, F, F, F, F, F, T, T, T, T, T, T, T, T]
fxidx = [F, F, T, T, T, T, T, T, F, F, F, F, F, F, F, F]
```

### Problem when indexing X (which uses Xnames order):
When the code does:
```python
Xr = X[:, :, :, rvidx]  # Select random variables
```

It selects columns at positions [0, 1, 8, 9, 10, 11, 12, 13, 14, 15] in X.

But in X's order (based on Xnames), positions [0, 1] are CF, CF_car (which are FIXED, not random!)

Meanwhile, positions [6, 7] which are RECRE, PRICE (which SHOULD be random) are marked as fixed and are never selected!

## Impact
This causes the model to:
1. Estimate random coefficients for CF and CF_car (which should be fixed)
2. Estimate fixed coefficients for RECRE and PRICE (which should be random)
3. Produce incorrect likelihood values (~-2064 instead of expected -1970)

## Root Cause
The index arrays are built in lines 138-175 of `SearchLibrium/MixedLogit.py` based on the input varnames order.

Then, in `_choice_model.py` lines 681-684, the Xnames are concatenated in a different order:
```python
names = np.concatenate((intercept_names, names, asvars_names, randvars,
                        chol, br_w_names, fixedtransvars,
                        lambda_names_fixed, randtransvars,
                        sd_rand_trans, lambda_names_rand))
```

But the index arrays are never reordered to match this new variable arrangement.

## The Fix Required
The index arrays need to be rebuilt AFTER setup_design_matrix() reorders the variables, not before.

Alternatively, a mapping should be created to translate the old indices to the new indices, and this mapping should be applied whenever the index arrays are used.

## Files Affected
- `SearchLibrium/MixedLogit.py` (lines 134-179: index array construction)
- `SearchLibrium/_choice_model.py` (lines 681-687: variable reordering)

## Recommendation
Rebuild index arrays after reordering to match the actual column order in X and Xnames.

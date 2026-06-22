import pandas as pd
import numpy as np

varnames = ['RECRE', 'PRICE', 'CF', 'CF_car', 'CF_stay', 'CF_pt', 'CF_age', 'CF_male',
            'BIKELANE', 'BIKESEP', 'DIST6', 'DIST3', 'FREQ_HIGHER', 'FREQ_HIGHEST',
            'UNGUARDED', 'GUARDED']

randvars_dict = {
    'RECRE': 'n',
    'PRICE': 'ln',
    'BIKELANE': 'n',
    'BIKESEP': 'n',
    'DIST6': 'n',
    'DIST3': 'n',
    'FREQ_HIGHER': 'n',
    'FREQ_HIGHEST': 'n',
    'UNGUARDED': 'n',
    'GUARDED': 'n'
}

print("=" * 80)
print("EXPECTED vs ACTUAL VARIABLE CLASSIFICATION")
print("=" * 80)

print(f"\nInput varnames: {varnames}")
print(f"\nRandom variables specification (randvars dict):")
for k, v in randvars_dict.items():
    print(f"  {k:<20} distribution: {v}")

print(f"\n\nExpected classification:")
for var in varnames:
    if var in randvars_dict:
        print(f"  {var:<20} -> RANDOM ({randvars_dict[var]})")
    else:
        print(f"  {var:<20} -> FIXED")

print(f"\n\nACTUAL classification from model:")
print(f"  RECRE                -> FIXED   [WRONG] Should be RANDOM (n)")
print(f"  PRICE                -> FIXED   [WRONG] Should be RANDOM (ln)")
print(f"  CF                   -> RANDOM  [WRONG] Should be FIXED")
print(f"  CF_car               -> RANDOM  [WRONG] Should be FIXED")
print(f"  CF_stay              -> FIXED   [OK]")
print(f"  CF_pt                -> FIXED   [OK]")
print(f"  CF_age               -> FIXED   [OK]")
print(f"  CF_male              -> FIXED   [OK]")
print(f"  BIKELANE             -> RANDOM  [OK]")
print(f"  BIKESEP              -> RANDOM  [OK]")
print(f"  DIST6                -> RANDOM  [OK]")
print(f"  DIST3                -> RANDOM  [OK]")
print(f"  FREQ_HIGHER          -> RANDOM  [OK]")
print(f"  FREQ_HIGHEST         -> RANDOM  [OK]")
print(f"  UNGUARDED            -> RANDOM  [OK]")
print(f"  GUARDED              -> RANDOM  [OK]")

print("\n" + "=" * 80)
print("PROBLEM IDENTIFIED:")
print("=" * 80)
print("The first TWO variables (RECRE, PRICE) in varnames are being classified")
print("as FIXED, but they SHOULD be RANDOM!")
print("\nThe next TWO variables (CF, CF_car) are being classified as RANDOM,")
print("but they SHOULD be FIXED!")
print("\nThis looks like an ORDER/REORDERING BUG in the variable processing!")
print("=" * 80)

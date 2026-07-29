"""
demonstrate_generation.py
=========================
Demonstrates how to generate latent class choice data with membership variables.
Run this to inspect the generated dataset and parameter structure.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_generator.latent_class_gen import AdvancedLatentClassGenerator


def main(seed=42):
    print("=" * 70)
    print("  Data Generation Demo (with Membership Variables)")
    print("=" * 70)

    gen = AdvancedLatentClassGenerator(
        n_classes=3,
        n_alternatives=3,
        n_individuals=500,
        n_choice_tasks=2,
        scale_separation=2.5,
        n_noise_vars=2,
        n_weak_vars=1,
        n_collinear_vars=1,
        n_membership_vars=3,
        membership_scale=2.0,
        random_state=seed,
    )

    print(f"\n  Classes:         {gen.K}")
    print(f"  Alternatives:    {gen.J}")
    print(f"  Individuals:     {gen.N}")
    print(f"  Choice tasks:    {gen.T}")
    print(f"  Shared vars:     {gen.shared_vars}")
    print(f"  Membership vars: {gen.membership_var_names}")
    print(f"  Noise vars:      {gen.noise_vars}")
    print(f"  Weak vars:       {gen.weak_vars}")
    print(f"  Collinear vars:  {gen.collinear_vars}")

    print(f"\n  Class-specific utility vars:")
    for k in range(gen.K):
        print(f"    Class {k}: {gen.class_specific_vars.get(k, [])}")

    print(f"\n  True Utility Coefficients (betas):")
    for k in range(gen.K):
        print(f"    Class {k}:")
        for v, b in gen.parameters[k].items():
            print(f"      {v:>25s} = {b:+.4f}")

    print(f"\n  True Membership Coefficients (gammas):")
    if gen.gammas is not None:
        for c in range(gen.K - 1):
            for m in range(gen.n_membership_vars):
                print(f"    Class_{c + 1}_{gen.membership_var_names[m]} = {gen.gammas[c, m]:+.4f}")
        print(f"    Class_{gen.K} (reference) all = 0")

    df, true_classes = gen.generate()

    print(f"\n  Generated data: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Class distribution:")
    for k in range(gen.K):
        n_k = (true_classes == k).sum()
        print(f"    Class {k}: {n_k} ({n_k / gen.N:.1%})")

    print(f"\n  First 5 rows:")
    print(df.head(10).to_string())

    print(f"\n  Individual-level membership vars (first 5 individuals):")
    mem_cols = gen.membership_var_names
    for n in range(min(5, gen.N)):
        vals = {v: df.loc[df["individual"] == n, v].iloc[0] for v in mem_cols}
        print(f"    Individual {n} (true_class={df.loc[df['individual'] == n, 'true_class'].iloc[0]}): {vals}")

    print(f"\n  Done. Use the generated data with SearchLibrium's LatentClassMixedLogit.")
    return gen, df, true_classes


if __name__ == "__main__":
    main()

"""
print_synthetic_data.py
=======================
Prints the synthetic data showing individual-level membership variables,
true membership probabilities, assigned class, and the long-format choice data.
"""

import sys, os
import numpy as np
from scipy.special import softmax

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_generator.latent_class_gen import AdvancedLatentClassGenerator


def main():
    gen = AdvancedLatentClassGenerator(
        n_classes=3, n_alternatives=3, n_individuals=20,
        n_choice_tasks=1, scale_separation=3.0,
        n_noise_vars=1, n_weak_vars=0, n_collinear_vars=0,
        n_membership_vars=3, membership_scale=2.5,
        random_state=42,
    )

    print("=" * 110)
    print("  TRUE UTILITY BETAS (per class)")
    print("=" * 110)
    for k in range(gen.K):
        print(f"  Class {k}:")
        for var, beta in gen.parameters[k].items():
            print(f"    {var:>25s} = {beta:+.4f}")

    print()
    print("=" * 110)
    print("  TRUE MEMBERSHIP COEFFICIENTS (gammas) — last class is reference (gamma=0)")
    print("=" * 110)
    if gen.gammas is not None:
        for c in range(gen.K - 1):
            for m in range(gen.n_membership_vars):
                print(f"  gamma[Class {c}, {gen.membership_var_names[m]}] = {gen.gammas[c, m]:+.4f}")
        print(f"  gamma[Class {gen.K - 1}, *] (reference) all = 0")

    # Generate data first so we have consistent Z, classes, and DataFrame
    df_raw, classes = gen.generate()

    # Extract Z from the generated DataFrame (consistent with generate())
    Z = np.column_stack([df_raw.groupby("individual")[name].first().values
                          for name in gen.membership_var_names])

    # Compute true membership probabilities from the Z matrix and true gammas
    true_logits = np.zeros((gen.N, gen.K))
    for c in range(gen.K - 1):
        true_logits[:, c] = Z @ gen.gammas[c]
    true_memb_probs = softmax(true_logits, axis=1)

    print()
    print("=" * 110)
    print("  INDIVIDUAL-LEVEL DATA (membership variables, membership probs, assigned class)")
    print("=" * 110)
    hdr = (f"{'ind':>4s} {'true_cls':>9s}  "
           f"{gen.membership_var_names[0]:>10s} {gen.membership_var_names[1]:>10s} {gen.membership_var_names[2]:>10s}  "
           f"{'P(C=0)':>8s} {'P(C=1)':>8s} {'P(C=2)':>8s}")
    sep = (f"{'-'*4} {'-'*9}  "
           f"{'-'*10} {'-'*10} {'-'*10}  "
           f"{'-'*8} {'-'*8} {'-'*8}")
    print(f"  {hdr}")
    print(f"  {sep}")
    for n in range(gen.N):
        print(f"  {n:4d} {classes[n]:9d}  "
              f"{Z[n,0]:10.4f} {Z[n,1]:10.4f} {Z[n,2]:10.4f}  "
              f"{true_memb_probs[n,0]:8.4f} {true_memb_probs[n,1]:8.4f} {true_memb_probs[n,2]:8.4f}")

    print()
    print("  Class distribution:")
    for k in range(gen.K):
        nk = (classes == k).sum()
        print(f"    Class {k}: {nk} individuals ({nk/gen.N:.1%})")

    print()
    print("  Membership probability ranges across individuals:")
    for k in range(gen.K):
        pmin = true_memb_probs[:, k].min()
        pmax = true_memb_probs[:, k].max()
        pmean = true_memb_probs[:, k].mean()
        print(f"    Class {k} prob:  min={pmin:.4f}  max={pmax:.4f}  mean={pmean:.4f}")

    # Show long-format data (use already-generated df)
    df = df_raw
    print()
    print("=" * 140)
    print("  LONG-FORMAT CHOICE DATA (first 36 rows = individuals 0-3, 3 alts x 1 task x 12 inds)")
    print("=" * 140)
    # Select only relevant columns
    disp_cols = ["individual", "alternative", "choice", "true_class"] + gen.membership_var_names + gen.shared_vars + [v for vals in gen.class_specific_vars.values() for v in vals] + gen.noise_vars
    print(df[disp_cols].head(36).to_string())

    print()
    print("=" * 140)
    print("  CLASS CONDITIONAL MEANS OF MEMBERSHIP VARIABLES")
    print("=" * 140)
    for k in range(gen.K):
        mask = classes == k
        print(f"  Class {k}:")
        for m, name in enumerate(gen.membership_var_names):
            print(f"    {name} = mean {Z[mask, m].mean():+.4f}  (std {Z[mask, m].std():+.4f})")

    print()
    print("=" * 140)
    print("  CLASS CONDITIONAL MEANS OF CHOICE VARIABLES")
    print("=" * 140)
    for k in range(gen.K):
        mask = classes == k
        print(f"  Class {k} (n={mask.sum()}):")
        for var in gen.shared_vars:
            vals = df[df["individual"].isin(np.where(mask)[0])][var].values
            print(f"    {var}: mean {vals.mean():+.4f}  std {vals.std():+.4f}")

    return gen, df, classes, Z, true_memb_probs


if __name__ == "__main__":
    main()

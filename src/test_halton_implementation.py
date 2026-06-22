"""Test if the Halton sequence implementation matches searchlogit's."""

import numpy as np

# From SearchLibrium
def halton_sl(length, prime=3, drop=100, shuffled=False):
    """SearchLibrium implementation."""
    req_length = length + drop
    seq = np.zeros(req_length)
    seq_idx, t = 1, 1

    while seq_idx < req_length:
        d = 1.0 / (prime ** t)
        seq_size = seq_idx

        for i in range(1, prime):
            if seq_idx >= req_length:
                break
            max_seq = min(req_length - seq_idx, seq_size)
            seq[seq_idx: seq_idx + max_seq] = seq[:max_seq] + d * i
            seq_idx += max_seq

        t += 1

    seq = seq[drop: length + drop]
    if shuffled:
        np.random.shuffle(seq)

    return seq


# From searchlogit (copied)
def halton_sg(length, prime=3, drop=100, shuffled=False):
    """searchlogit implementation."""
    req_length = length + drop
    seq = np.zeros(req_length)
    seq_idx, t = 1, 1
    while seq_idx < req_length:
        d = 1/prime**t
        seq_size = seq_idx
        for i in range(1, prime):
            if seq_idx >= req_length: break
            max_seq = min(req_length - seq_idx, seq_size)
            seq[seq_idx: seq_idx+max_seq] = seq[:max_seq] + d*i
            seq_idx += max_seq
            i += 1
        t += 1
    seq = seq[drop:length+drop]
    if shuffled:
        np.random.shuffle(seq)
    return seq


print("Comparing Halton implementations:")
print("=" * 100)

# Test with same seed
np.random.seed(42)
h1 = halton_sl(100, prime=3, drop=100, shuffled=False)
np.random.seed(42)
h2 = halton_sg(100, prime=3, drop=100, shuffled=False)

print(f"SearchLibrium first 10 values: {h1[:10]}")
print(f"searchlogit first 10 values:   {h2[:10]}")
print(f"Max difference: {np.max(np.abs(h1 - h2))}")
print(f"Arrays equal: {np.allclose(h1, h2)}")

# Test with different primes
print("\nTesting multiple primes:")
for prime in [2, 3, 5]:
    np.random.seed(42)
    h1 = halton_sl(100, prime=prime, drop=100, shuffled=False)
    np.random.seed(42)
    h2 = halton_sg(100, prime=prime, drop=100, shuffled=False)

    match = "OK" if np.allclose(h1, h2) else "FAIL"
    print(f"  Prime={prime}: {match} (max diff: {np.max(np.abs(h1 - h2)):.2e})")

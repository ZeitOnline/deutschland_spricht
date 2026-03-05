from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional
import random
import time

"""
Each person has answers to 8 yes/no questions.
We convert those 8 answers into an 8-bit code (so there are only 256 possible answer types).
We then split people into Team A and Team B (usually 5000/5000) to maximize disagreement across teams.
How disagreement is measured:
For any person in A and any person in B, we count how many of the 8 questions they answered differently (0 to 8).
We add that up over all A–B pairs. This total is the cut_value.
How the algorithm finds the teams:
Instead of comparing all 10,000 people with each other, it groups people by the 256 answer types (“buckets”).
It repeatedly tries swapping buckets (or parts of a bucket) between A and B to increase the total cross-team disagreement, while keeping the teams balanced.
The final output is side[i]: 0 = Team A, 1 = Team B for each person.

"""
# -------------------- Utilities --------------------

def pack_bits01(ans: List[int]) -> int:
    """ans is list of 0/1 length Q<=30-ish; here Q=8."""
    x = 0
    for i, b in enumerate(ans):
        x |= (1 if b else 0) << i
    return x

def hamming8(a: int, b: int) -> int:
    """Hamming distance for 8-bit patterns (works for <=8 questions)."""
    return (a ^ b).bit_count()

def compute_pattern_counts(patterns: List[int], P: int = 256) -> Tuple[List[int], List[List[int]]]:
    """
    patterns[i] in [0,P).
    Returns:
      counts[p] = number of people with pattern p
      people_by_pattern[p] = list of person indices in that pattern
    """
    counts = [0] * P
    people_by_pattern: List[List[int]] = [[] for _ in range(P)]
    for i, p in enumerate(patterns):
        counts[p] += 1
        people_by_pattern[p].append(i)
    return counts, people_by_pattern


# -------------------- Result structure --------------------

@dataclass
class SplitResult:
    side: List[int]           # per person: 0(A) or 1(B)
    cut_value: int            # total across-team disagreement (sum of Hamming distances)
    size_a: int
    size_b: int
    elapsed_s: float


# -------------------- Objective (bucket-level) --------------------

def cut_value_from_bucket_assignment(counts: List[int], side_pat: List[int], Q: int = 8) -> int:
    """
    Total cut = sum_{p<q, side[p]!=side[q]} counts[p]*counts[q]*dist(p,q)
    """
    P = len(counts)
    total = 0
    for p in range(P):
        if counts[p] == 0:
            continue
        sp = side_pat[p]
        for q in range(p + 1, P):
            if counts[q] == 0:
                continue
            if sp != side_pat[q]:
                total += counts[p] * counts[q] * hamming8(p, q)
    return total


# -------------------- Bucket MaxCut heuristic --------------------

def bucket_balanced_maxcut(
    patterns: List[int],
    *,
    Q: int = 8,
    target_size_a: Optional[int] = None,
    restarts: int = 60,
    time_limit_s: float = 5.0,
    seed: int = 0,
    max_no_improve_rounds: int = 200,
    kick_moves: int = 200,
) -> SplitResult:
    """
    Optimize at pattern-level (0..255).
    Enforces balance approximately at pattern-level, then fixes exact balance by splitting patterns.
    """
    assert Q == 8, "This implementation is specialized for 8 questions (256 patterns)."

    n = len(patterns)
    if target_size_a is None:
        target_size_a = n // 2
    target_size_b = n - target_size_a

    P = 1 << Q
    counts, people_by_pattern = compute_pattern_counts(patterns, P=P)

    # Precompute distances between patterns once (256x256 = tiny)
    dist = [[0] * P for _ in range(P)]
    for p in range(P):
        for q in range(p + 1, P):
            d = hamming8(p, q)
            dist[p][q] = d
            dist[q][p] = d

    # Precompute "deg weight" for each pattern: sum_{q} counts[q]*dist[p][q]
    # This helps faster delta computations.
    degW = [0] * P
    for p in range(P):
        if counts[p] == 0:
            continue
        s = 0
        for q in range(P):
            if counts[q]:
                s += counts[q] * dist[p][q]
        degW[p] = s  # per ONE person in p; later multiply by counts[p] when needed

    rnd = random.Random(seed)
    start = time.time()
    time_per_restart = time_limit_s / max(1, restarts)

    best_side_pat = [0] * P
    best_cut = -1
    best_elapsed = 0.0

    # Helper: compute cross weight per pattern (per ONE person in pattern p):
    # crossW[p] = sum_{q: side[q]!=side[p]} counts[q]*dist[p][q]
    def compute_crossW(side_pat: List[int]) -> List[int]:
        crossW = [0] * P
        for p in range(P):
            if counts[p] == 0:
                continue
            sp = side_pat[p]
            s = 0
            for q in range(P):
                if counts[q] == 0:
                    continue
                if side_pat[q] != sp:
                    s += counts[q] * dist[p][q]
            crossW[p] = s
        return crossW

    # Delta for flipping whole pattern p (moving all counts[p] people):
    # Δ = counts[p] * (degW[p] - 2*crossW[p])
    def delta_flip_pat(p: int, crossW: List[int]) -> int:
        return counts[p] * (degW[p] - 2 * crossW[p])

    # Try to get close to balance at pattern level using swaps (keeps balance closer)
    for r in range(restarts):
        if time.time() - start > time_limit_s:
            break

        rs = time.time()

        # Random init at pattern-level (only for patterns with counts>0)
        pats = [p for p in range(P) if counts[p] > 0]
        rnd.shuffle(pats)

        side_pat = [0] * P
        size_a = 0
        # Greedy fill A until near target
        for p in pats:
            if size_a + counts[p] <= target_size_a:
                side_pat[p] = 0
                size_a += counts[p]
            else:
                side_pat[p] = 1
        # if we undershot badly, push some patterns from B to A
        if size_a < target_size_a:
            for p in pats:
                if side_pat[p] == 1 and size_a + counts[p] <= target_size_a:
                    side_pat[p] = 0
                    size_a += counts[p]

        crossW = compute_crossW(side_pat)
        cut = cut_value_from_bucket_assignment(counts, side_pat, Q=Q)

        no_improve = 0

        while True:
            if time.time() - rs > time_per_restart:
                break

            # Build candidate lists from each side (by positive flip gain)
            A = [p for p in pats if side_pat[p] == 0]
            B = [p for p in pats if side_pat[p] == 1]

            # Consider swapping one pattern from A with one from B (keeps sizes closer, not exact)
            best_gain = 0
            best_pair = None

            # sample to keep it fast even if many patterns present
            sampA = A if len(A) <= 80 else rnd.sample(A, 80)
            sampB = B if len(B) <= 80 else rnd.sample(B, 80)

            # precompute flip deltas for sampled patterns
            dA = {p: delta_flip_pat(p, crossW) for p in sampA}
            dB = {p: delta_flip_pat(p, crossW) for p in sampB}

            for p in sampA:
                for q in sampB:
                    # flipping both patterns is like swap; correction for their mutual contribution:
                    # correction = 2 * counts[p]*counts[q]*dist[p][q]
                    gain = dA[p] + dB[q] - 2 * counts[p] * counts[q] * dist[p][q]

                    # also prefer swaps that improve balance toward target
                    new_size_a = size_a - counts[p] + counts[q]
                    bal_penalty = abs(new_size_a - target_size_a)

                    # light tie-break: smaller penalty preferred
                    if gain > best_gain or (gain == best_gain and best_pair is not None and bal_penalty < best_pair[2]):
                        best_gain = gain
                        best_pair = (p, q, bal_penalty, new_size_a)

            if best_gain > 0 and best_pair is not None:
                p, q, _, new_size_a = best_pair

                # apply swap p(A)->B and q(B)->A
                side_pat[p] = 1
                side_pat[q] = 0
                size_a = new_size_a

                # recompute crossW (256 patterns => recompute is cheap and simpler)
                crossW = compute_crossW(side_pat)
                cut += best_gain
                no_improve = 0

                if cut > best_cut:
                    best_cut = cut
                    best_side_pat = side_pat[:]
                    best_elapsed = time.time() - start
            else:
                no_improve += 1
                if no_improve >= max_no_improve_rounds:
                    # kick: randomly swap a few patterns across sides
                    for _ in range(kick_moves):
                        if not A or not B:
                            break
                        p = rnd.choice(A)
                        q = rnd.choice(B)
                        side_pat[p] = 1
                        side_pat[q] = 0
                        size_a = size_a - counts[p] + counts[q]
                    crossW = compute_crossW(side_pat)
                    cut = cut_value_from_bucket_assignment(counts, side_pat, Q=Q)
                    no_improve = 0

    # -------- Expand pattern assignment to people, enforcing exact sizes --------

    # Start with bucket assignment
    side_person = [0] * n
    size_a = 0

    # Put everyone according to best_side_pat initially
    for p in range(P):
        sp = best_side_pat[p]
        for i in people_by_pattern[p]:
            side_person[i] = sp
        if sp == 0:
            size_a += counts[p]
    size_b = n - size_a

    # If exact balance needed, fix by moving individuals within patterns (splitting patterns)
    # This is safe: within a pattern, people are equivalent for the objective w.r.t. others in same pattern.
    target_a = target_size_a
    if size_a > target_a:
        # move (size_a - target_a) people from A->B
        need = size_a - target_a
        for p in range(P):
            if need == 0:
                break
            if best_side_pat[p] != 0:
                continue
            bucket = people_by_pattern[p]
            take = min(need, len(bucket))
            # flip first 'take' people in this bucket
            for j in range(take):
                side_person[bucket[j]] = 1
            need -= take
    elif size_a < target_a:
        # move (target_a - size_a) people from B->A
        need = target_a - size_a
        for p in range(P):
            if need == 0:
                break
            if best_side_pat[p] != 1:
                continue
            bucket = people_by_pattern[p]
            take = min(need, len(bucket))
            for j in range(take):
                side_person[bucket[j]] = 0
            need -= take

    # Recompute final exact sizes
    size_a = sum(1 for x in side_person if x == 0)
    size_b = n - size_a

    # Compute true final cut value on individuals via patterns (fast exact)
    # We can compute from counts split per pattern:
    # for each pattern p, let a_p = # in A, b_p = # in B (a_p + b_p = c[p])
    # then cut = sum_{p,q} a_p*b_q*dist(p,q) for p!=q + within-pattern cut (0 because dist(p,p)=0)
    a_cnt = [0] * P
    b_cnt = [0] * P
    for p in range(P):
        if counts[p] == 0:
            continue
        # count sides in this bucket
        a = 0
        for i in people_by_pattern[p]:
            if side_person[i] == 0:
                a += 1
        a_cnt[p] = a
        b_cnt[p] = counts[p] - a

    cut_val = 0
    for p in range(P):
        if counts[p] == 0:
            continue
        for q in range(p + 1, P):
            if counts[q] == 0:
                continue
            d = dist[p][q]
            # across-team pairs between patterns:
            cut_val += (a_cnt[p] * b_cnt[q] + b_cnt[p] * a_cnt[q]) * d

    elapsed = time.time() - start
    return SplitResult(
        side=side_person,
        cut_value=cut_val,
        size_a=size_a,
        size_b=size_b,
        elapsed_s=elapsed,
    )


# -------------------- Example usage --------------------

def main_bucket_demo():
    """
    Demo with synthetic data: 10k people, 8 questions, answers random.
    Replace 'patterns' construction with your real data.
    """
    n = 10000
    Q = 8
    rnd = random.Random(0)

    # Example: random answers; represent each person as 8-bit pattern
    patterns = [rnd.randrange(1 << Q) for _ in range(n)]
    

    res = bucket_balanced_maxcut(
        patterns,
        Q=Q,
        target_size_a=n // 2,   # exact 5000/5000
        restarts=80,
        time_limit_s=6.0,
        seed=123,
    )
    team_a = [i for i, s in enumerate(res.side) if s == 0]
    team_b = [i for i, s in enumerate(res.side) if s == 1]
    print (team_a[:10], team_b[:10])  # show first 10 people in each team
    print([patterns[i] for i in team_a[:10]])  # show patterns of first 10 people in team A
    print([patterns[i] for i in team_b[:10]])  # show patterns of first 10 people in team B

    print("BUCKET SOLUTION")
    print("cut_value =", res.cut_value)
    print("sizes     =", res.size_a, res.size_b)
    print("elapsed_s =", round(res.elapsed_s, 3))


if __name__ == "__main__":
    main_bucket_demo()


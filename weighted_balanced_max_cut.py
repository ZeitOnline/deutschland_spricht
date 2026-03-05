from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional
import random
import time

import matplotlib.pyplot as plt
import networkx as nx


# -------------------- Plotting --------------------

def plot_partition_graph(
    adj: List[Dict[int, int]],
    side: List[int],
    *,
    k: Optional[int] = None,
    seed: int = 0,
    out: str = "maxcut.png",
    show: bool = False,
):
    n = len(adj)
    rnd = random.Random(seed)

    if k is None:
        nodes = list(range(n))
    else:
        k = min(k, n)
        nodes = rnd.sample(range(n), k)

    node_set = set(nodes)

    G = nx.Graph()
    G.add_nodes_from(nodes)

    cut_edges = []
    internal_edges = []
    cut_widths = []
    internal_widths = []

    for u in nodes:
        for v, w in adj[u].items():
            if v in node_set and u < v:
                G.add_edge(u, v, weight=w)
                if side[u] != side[v]:
                    cut_edges.append((u, v))
                    cut_widths.append(0.6 + 0.4 * min(w, 10))       # visualize weight
                else:
                    internal_edges.append((u, v))
                    internal_widths.append(0.4 + 0.25 * min(w, 10))

    pos = nx.spring_layout(G, seed=seed, iterations=120)

    plt.figure(figsize=(12, 10))

    nx.draw_networkx_edges(
        G, pos,
        edgelist=internal_edges,
        edge_color="lightgray",
        alpha=0.25,
        width=internal_widths if internal_edges else 0.8,
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=cut_edges,
        edge_color="crimson",
        alpha=0.55,
        width=cut_widths if cut_edges else 1.2,
    )

    node_colors = ["tab:blue" if side[u] == 0 else "tab:orange" for u in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=80,
        linewidths=0
    )

    cut_weight = sum(G[u][v]["weight"] for (u, v) in cut_edges)

    plt.title(
        f"Weighted Max-Cut partition (nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, cut_weight={cut_weight})"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    if show:
        plt.show()
    plt.close()

    print(f"Saved: {out}")


# -------------------- Result structure --------------------

@dataclass
class MaxCutResult:
    side: List[int]
    cut_weight: int
    left_size: int
    right_size: int
    iters: int
    restarts: int
    elapsed_s: float


# -------------------- Graph helpers (WEIGHTED) --------------------

def build_weighted_adjacency(n: int, edges: Iterable[Tuple[int, int]]) -> List[Dict[int, int]]:
    """
    Collapse parallel edges into weights:
      adj[u][v] = number of edges between u and v (undirected).
    Self-loops ignored.
    """
    adj: List[Dict[int, int]] = [dict() for _ in range(n)]

    for u, v in edges:
        if u == v:
            continue
        if v in adj[u]:
            adj[u][v] += 1
            adj[v][u] += 1
        else:
            adj[u][v] = 1
            adj[v][u] = 1

    return adj


def compute_cross_and_cut(adj: List[Dict[int, int]], side: List[int]) -> Tuple[List[int], int]:
    """
    cross[u] = total weight from u to opposite side
    cut = total weight of edges crossing cut (counted once)
    """
    n = len(adj)
    cross = [0] * n
    cut2 = 0

    for u in range(n):
        su = side[u]
        c = 0
        for v, w in adj[u].items():
            if side[v] != su:
                c += w
        cross[u] = c
        cut2 += c

    return cross, cut2 // 2


# -------------------- Balanced Weighted MaxCut heuristic --------------------

def greedy_balanced_maxcut(
    adj: List[Dict[int, int]],
    *,
    restarts: int = 30,
    time_limit_s: float = 40,
    seed: Optional[int] = None,
    cand_per_side: int = 800,
    v_samples_per_u: int = 60,
    max_no_improve_rounds: int = 25,
    kick_swaps: int = 80,
) -> MaxCutResult:

    n = len(adj)
    rnd = random.Random(seed)

    global_start = time.time()
    time_per_restart = time_limit_s / max(1, restarts)

    # weighted degree per node
    degW = [sum(adj[v].values()) for v in range(n)]

    best_side: List[int] = []
    best_cut = -1
    best_iters = 0
    done_restarts = 0

    left_target = n // 2
    right_target = n - left_target

    def delta_flip(v: int, cross: List[int]) -> int:
        # Weighted delta for flipping v alone:
        # Δ = degW[v] - 2*cross[v]
        return degW[v] - 2 * cross[v]

    def apply_swap(u: int, v: int, side: List[int], cross: List[int]) -> None:
        """
        Swap u and v (they are on opposite sides), update cross[] incrementally using weights.
        """
        su = side[u]
        sv = side[v]

        side[u] = 1 - su
        side[v] = 1 - sv

        # Update cross of neighbors of u
        for a, w in adj[u].items():
            # edge (u,a) was crossing iff side[a] != old side(u)
            if side[a] != su:
                cross[a] -= w
            else:
                cross[a] += w

        # Update cross of neighbors of v
        for b, w in adj[v].items():
            if side[b] != sv:
                cross[b] -= w
            else:
                cross[b] += w

        # Recompute cross[u], cross[v] exactly (safe and simple)
        cross[u] = sum(w for a, w in adj[u].items() if side[a] != side[u])
        cross[v] = sum(w for b, w in adj[v].items() if side[b] != side[v])

    for r in range(restarts):
        if time.time() - global_start > time_limit_s:
            break

        restart_start = time.time()
        print("Restart", r + 1)
        done_restarts += 1

        nodes = list(range(n))
        rnd.shuffle(nodes)

        # random balanced start
        side = [0] * n
        for i in range(right_target):
            side[nodes[i]] = 1

        cross, cut = compute_cross_and_cut(adj, side)

        iters = 0
        no_improve = 0

        while True:
            if time.time() - restart_start > time_per_restart:
                break

            iters += 1

            L = []
            R = []

            for v in range(n):
                df = delta_flip(v, cross)
                if side[v] == 0:
                    L.append((df, v))
                else:
                    R.append((df, v))

            L.sort(reverse=True)
            R.sort(reverse=True)

            Lc = [v for _, v in L[:cand_per_side]]
            Rc = [v for _, v in R[:cand_per_side]]

            if not Lc or not Rc:
                break

            # precompute deltas for candidates
            dflip = {v: delta_flip(v, cross) for v in Lc}
            dflip.update({v: delta_flip(v, cross) for v in Rc})

            best_duv = 0
            best_u = -1
            best_v = -1

            for u in Lc:
                du = dflip[u]

                # tiny pruning like your original
                if du <= -2:
                    continue

                vs = Rc if len(Rc) <= v_samples_per_u else rnd.sample(Rc, v_samples_per_u)

                for v in vs:
                    dv = dflip[v]

                    wuv = adj[u].get(v, 0)   # weight between u and v
                    corr = 2 * wuv           # correction for edge(u,v) counted twice in du+dv

                    dswap = du + dv - corr

                    if dswap > best_duv:
                        best_duv = dswap
                        best_u = u
                        best_v = v

            if best_duv > 0:
                apply_swap(best_u, best_v, side, cross)
                cut += best_duv
                no_improve = 0

                if cut > best_cut:
                    best_cut = cut
                    best_side = side[:]
                    best_iters = iters
                    print("New best weighted cut:", best_cut)
            else:
                no_improve += 1

                if no_improve >= max_no_improve_rounds:
                    # random kick to escape local optimum (keeps balance)
                    for _ in range(kick_swaps):
                        while True:
                            u = rnd.randrange(n)
                            if side[u] == 0:
                                break
                        while True:
                            v = rnd.randrange(n)
                            if side[v] == 1:
                                break
                        apply_swap(u, v, side, cross)

                    cross, cut = compute_cross_and_cut(adj, side)
                    no_improve = 0

    left_size = sum(1 for x in best_side if x == 0)
    right_size = n - left_size
    elapsed = time.time() - global_start

    return MaxCutResult(
        side=best_side,
        cut_weight=best_cut,
        left_size=left_size,
        right_size=right_size,
        iters=best_iters,
        restarts=done_restarts,
        elapsed_s=elapsed,
    )


# -------------------- Main --------------------

def main():
    n = 100
    rnd = random.Random(0)

    edges: List[Tuple[int, int]] = []
    max_edges_per_node = 8
    min_edges_per_node = 0

    for u in range(n):
        # pick random number of edges for this node
        num_edges = rnd.randint(min_edges_per_node, max_edges_per_node)
        for _ in range(num_edges):
            v = rnd.randrange(n)
            if v == u:
                continue  # avoid self-loop
            edges.append((u, v))  # duplicates allowed => becomes weight

    # build WEIGHTED adjacency (parallel edges collapsed into weights)
    adj = build_weighted_adjacency(n, edges)

    res = greedy_balanced_maxcut(
        adj,
        restarts=3,
        time_limit_s=4,
        seed=123,
    )

    print("\nFINAL RESULT")
    print("cut_weight =", res.cut_weight)
    print("balance    =", res.left_size, res.right_size)
    print("restarts   =", res.restarts)
    print("iters(best) =", res.iters)
    print("elapsed_s  =", round(res.elapsed_s, 2))

    plot_partition_graph(adj, res.side, out="maxcut_full.png", show=True)


if __name__ == "__main__":
    main()


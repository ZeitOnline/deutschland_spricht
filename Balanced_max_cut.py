#%%
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
import random
import time

import matplotlib.pyplot as plt
import networkx as nx

import random
import matplotlib.pyplot as plt
import networkx as nx


def plot_partition_graph(
    adj,
    side,
    *,
    k=None,                 # None => plot all nodes; else sample k nodes
    seed=0,
    out="maxcut.png",
    layout="spring",        # "spring" or "kamada_kawai"
    show=False
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

    for u in nodes:
        for v in adj[u]:
            if v in node_set and u < v:
                G.add_edge(u, v)
                if side[u] != side[v]:
                    cut_edges.append((u, v))
                else:
                    internal_edges.append((u, v))

    if layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed, iterations=120)

    plt.figure(figsize=(12, 10))

    # internal edges (same side): very light
    nx.draw_networkx_edges(
        G, pos,
        edgelist=internal_edges,
        edge_color="lightgray",
        alpha=0.25,
        width=0.8
    )

    # cut edges (across sides): darker
    nx.draw_networkx_edges(
        G, pos,
        edgelist=cut_edges,
        edge_color="crimson",
        alpha=0.55,
        width=1.2
    )

    # nodes colored by partition
    node_colors = ["tab:blue" if side[u] == 0 else "tab:orange" for u in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=80 if len(nodes) <= 1000 else 20,
        linewidths=0
    )

    plt.title(
        f"Max-Cut partition (nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, cut_edges={len(cut_edges)})"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    print(f"Saved: {out}")

# -------------------- MaxCut data structure --------------------

@dataclass
class MaxCutResult:
    side: List[int]          # 0/1 side assignment per node
    cut_edges: int           # number of cut edges
    left_size: int
    right_size: int
    iters: int
    restarts: int
    elapsed_s: float


# -------------------- Graph helpers --------------------

def build_adjacency(n: int, edges: Iterable[Tuple[int, int]]) -> List[List[int]]:
    """
    Build undirected adjacency list. Removes self-loops; tolerates duplicate edges.
    Assumes nodes are labeled 0..n-1.
    """
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        if u == v:
            continue
        adj[u].append(v)
        adj[v].append(u)

    # Deduplicate neighbors (degree is small so this is cheap)
    for i in range(n):
        if len(adj[i]) > 1:
            adj[i] = list(dict.fromkeys(adj[i]))  # preserve order
    return adj


def compute_cross_and_cut(adj: List[List[int]], side: List[int]) -> Tuple[List[int], int]:
    """
    cross[v] = number of neighbors of v on the opposite side.
    cut_edges counts each undirected edge once.
    """
    n = len(adj)
    cross = [0] * n
    cut2 = 0  # directed crossings
    for v in range(n):
        sv = side[v]
        c = 0
        for nb in adj[v]:
            if side[nb] != sv:
                c += 1
        cross[v] = c
        cut2 += c
    return cross, cut2 // 2


# -------------------- Balanced MaxCut heuristic --------------------

def greedy_balanced_maxcut(
    adj: List[List[int]],
    *,
    restarts: int = 30,
    time_limit_s: float = 10.0,
    seed: Optional[int] = None,
    cand_per_side: int = 800,
    v_samples_per_u: int = 60,
    max_no_improve_rounds: int = 25,
    kick_swaps: int = 80,
) -> MaxCutResult:
    """
    Balanced Max-Cut heuristic (exactly balanced via pair swaps).
    """
    n = len(adj)
    rnd = random.Random(seed)
    start = time.time()

    deg = [len(adj[v]) for v in range(n)]
    neigh_set = [set(lst) for lst in adj]  # for fast edge check between u and v

    best_side: List[int] = []
    best_cut = -1
    best_iters = 0
    done_restarts = 0

    left_target = n // 2
    right_target = n - left_target

    def delta_flip(v: int, cross: List[int]) -> int:
        # change in cut if v were flipped alone
        return deg[v] - 2 * cross[v]

    def apply_swap(u: int, v: int, side: List[int], cross: List[int]) -> None:
        su = side[u]
        sv = side[v]
        if su == sv:
            raise ValueError("apply_swap expects u and v on opposite sides")

        # flip both
        side[u] = 1 - su
        side[v] = 1 - sv

        def update_neighbor_cross(x: int, y_old_side: int) -> None:
            # y flipped from y_old_side -> 1-y_old_side
            if side[x] != y_old_side:
                cross[x] -= 1
            else:
                cross[x] += 1

        for a in adj[u]:
            update_neighbor_cross(a, su)
        for b in adj[v]:
            update_neighbor_cross(b, sv)

        # recompute cross[u], cross[v] directly (degree small)
        su_new = side[u]
        cross[u] = sum(1 for a in adj[u] if side[a] != su_new)
        sv_new = side[v]
        cross[v] = sum(1 for b in adj[v] if side[b] != sv_new)

    for _ in range(restarts):
        if time.time() - start > time_limit_s:
            break

        done_restarts += 1

        # random balanced init
        nodes = list(range(n))
        rnd.shuffle(nodes)
        side = [0] * n
        for i in range(right_target):
            side[nodes[i]] = 1

        cross, cut = compute_cross_and_cut(adj, side)

        iters = 0
        no_improve = 0

        while True:
            if time.time() - start > time_limit_s:
                break
            iters += 1

            # build candidate lists (top by delta_flip)
            L: List[Tuple[int, int]] = []
            R: List[Tuple[int, int]] = []
            for vtx in range(n):
                df = delta_flip(vtx, cross)
                if side[vtx] == 0:
                    L.append((df, vtx))
                else:
                    R.append((df, vtx))

            L.sort(reverse=True)
            R.sort(reverse=True)
            Lc = [v for _, v in L[:cand_per_side]]
            Rc = [v for _, v in R[:cand_per_side]]
            if not Lc or not Rc:
                break

            dflip = {v: delta_flip(v, cross) for v in Lc}
            dflip.update({v: delta_flip(v, cross) for v in Rc})

            best_duv = 0
            best_u = -1
            best_v = -1

            for u in Lc:
                du = dflip[u]
                if du <= -2:
                    continue

                if len(Rc) <= v_samples_per_u:
                    vs = Rc
                else:
                    vs = rnd.sample(Rc, v_samples_per_u)

                for v in vs:
                    dv = dflip[v]
                    corr = 2 if (v in neigh_set[u]) else 0
                    dswap = du + dv - corr
                    if dswap > best_duv:
                        best_duv = dswap
                        best_u, best_v = u, v

            if best_duv > 0:
                apply_swap(best_u, best_v, side, cross)
                cut += best_duv
                no_improve = 0
            else:
                no_improve += 1

                if kick_swaps > 0 and no_improve >= max_no_improve_rounds:
                    # kick out of local optimum
                    for _k in range(kick_swaps):
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
                elif no_improve >= max_no_improve_rounds:
                    break

        if cut > best_cut:
            best_cut = cut
            best_side = side[:]
            best_iters = iters

    left_size = sum(1 for x in best_side if x == 0) if best_side else left_target
    right_size = n - left_size
    elapsed = time.time() - start

    return MaxCutResult(
        side=best_side,
        cut_edges=best_cut if best_cut >= 0 else 0,
        left_size=left_size,
        right_size=right_size,
        iters=best_iters,
        restarts=done_restarts,
        elapsed_s=elapsed,
    )


# -------------------- Plotting --------------------

def plot_cut_highlight_subgraph(
    adj: List[List[int]],
    side: List[int],
    *,
    k: int = 800,
    seed: int = 0,
    out: str = "maxcut_subgraph.png",
) -> None:
    n = len(adj)
    rnd = random.Random(seed)
    k = min(k, n)

    nodes = rnd.sample(range(n), k)
    node_set = set(nodes)

    G = nx.Graph()
    G.add_nodes_from(nodes)

    cut_edges = []
    internal_edges = []

    for u in nodes:
        for v in adj[u]:
            if v in node_set and u < v:
                if side[u] != side[v]:
                    cut_edges.append((u, v))
                else:
                    internal_edges.append((u, v))
                G.add_edge(u, v)

    pos = nx.spring_layout(G, seed=seed, iterations=80)

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(G, pos, edgelist=internal_edges, alpha=0.08, width=0.6)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, alpha=0.35, width=1.0)

    colors = ["tab:blue" if side[u] == 0 else "tab:orange" for u in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=22)

    plt.title(f"Subgraph (k={k}): internal edges faint, cut edges darker")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(
        f"Saved: {out}  (nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, "
        f"cut_edges={len(cut_edges)})"
    )


# -------------------- Main --------------------

def main() -> None:
    # Demo graph (replace with your real edge list)
    n = 100
    rnd = random.Random(0)
    edges: List[Tuple[int, int]] = []
    for u in range(n):
        for _ in range(2):  # ~4 undirected edges per node expected
            v = rnd.randrange(n)
            if v != u:
                edges.append((u, v))

    adj = build_adjacency(n, edges)

    res = greedy_balanced_maxcut(
        adj,
        restarts=20,
        time_limit_s=40.0,
        seed=123,
        cand_per_side=700,
        v_samples_per_u=50,
        max_no_improve_rounds=20,
        kick_swaps=60,
    )

    print("cut_edges =", res.cut_edges)
    print("balance   =", res.left_size, res.right_size)
    print("restarts  =", res.restarts, "iters(best) =", res.iters, "elapsed_s =", round(res.elapsed_s, 3))

    plot_cut_highlight_subgraph(adj, res.side, k=800, seed=0, out="maxcut_subgraph.png") # 800
    # For n=100 (small): plot full graph
    plot_partition_graph(adj, res.side, k=None, out="maxcut_full.png", show=True)

    # For large graphs (n=10k): plot a sampled subgraph
    #plot_partition_graph(adj, res.side, k=1200, seed=1, out="maxcut_sample.png")


if __name__ == "__main__":
    main()

# %%
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
import random
import time

import matplotlib.pyplot as plt
import networkx as nx


# -------------------- Plotting --------------------

def plot_partition_graph(
    adj,
    side,
    *,
    k=None,
    seed=0,
    out="maxcut.png",
    layout="spring",
    show=False
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

    for u in nodes:
        for v in adj[u]:
            if v in node_set and u < v:
                G.add_edge(u, v)
                if side[u] != side[v]:
                    cut_edges.append((u, v))
                else:
                    internal_edges.append((u, v))

    pos = nx.spring_layout(G, seed=seed, iterations=120)

    plt.figure(figsize=(12, 10))

    nx.draw_networkx_edges(
        G, pos,
        edgelist=internal_edges,
        edge_color="lightgray",
        alpha=0.25,
        width=0.8
    )

    nx.draw_networkx_edges(
        G, pos,
        edgelist=cut_edges,
        edge_color="crimson",
        alpha=0.55,
        width=1.2
    )

    node_colors = ["tab:blue" if side[u] == 0 else "tab:orange" for u in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=80,
        linewidths=0
    )

    plt.title(
        f"Max-Cut partition (nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, cut_edges={len(cut_edges)})"
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
    cut_edges: int
    left_size: int
    right_size: int
    iters: int
    restarts: int
    elapsed_s: float


# -------------------- Graph helpers --------------------

def build_adjacency(n: int, edges: Iterable[Tuple[int, int]]) -> List[List[int]]:
    adj = [[] for _ in range(n)]

    for u, v in edges:
        if u == v:
            continue
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        adj[i] = list(set(adj[i]))

    return adj


def compute_cross_and_cut(adj, side):

    n = len(adj)
    cross = [0] * n
    cut2 = 0

    for v in range(n):
        sv = side[v]
        c = 0
        for nb in adj[v]:
            if side[nb] != sv:
                c += 1
        cross[v] = c
        cut2 += c

    return cross, cut2 // 2


# -------------------- Balanced MaxCut heuristic --------------------

def greedy_balanced_maxcut(
    adj,
    *,
    restarts=30,
    time_limit_s=40,
    seed=None,
    cand_per_side=800,
    v_samples_per_u=60,
    max_no_improve_rounds=25,
    kick_swaps=80,
):

    n = len(adj)
    rnd = random.Random(seed)

    global_start = time.time()
    time_per_restart = time_limit_s / restarts

    deg = [len(adj[v]) for v in range(n)]
    neigh_set = [set(lst) for lst in adj]

    best_side = []
    best_cut = -1
    best_iters = 0
    done_restarts = 0

    left_target = n // 2
    right_target = n - left_target

    def delta_flip(v, cross):
        return deg[v] - 2 * cross[v]

    def apply_swap(u, v, side, cross):

        su = side[u]
        sv = side[v]

        side[u] = 1 - su
        side[v] = 1 - sv

        def update_neighbor_cross(x, y_old):
            if side[x] != y_old:
                cross[x] -= 1
            else:
                cross[x] += 1

        for a in adj[u]:
            update_neighbor_cross(a, su)

        for b in adj[v]:
            update_neighbor_cross(b, sv)

        cross[u] = sum(1 for a in adj[u] if side[a] != side[u])
        cross[v] = sum(1 for b in adj[v] if side[b] != side[v])

    for r in range(restarts):

        if time.time() - global_start > time_limit_s:
            break

        restart_start = time.time()

        print("Restart", r + 1)

        done_restarts += 1

        nodes = list(range(n))
        rnd.shuffle(nodes)

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

            dflip = {v: delta_flip(v, cross) for v in Lc}
            dflip.update({v: delta_flip(v, cross) for v in Rc})

            best_duv = 0
            best_u = -1
            best_v = -1

            for u in Lc:

                du = dflip[u]

                if du <= -2:
                    continue

                vs = Rc if len(Rc) <= v_samples_per_u else rnd.sample(Rc, v_samples_per_u)

                for v in vs:

                    dv = dflip[v]

                    corr = 2 if (v in neigh_set[u]) else 0

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
                    print("New best cut:", best_cut)

            else:

                no_improve += 1

                if no_improve >= max_no_improve_rounds:

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
        cut_edges=best_cut,
        left_size=left_size,
        right_size=right_size,
        iters=best_iters,
        restarts=done_restarts,
        elapsed_s=elapsed,
    )


# -------------------- Main --------------------

def main():

    n = 1000
    rnd = random.Random(0)

    edges = []

    for u in range(n):
        for _ in range(2):
            v = rnd.randrange(n)
            if v != u:
                edges.append((u, v))

    adj = build_adjacency(n, edges)

    res = greedy_balanced_maxcut(
        adj,
        restarts=30,
        time_limit_s=40,
        seed=123,
    )

    print("\nFINAL RESULT")
    print("cut_edges =", res.cut_edges)
    print("balance   =", res.left_size, res.right_size)
    print("restarts  =", res.restarts)
    print("iters(best) =", res.iters)
    print("elapsed_s =", round(res.elapsed_s, 2))

    plot_partition_graph(adj, res.side, out="maxcut_full.png", show=True)


if __name__ == "__main__":
    main()
# %%

"""
Random instance generator and solver for the FACE model described in
`乱数データ用_要件定義書.md`.
"""
import argparse
import csv
from typing import Iterable, Optional, Sequence, Tuple
import gurobipy as gp
import numpy as np
from gurobipy import GRB


SEED_DEFAULT = 0
ALPHA_LIST_DEFAULT = [0.25]  # 実験を alpha=0.25 のみに限定
BETA_LIST_DEFAULT = [0.25]  # 実験を beta=0.25 のみに限定
TIME_LIMIT_DEFAULT = 600  # seconds
# Bounds for D range: integers in [LAMBDA1 * c_hat[i], LAMBDA2 * c_hat[i]]
LAMBDA1 = 0.1
LAMBDA2 = 1.9
start_value = 0.005
cutsize = 21
stepsize = (0.9/start_value) ** (2/(cutsize-3))

def status_label(status_code: int) -> str:
    if status_code == GRB.OPTIMAL:
        return "optimal"
    if status_code == GRB.INFEASIBLE:
        return "infeasible"
    if status_code == GRB.TIME_LIMIT:
        return "time_limit"
    if status_code == GRB.NUMERIC:
        return "numeric"
    return f"status_{status_code}"


def solve_original_LP(
    A: np.ndarray, b_vecs: np.ndarray, c_hat: np.ndarray
) -> np.ndarray:
    """
    Solve the base LP:
        min sum_k c_hat @ x_k
        s.t. A x_k >= b_k, x_k >= 0
    Returns the optimal x_hat with shape (K, n).
    """
    K, _ = b_vecs.shape
    m, n = A.shape
    model = gp.Model("original_lp")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0

    x = model.addMVar((K, n), lb=0.0, name="x")
    for k in range(K):
        model.addConstr(A @ x[k] >= b_vecs[k], name=f"Ax_ge_b_{k}")

    objective = gp.quicksum(c_hat @ x[k] for k in range(K))
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Original LP not optimal. Status: {model.Status}")
    return x.X


def solve_FACE(
    A: np.ndarray,
    b_vecs: np.ndarray,
    c_hat: np.ndarray,
    h: np.ndarray,
    x_hat: np.ndarray,
    alpha: float,
    beta: float,
    start_value: float,
    cutsize: float,
    stepsize: float,
    time_limit: int
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[float],
    Optional[float],
    float,
    Optional[float],
    int,
]:
    """
    Solve the FACE model with the given parameters.
    Returns X, c, v, v_ave, v_max, runtime, gap, status.
    Any of X/c/v/v_ave/v_max may be None if no feasible solution was found.
    """
    K, _ = b_vecs.shape
    m, n = A.shape
    # Build D_i from multiplicative tweaks around c_hat[i]:
    # {1 + 0.0016 * 5^l for l=0..4} U {1 - 0.0016 * 5^l for l=1..4}, scaled by c_hat[i]
    D_values = []
    plus_factors = [1 + start_value * (stepsize ** l) for l in range(cutsize//2)]
    minus_factors = [1 - start_value * (stepsize ** l) for l in range(cutsize//2)]
    factors = plus_factors + minus_factors + [1.0]
    for i in range(n):
        candidates = sorted({c_hat[i] * f for f in factors if c_hat[i] * f > 0})
        D_values.append(np.array(candidates, dtype=float))

    model = gp.Model("face")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.NumericFocus = 0
    model.Params.Seed = 0

    X = model.addMVar((K, n), lb=0.0, name="X")
    Y = model.addMVar((K, m), lb=0.0, name="Y")
    c = model.addMVar(n, lb=0.0, name="c")
    # Z and U are ragged per i, so use dicts keyed by (i, j_idx) and (k, i, j_idx)
    Z = {}
    U = {}
    for i, d_list in enumerate(D_values):
        for j_idx, _ in enumerate(d_list):
            Z[(i, j_idx)] = model.addVar(vtype=GRB.BINARY, name=f"Z_{i}_{j_idx}")
            for k in range(K):
                U[(k, i, j_idx)] = model.addVar(lb=0.0, name=f"U_{k}_{i}_{j_idx}")
    v = model.addMVar(K, lb=-GRB.INFINITY, name="v")
    v_ave = model.addVar(lb=-GRB.INFINITY, name="v_ave")
    v_max = model.addVar(lb=0.0, name="v_max")

    # Primal feasibility constraints
    for k in range(K):
        model.addConstr(A @ X[k] >= b_vecs[k], name=f"Ax_ge_b_{k}")

    # Dual feasibility constraints
    for k in range(K):
        model.addConstr(A.T @ Y[k] <= c, name=f"ATy_le_c_{k}")

    # Capacity constraint
    model.addConstr(X.sum(axis=0) <= h, name="capacity_h")

    # Linking c_i with z_{i,j}
    for i in range(n):
        model.addConstr(gp.quicksum(Z[(i, j_idx)] for j_idx in range(len(D_values[i]))) == 1, name=f"z_onehot_{i}")
        model.addConstr(
            c[i] == gp.quicksum(D_values[i][j_idx] * Z[(i, j_idx)] for j_idx in range(len(D_values[i]))),
            name=f"c_def_{i}",
        )

    # u_{k,i,j} linearization with x_{k,i} and z_{i,j}
    for k in range(K):
        for i in range(n):
            for j_idx in range(len(D_values[i])):
                model.addConstr(U[(k, i, j_idx)] <= h[i] * Z[(i, j_idx)], name=f"u_bigM_pos_{k}_{i}_{j_idx}")
                model.addConstr(U[(k, i, j_idx)] <= X[k, i], name=f"u_le_x_{k}_{i}_{j_idx}")
                model.addConstr(
                    U[(k, i, j_idx)] >= X[k, i] - h[i] * (1 - Z[(i, j_idx)]),
                    name=f"u_ge_x_bigM_{k}_{i}_{j_idx}",
                )

    # Duality and fairness-related expressions
    dual_exprs = []
    for k in range(K):
        expr = gp.quicksum(
            D_values[i][j_idx] * U[(k, i, j_idx)] for i in range(n) for j_idx in range(len(D_values[i]))
        )
        dual_exprs.append(expr)
        model.addConstr(b_vecs[k] @ Y[k] == expr, name=f"strong_duality_{k}")

    for k in range(K):
        denom = float(c_hat @ x_hat[k] + 1e-8)
        model.addConstr(v[k] == 100 * dual_exprs[k] / denom, name=f"v_def_{k}")
        model.addConstr(v_max >= v[k], name=f"vmax_ge_v_{k}")
    model.addConstr(v_ave == (1.0 / K) * gp.quicksum(v[k] for k in range(K)), name="v_average")

    price_term = gp.quicksum(10000 * ((c[i] - c_hat[i]) / c_hat[i]) * ((c[i] - c_hat[i]) / c_hat[i]) for i in range(n))
    fairness_term1 = gp.quicksum((v[k] - v_ave) * (v[k] - v_ave) for k in range(K))
    fairness_term2 = v_max * v_max
    objective = (1 - alpha - beta) * (1.0 / n) * price_term + alpha * (1.0 / K) * fairness_term1 + beta * fairness_term2
    model.setObjective(objective, GRB.MINIMIZE)

    model.optimize()

    runtime = model.Runtime
    status = model.Status
    gap = None
    if model.SolCount > 0:
        try:
            gap = model.MIPGap
        except gp.GurobiError:
            gap = 0.0
        return X.X, c.X, v.X, v_ave.X, v_max.X, runtime, gap, status

    return None, None, None, None, None, runtime, gap, status


def compute_metrics(
    c_sol: np.ndarray,
    X_sol: np.ndarray,
    c_hat: np.ndarray,
    x_hat: np.ndarray,
    alpha: float,
    beta: float,
) -> Tuple[np.ndarray, float, float, float, float, float, float, float]:
    K_local = X_sol.shape[0]
    n_local = c_hat.shape[0]
    denom_eps = 1e-8
    v_values = np.array(
        [
            100.0 * float(c_sol @ X_sol[k]) / float(c_hat @ x_hat[k] + denom_eps)
            for k in range(K_local)
        ],
        dtype=float,
    )
    v_ave = float(np.mean(v_values))
    v_max = float(max(0.0, np.max(v_values)))
    price_term = float(np.sum((100.0 * (c_sol - c_hat) / c_hat) ** 2))
    fairness_term1 = float(np.sum((v_values - v_ave) ** 2))
    fairness_term2 = float(v_max**2)
    G = (1.0 / n_local) * price_term + (1.0 / K_local) * fairness_term1 + fairness_term2
    objective_val = (
        (1 - alpha - beta) * (1.0 / n_local) * price_term
        + alpha * (1.0 / K_local) * fairness_term1
        + beta * fairness_term2
    )
    return v_values, v_ave, v_max, price_term, fairness_term1, fairness_term2, G, objective_val


def generate_random_matrices(
    rng: np.random.Generator, K: int, m: int, n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random A (m x n), b_vecs (K x m), and c_hat (length n).
    c_hat is drawn uniformly from [0.1, 1.0].
    """
    A = rng.uniform(low=0.1, high=1.0, size=(m, n))
    b_vecs = rng.uniform(low=0.1, high=1.0, size=(K, m))
    c_hat = rng.uniform(low=0.1, high=1.0, size=n)
    return A, b_vecs, c_hat


def build_h_vector(rng: np.random.Generator, x_hat: np.ndarray) -> np.ndarray:
    """
    Build h with capacity tied to x_hat:
    - choose n/3 indices among those with positive sum_k x_hat[k,i] and set h[i]=0.8*sum_k x_hat[k,i]
    - otherwise h[i]=10*K
    """
    K, n = x_hat.shape
    col_sums = np.sum(x_hat, axis=0)
    h = np.zeros(n, dtype=float)

    positive_indices = np.where(col_sums > 0)[0]
    target = n // 3
    chosen_count = min(len(positive_indices), target) if target > 0 else 0
    chosen = rng.choice(positive_indices, size=chosen_count, replace=False) if chosen_count > 0 else np.array([], dtype=int)

    chosen_set = set(int(i) for i in chosen)
    for i in range(n):
        if i in chosen_set:
            h[i] = 0.8 * col_sums[i]
        else:
            h[i] = 10.0 * K
    return h


def experiment_iterator() -> Iterable[Tuple[int, int, int]]:
    # k_values = [2, 10, 50]
    # n_values = [25, 50, 100]
    # rho_values = [0.5, 1.0, 2.0]
    k_values = [50]
    n_values = [100]
    rho_values = [1.0]

    for K in k_values:
        for n in n_values:
            m_candidates = set()
            for rho in rho_values:
                m_val = int(round(rho * n))
                m_candidates.add(m_val)
            for m in sorted(m_candidates):
                yield K, n, m


def run_experiments(
    output_csv: str,
    seed: int = SEED_DEFAULT,
    alpha_list: Sequence[float] = ALPHA_LIST_DEFAULT,
    beta_list: Sequence[float] = BETA_LIST_DEFAULT,
    time_limit: int = TIME_LIMIT_DEFAULT,
    start_value: float = start_value,
    cutsize: float = cutsize,
    stepsize: float = stepsize,
) -> None:
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed_data",
                "K",
                "n",
                "m",
                "alpha",
                "beta",
                "G",
                "price_term",
                "fairness_term1",
                "fairness_term2",
                "objective",
                "t",
                "gap",
                "status",
            ]
        )

        for seed_data in range(1):
            for K, n, m in experiment_iterator():
                # Independent RNG per (K, n, m, seed_data) for reproducibility
                rng = np.random.default_rng(np.random.SeedSequence([seed, K, n, m, seed_data]))
                A, b_vecs, c_hat = generate_random_matrices(rng, K, m, n)
                x_hat = solve_original_LP(A, b_vecs, c_hat)
                h = build_h_vector(rng, x_hat)

                for alpha in alpha_list:
                    for beta in beta_list:
                        if alpha + beta > 1:
                            continue
                        (
                            X_sol,
                            c_sol,
                            v_sol,
                            v_ave_sol,
                            v_max_sol,
                            runtime,
                            gap,
                            status,
                        ) = solve_FACE(A, b_vecs, c_hat, h, x_hat, alpha, beta, start_value=start_value, cutsize=cutsize, stepsize=stepsize,time_limit=time_limit)
                        if X_sol is None or c_sol is None or v_sol is None or v_ave_sol is None or v_max_sol is None:
                            writer.writerow(
                                [
                                    seed_data,
                                    K,
                                    n,
                                    m,
                                    alpha,
                                    beta,
                                    np.nan,
                                    np.nan,
                                    np.nan,
                                    np.nan,
                                    np.nan,
                                    runtime,
                                    gap,
                                    status_label(status),
                                ]
                            )
                            continue

                        (
                            v_values,
                            v_ave,
                            v_max,
                            price_term,
                            fairness_term1,
                            fairness_term2,
                            G,
                            objective_val,
                        ) = compute_metrics(c_sol, X_sol, c_hat, x_hat, alpha, beta)

                        writer.writerow(
                            [
                                seed_data,
                                K,
                                n,
                                m,
                                alpha,
                                beta,
                                G,
                                price_term,
                                fairness_term1,
                                fairness_term2,
                                objective_val,
                                runtime,
                                gap,
                                status_label(status),
                            ]
                        )
                        # Show completion of each experiment instance
                        print(
                            f"Done seed={seed_data}, K={K}, n={n}, m={m}, alpha={alpha}, beta={beta}, "
                            f"status={status_label(status)}, t={runtime:.2f}"
                        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FACE experiments from the requirement document.")
    parser.add_argument("--output", default="c_face_random_results.csv", help="Output CSV path.")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT, help="Random seed.")
    parser.add_argument("--time-limit", type=int, default=TIME_LIMIT_DEFAULT, help="Time limit for FACE model (seconds).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(
        output_csv=args.output,
        seed=args.seed,
        time_limit=args.time_limit,
    )


if __name__ == "__main__":
    main()

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


# Default parameters
SEED_LIST_DEFAULT = [0, 1, 2, 3, 4]
# SEED_LIST_DEFAULT = [1,2]
ALPHA_BETA_LIST_DEFAULT = [(0.0, 0.0), (0.25, 0.25), (0.5, 0.0), (0.0, 0.5)]
# ALPHA_BETA_LIST_DEFAULT = [(0.25, 0.25)]
TIME_LIMIT_DEFAULT = 60  # seconds
LAMBDA1 = 0.1
LAMBDA2 = 1.9


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
    time_limit: int = TIME_LIMIT_DEFAULT,
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

    model = gp.Model("face")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.NonConvex = 2  # required for bilinear terms c^T x_k
    model.Params.Seed = 0
    model.Params.NumericFocus = 0

    X = model.addMVar((K, n), lb=0.0, name="X")
    Y = model.addMVar((K, m), lb=0.0, name="Y")
    c = model.addMVar(n, lb=0.0, name="c")
    v = model.addMVar(K, lb=-GRB.INFINITY, name="v")
    v_ave = model.addVar(lb=-GRB.INFINITY, name="v_ave")
    v_max = model.addVar(lb=1.0, name="v_max")

    # Primal feasibility constraints
    for k in range(K):
        model.addConstr(A @ X[k] >= b_vecs[k], name=f"Ax_ge_b_{k}")

    # Dual feasibility constraints
    for k in range(K):
        model.addConstr(A.T @ Y[k] <= c, name=f"ATy_le_c_{k}")

    # Capacity constraint
    model.addConstr(X.sum(axis=0) <= h, name="capacity_h")

    # Bounds on c
    model.addConstr(c >= LAMBDA1 * c_hat, name="c_lower_bound")
    model.addConstr(c <= LAMBDA2 * c_hat, name="c_upper_bound")

    # Strong duality and fairness variables
    primal_cost_exprs = []
    for k in range(K):
        cost_expr = gp.quicksum(c[i] * X[k, i] for i in range(n))
        primal_cost_exprs.append(cost_expr)
        model.addConstr(b_vecs[k] @ Y[k] == cost_expr, name=f"strong_duality_{k}")

    for k in range(K):
        denom = float(c_hat @ x_hat[k]+1e-8)
        model.addConstr(v[k] == primal_cost_exprs[k] / denom, name=f"v_def_{k}")
        model.addConstr(v_max >= v[k], name=f"vmax_ge_v_{k}")
    model.addConstr(v_ave == (1.0 / K) * gp.quicksum(v[k] for k in range(K)), name="v_average")

    price_term = gp.quicksum(((c[i] - c_hat[i]) / c_hat[i]) * ((c[i] - c_hat[i]) / c_hat[i]) for i in range(n))
    fairness_term1 = gp.quicksum((v[k] - v_ave) * (v[k] - v_ave) for k in range(K))
    fairness_term2 = (v_max-1)**2
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
            float(c_sol @ X_sol[k]) / float(c_hat @ x_hat[k]+denom_eps)
            for k in range(K_local)
        ],
        dtype=float,
    )
    v_ave = float(np.mean(v_values))
    v_max = float(max(1.0, np.max(v_values)))
    price_term = float(np.sum(((c_sol - c_hat) / c_hat) ** 2)) ** 0.5
    fairness_term1 = float(np.sum((v_values - v_ave) ** 2)) ** 0.5
    fairness_term2 = float(v_max - 1)
    G = (1.0 / n_local) * price_term + (1.0 / K_local) * fairness_term1 + fairness_term2
    objective_val = (
        (1 - alpha - beta) * (1.0 / n_local) * (price_term ** 2)
        + alpha * (1.0 / K_local) * (fairness_term1 ** 2)
        + beta * (fairness_term2 ** 2)
    )
    return v_values, v_ave, v_max, price_term, fairness_term1, fairness_term2, G, objective_val


def generate_random_matrices(
    rng: np.random.Generator, K: int, m: int, n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random A (m x n), b_vecs (K x m), and c_hat (length n).
    c_hat is drawn uniformly from [1.0, 10.0].
    """
    A = rng.uniform(low=0.0, high=1.0, size=(m, n))
    b_vecs = rng.uniform(low=0.0, high=1.0, size=(K, m))
    c_hat = rng.uniform(low=1.0, high=10.0, size=n)
    return A, b_vecs, c_hat


def build_h_vector(rng: np.random.Generator, x_hat: np.ndarray, c_hat: np.ndarray) -> np.ndarray:
    """
    Build h with capacity tied to x_hat:
    - choose n/3 indices among positive-sum columns and set h[i]=0.8*sum_k x_hat[k,i]
    - otherwise h[i]=10*K
    """
    _ = c_hat  # kept for signature compatibility
    _, n = x_hat.shape
    col_sums = np.sum(x_hat, axis=0)
    h = np.zeros(n, dtype=float)

    positive_indices = np.where(col_sums > 0)[0]
    target = n // 3
    chosen_count = min(len(positive_indices), target) if target > 0 else 0
    chosen = (
        rng.choice(positive_indices, size=chosen_count, replace=False)
        if chosen_count > 0
        else np.array([], dtype=int)
    )
    chosen_set = set(int(i) for i in chosen)

    s_max = float(np.max(col_sums)) if col_sums.size > 0 else 0.0
    for i in range(n):
        if i in chosen_set:
            h[i] = 0.8 * col_sums[i]
        else:
            h[i] = 5.0 * s_max
    return h


K_LIST = [2, 4, 8]
N_LIST = [5, 10, 20]
M_LIST = [5, 10, 20]


def experiment_iterator() -> Iterable[Tuple[int, int, int]]:
    for K in K_LIST:
        for m in M_LIST:
            for n in N_LIST:
                yield K, m, n


def parse_only_list(only: Optional[Sequence[str]]) -> Optional[Sequence[Tuple[int, int, int]]]:
    if not only:
        return None
    parsed: list[Tuple[int, int, int]] = []
    for item in only:
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 3:
            raise ValueError(f"--only must be 'K,m,n' format: {item}")
        K, m, n = (int(p) for p in parts)
        parsed.append((K, m, n))
    return parsed


def parse_alpha_beta_list(
    alpha_beta: Optional[Sequence[str]],
) -> Optional[Sequence[Tuple[float, float]]]:
    if not alpha_beta:
        return None
    parsed: list[Tuple[float, float]] = []
    for item in alpha_beta:
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 2:
            raise ValueError(f"--alpha-beta must be 'alpha,beta' format: {item}")
        alpha, beta = (float(p) for p in parts)
        parsed.append((alpha, beta))
    return parsed


def run_experiments(
    output_csv: str,
    seed_list: Sequence[int] = SEED_LIST_DEFAULT,
    alpha_beta_list: Sequence[Tuple[float, float]] = ALPHA_BETA_LIST_DEFAULT,
    time_limit: int = TIME_LIMIT_DEFAULT,
    only_list: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> None:
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "K",
                "m",
                "n",
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
        experiment_items = (
            list(only_list) if only_list is not None else list(experiment_iterator())
        )
        for seed_value in seed_list:
            for K, m, n in experiment_items:
                rng = np.random.default_rng(np.random.SeedSequence([seed_value, K, n, m]))
                A, b_vecs, c_hat = generate_random_matrices(rng, K, m, n)
                x_hat = solve_original_LP(A, b_vecs, c_hat)
                h = build_h_vector(rng, x_hat, c_hat)

                for alpha, beta in alpha_beta_list:
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
                    ) = solve_FACE(A, b_vecs, c_hat, h, x_hat, alpha, beta, time_limit=time_limit)
                    if X_sol is None or c_sol is None or v_sol is None or v_ave_sol is None or v_max_sol is None:
                        writer.writerow(
                            [
                                seed_value,
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
                            seed_value,
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
                    print(
                        f"Done seed={seed_value}, K={K}, n={n}, m={m}, alpha={alpha}, beta={beta}, "
                        f"status={status_label(status)}, t={runtime:.2f}"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FACE experiments from the requirement document.")
    parser.add_argument("--output", default="face_random_results.csv", help="Output CSV path.")
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Override seeds (repeatable). If omitted, uses defaults.",
    )
    parser.add_argument("--time-limit", type=int, default=TIME_LIMIT_DEFAULT, help="Time limit for FACE model (seconds).")
    parser.add_argument(
        "--only",
        action="append",
        help="Run only a specific tuple in 'K,m,n' format. Can be repeated.",
    )
    parser.add_argument(
        "--alpha-beta",
        action="append",
        help="Run only specific alpha,beta pairs in 'alpha,beta' format. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(
        output_csv=args.output,
        time_limit=args.time_limit,
        only_list=parse_only_list(args.only),
        alpha_beta_list=parse_alpha_beta_list(args.alpha_beta) or ALPHA_BETA_LIST_DEFAULT,
        seed_list=args.seed or SEED_LIST_DEFAULT,
    )


if __name__ == "__main__":
    main()

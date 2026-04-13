import argparse
import csv
from typing import Iterable, Optional, Sequence, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB


SEED_LIST_DEFAULT = [0, 1, 2, 3, 4]
# SEED_LIST_DEFAULT = [0,1,2]
ALPHA_BETA_LIST_DEFAULT = [(0.0, 0.0), (0.25, 0.25), (0.5, 0.0), (0.0, 0.5)]
# ALPHA_BETA_LIST_DEFAULT = [(0.25, 0.25)]
LIMIT_1 = 60
LIMIT_2 = 60
N_LIST = [5, 10, 20]
M_LIST = [5, 10, 20]

K_LIST = [2,4,8]

L_1 = 5
R_1 = 0.9 / L_1
L_2 = 5
R_2 = R_1 / (L_2)
LAMBDA1 = 0.1
LAMBDA2 = 1.9

GAMMA_LIST_1 = [0.0]
for l in range(1, L_1 + 1):
    GAMMA_LIST_1.append(R_1 * l)
    GAMMA_LIST_1.append(-R_1 * l)
J_1 = len(GAMMA_LIST_1)

GAMMA_LIST_2 = [0.0]
for l in range(1, L_2 + 1):
    GAMMA_LIST_2.append(R_2 * l)
    GAMMA_LIST_2.append(-R_2 * l)
J_2 = len(GAMMA_LIST_2)


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


def size_label(n: int) -> str:
    if n == 5:
        return "small"
    if n == 25:
        return "middle"
    return "big"


def solve_original_LP(A: np.ndarray, b_vecs: np.ndarray, c_hat: np.ndarray) -> np.ndarray:
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


def generate_random_matrices(
    rng: np.random.Generator, K: int, m: int, n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = rng.uniform(low=0.0, high=1.0, size=(m, n))
    b_vecs = rng.uniform(low=0.0, high=1.0, size=(K, m))
    c_hat = rng.uniform(low=1.0, high=10.0, size=n)
    return A, b_vecs, c_hat


def build_h_vector(rng: np.random.Generator, x_hat: np.ndarray) -> np.ndarray:
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
            h[i] = 20.0 * s_max
    return h


def solve_dd_face(
    A: np.ndarray,
    b_vecs: np.ndarray,
    c_hat: np.ndarray,
    h: np.ndarray,
    x_hat: np.ndarray,
    c_bar: np.ndarray,
    alpha: float,
    beta: float,
    time_limit: int,
    numeric_focus: int,
    mip_focus: int,
    heuristics: float,
    start_solution: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    iis_path: Optional[str] = None,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[float],
    Optional[float],
    float,
    Optional[float],
    Optional[float],
    int,
]:
    K, _ = b_vecs.shape
    m, n = A.shape
    _, J_local = c_bar.shape

    model = gp.Model("dd_face")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.Seed = 0
    model.Params.NumericFocus = numeric_focus
    model.Params.NonConvex = 2
    model.Params.Heuristics = heuristics
    model.Params.MIPFocus = mip_focus


    X = model.addMVar((K, n), lb=0.0, name="X")
    Y = model.addMVar((K, m), lb=0.0, name="Y")
    c = model.addMVar(n, lb=-GRB.INFINITY, name="c")
    z = model.addMVar((n, J_local), vtype=GRB.BINARY, name="z")
    v = model.addMVar(K, lb=-GRB.INFINITY, name="v")
    v_ave = model.addVar(lb=-GRB.INFINITY, name="v_ave")
    v_max = model.addVar(lb=1.0, name="v_max")

    if start_solution is not None:
        X_start, Y_start, c_start = start_solution
        X.Start = X_start
        Y.Start = Y_start
        c.Start = c_start

    for k in range(K):
        model.addConstr(A @ X[k] >= b_vecs[k], name=f"Ax_ge_b_{k}")
    for k in range(K):
        model.addConstr(A.T @ Y[k] <= c, name=f"ATy_le_c_{k}")
    model.addConstr(X.sum(axis=0) <= h, name="capacity_h")

    for i in range(n):
        model.addConstr(z[i].sum() == 1, name=f"z_onehot_{i}")
        model.addConstr(
            c[i] == gp.quicksum(c_bar[i, j] * z[i, j] for j in range(J_local)),
            name=f"c_def_{i}",
        )
    lower = LAMBDA1 * c_hat
    upper = LAMBDA2 * c_hat
    for i in range(n):
        for j in range(J_local):
            if c_bar[i, j] < lower[i] or c_bar[i, j] > upper[i]:
                model.addConstr(z[i, j] == 0, name=f"z_fix_{i}_{j}")

    cost_exprs = []
    for k in range(K):
        expr = gp.quicksum(c[i] * X[k, i] for i in range(n))
        cost_exprs.append(expr)
        model.addConstr(b_vecs[k] @ Y[k] == expr, name=f"strong_duality_{k}")

    for k in range(K):
        denom = float(c_hat @ x_hat[k] + 1e-8)
        model.addConstr(v[k] == cost_exprs[k] / denom, name=f"v_def_{k}")
        model.addConstr(v_max >= v[k], name=f"vmax_ge_v_{k}")
    model.addConstr(v_ave == (1.0 / K) * gp.quicksum(v[k] for k in range(K)), name="v_average")

    price_term = gp.quicksum(
        ((c[i] - c_hat[i]) / c_hat[i]) * ((c[i] - c_hat[i]) / c_hat[i])
        for i in range(n)
    )
    fairness_term1 = gp.quicksum((v[k] - v_ave) * (v[k] - v_ave) for k in range(K))
    fairness_term2 = (v_max - 1) * (v_max - 1)
    objective = (
        (1 - alpha - beta) * (1.0 / n) * price_term
        + alpha * (1.0 / K) * fairness_term1
        + beta * fairness_term2
    )
    model.setObjective(objective, GRB.MINIMIZE)

    first_feas_time: list[Optional[float]] = [None]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and first_feas_time[0] is None:
            first_feas_time[0] = model.cbGet(GRB.Callback.RUNTIME)

    model.optimize(cb)

    runtime = model.Runtime
    status = model.Status
    if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) and iis_path:
        if status == GRB.INF_OR_UNBD:
            model.Params.InfUnbdInfo = 1
            model.optimize()
            status = model.Status
        if status == GRB.INFEASIBLE:
            model.computeIIS()
            model.write(iis_path)

    gap = None
    if model.SolCount > 0:
        try:
            gap = model.MIPGap
        except gp.GurobiError:
            gap = 0.0
        return X.X, Y.X, c.X, z.X, v.X, v_ave.X, v_max.X, runtime, first_feas_time[0], gap, status
    return None, None, None, None, None, None, None, runtime, first_feas_time[0], gap, status


def compute_metrics(
    c_sol: np.ndarray,
    X_sol: np.ndarray,
    c_hat: np.ndarray,
    x_hat: np.ndarray,
    alpha: float,
    beta: float,
) -> Tuple[float, float, float, float, float, float]:
    K_local = X_sol.shape[0]
    n_local = c_hat.shape[0]
    denom_eps = 1e-8
    v_values = np.array(
        [
            float(c_sol @ X_sol[k]) / float(c_hat @ x_hat[k] + denom_eps)
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
    return v_max, price_term, fairness_term1, fairness_term2, G, objective_val


def build_c_bar_1(c_hat: np.ndarray) -> np.ndarray:
    n = c_hat.shape[0]
    c_bar = np.zeros((n, J_1), dtype=float)
    for j, gamma in enumerate(GAMMA_LIST_1):
        c_bar[:, j] = (1.0 + gamma) * c_hat
    return c_bar


def build_c_bar_2(c_star: np.ndarray, c_hat: np.ndarray) -> np.ndarray:
    n = c_hat.shape[0]
    c_bar = np.zeros((n, J_2), dtype=float)
    for j, gamma in enumerate(GAMMA_LIST_2):
        c_bar[:, j] = c_star + gamma * c_hat
    return c_bar


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
    only_list: Optional[Sequence[Tuple[int, int, int]]] = None,
    alpha_beta_list: Sequence[Tuple[float, float]] = ALPHA_BETA_LIST_DEFAULT,
    write_iis: bool = False,
) -> None:
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "K",
                "m",
                "n",
                "size",
                "alpha",
                "beta",
                "objective",
                "dd_face1_time",
                "dd_face1_first_feas",
                "dd_face1_gap",
                "dd_face1_status",
                "dd_face2_time",
                "dd_face2_first_feas",
                "dd_face2_gap",
                "dd_face2_status",
                "G",
                "price_term",
                "fairness_term1",
                "fairness_term2",
            ]
        )
        if only_list:
            experiment_items = list(only_list)
        else:
            experiment_items = list(experiment_iterator())
        for seed_value in seed_list:
            for K, m, n in experiment_items:
                size = size_label(n)
                rng = np.random.default_rng(np.random.SeedSequence([seed_value, K, n, m]))
                A, b_vecs, c_hat = generate_random_matrices(rng, K, m, n)
                x_hat = solve_original_LP(A, b_vecs, c_hat)
                h = build_h_vector(rng, x_hat)

                for alpha, beta in alpha_beta_list:
                    c_bar_1 = build_c_bar_1(c_hat)
                    iis_path_1 = None
                    if write_iis:
                        iis_path_1 = f"dd_face1_iis_seed{seed_value}_K{K}_m{m}_n{n}.ilp"
                    (
                        X_1,
                        Y_1,
                        c_1,
                        _,
                        v_1,
                        _,
                        _,
                        dd1_runtime,
                        dd1_first_feas,
                        dd1_gap,
                        dd1_status,
                    ) = solve_dd_face(
                        A,
                        b_vecs,
                        c_hat,
                        h,
                        x_hat,
                        c_bar_1,
                        alpha,
                        beta,
                        LIMIT_1,
                        numeric_focus=1,
                        mip_focus=3,
                        heuristics=0.5,
                        iis_path=iis_path_1,
                    )
                    print(
                        "D-D-FACEP_1 Done "
                        f"seed={seed_value}, K={K}, n={n}, m={m}, alpha={alpha}, beta={beta},"
                        f"status={status_label(dd1_status)},t={dd1_runtime:.2f},"
                        f"first_feas={dd1_first_feas}"
                    )

                    if X_1 is None or Y_1 is None or c_1 is None or v_1 is None:
                        writer.writerow(
                            [
                                seed_value,
                                K,
                                m,
                                n,
                                size,
                                alpha,
                                beta,
                                np.nan,
                                dd1_runtime,
                                dd1_first_feas,
                                dd1_gap,
                                status_label(dd1_status),
                                np.nan,
                                np.nan,
                                np.nan,
                                "",
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                            ]
                        )
                        continue

                    c_bar_2 = build_c_bar_2(c_1, c_hat)
                    iis_path_2 = None
                    if write_iis:
                        iis_path_2 = f"dd_face2_iis_seed{seed_value}_K{K}_m{m}_n{n}.ilp"
                    (
                        X_2,
                        Y_2,
                        c_2,
                        _,
                        v_2,
                        _,
                        _,
                        dd2_runtime,
                        dd2_first_feas,
                        dd2_gap,
                        dd2_status,
                    ) = solve_dd_face(
                        A,
                        b_vecs,
                        c_hat,
                        h,
                        x_hat,
                        c_bar_2,
                        alpha,
                        beta,
                        LIMIT_2,
                        numeric_focus=2,
                        mip_focus=0,
                        heuristics=0.0,
                        start_solution=(X_1, Y_1, c_1),
                        iis_path=iis_path_2,
                    )
                    print(
                        "D-D-FACEP_2 Done "
                        f"seed={seed_value}, K={K}, n={n}, m={m}, alpha={alpha}, beta={beta},"
                        f"status={status_label(dd2_status)},t={dd2_runtime:.2f},"
                        f"first_feas={dd2_first_feas}"
                    )

                    if X_2 is None or c_2 is None:
                        writer.writerow(
                            [
                                seed_value,
                                K,
                                m,
                                n,
                                size,
                                alpha,
                                beta,
                                np.nan,
                                dd1_runtime,
                                dd1_first_feas,
                                dd1_gap,
                                status_label(dd1_status),
                                dd2_runtime,
                                dd2_first_feas,
                                dd2_gap,
                                status_label(dd2_status),
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                            ]
                        )
                        continue

                    (_, price_term, fairness_term1, fairness_term2, G, objective_val) = compute_metrics(
                        c_2, X_2, c_hat, x_hat, alpha, beta
                    )
                    writer.writerow(
                        [
                            seed_value,
                            K,
                            m,
                            n,
                            size,
                            alpha,
                            beta,
                            objective_val,
                            dd1_runtime,
                            dd1_first_feas,
                            dd1_gap,
                            status_label(dd1_status),
                            dd2_runtime,
                            dd2_first_feas,
                            dd2_gap,
                            status_label(dd2_status),
                            G,
                            price_term,
                            fairness_term1,
                            fairness_term2,
                        ]
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D-D-FACE (discrete-discrete) experiments.")
    parser.add_argument("--output", default="random_D_D.csv", help="Output CSV path.")
    parser.add_argument(
        "--only",
        action="append",
        help="Run only a specific tuple in 'K,m,n' format. Can be repeated.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Override seeds (repeatable). If omitted, uses defaults.",
    )
    parser.add_argument(
        "--alpha-beta",
        action="append",
        help="Run only specific alpha,beta pairs in 'alpha,beta' format. Can be repeated.",
    )
    parser.add_argument(
        "--iis",
        action="store_true",
        help="Write IIS for infeasible D-D-FACE runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    only_list = parse_only_list(args.only)
    run_experiments(
        output_csv=args.output,
        seed_list=args.seed or SEED_LIST_DEFAULT,
        only_list=only_list,
        alpha_beta_list=parse_alpha_beta_list(args.alpha_beta) or ALPHA_BETA_LIST_DEFAULT,
        write_iis=args.iis,
    )


if __name__ == "__main__":
    main()

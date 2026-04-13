import argparse
import csv
from typing import Iterable, Optional, Sequence, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB


A = np.array(
    [
        [392, 343, 142, 133, 127],  # energy
        [6.1, 12.6, 12.2, 21.3, 22.5],  # protein
        [0.9, 1.7, 10.2, 5.9, 4.5],  # fat
    ]
)
b1 = np.array([8000, 415, 213.6])
b2 = np.array([7040, 600, 192.8])
c_hat = np.array([50, 40, 90, 70, 100])
h = np.array([100, 100, 20, 100, 100])

N = len(c_hat)
M = A.shape[0]
FOODS = ["米", "小麦", "卵", "鶏肉", "魚"]
K = 2

LAMBDA1 = 0.1
LAMBDA2 = 1.9

L_1 = 5
R_1 = 0.9 / L_1
L_2 = 5
R_2 = R_1 / (L_2)

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

ALPHA_BETA_PAIRS = [(0.0, 0.0), (0.25, 0.25), (0.0, 0.5), (0.5, 0.0)]
# ALPHA_BETA_PAIRS = [(0.0, 0.0)]

LIMIT_1 = 15
LIMIT_2 = 15


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


def solve_original_LP(A_mat: np.ndarray, b_vecs: np.ndarray, c_vec: np.ndarray) -> np.ndarray:
    K_local, _ = b_vecs.shape
    _, n_local = A_mat.shape
    model = gp.Model("original_lp")
    model.Params.OutputFlag = 0
    model.Params.Seed = 0

    x = model.addMVar((K_local, n_local), lb=0.0, name="x")
    for k in range(K_local):
        model.addConstr(A_mat @ x[k] >= b_vecs[k], name=f"Ax_ge_b_{k}")

    objective = gp.quicksum(c_vec @ x[k] for k in range(K_local))
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Original LP not optimal. Status: {model.Status}")
    return x.X


def build_c_bar_1(c_vec: np.ndarray) -> np.ndarray:
    n_local = c_vec.shape[0]
    c_bar = np.zeros((n_local, J_1), dtype=float)
    for j, gamma in enumerate(GAMMA_LIST_1):
        c_bar[:, j] = (1.0 + gamma) * c_vec
    return c_bar


def build_c_bar_2(c_star: np.ndarray, c_vec: np.ndarray) -> np.ndarray:
    n_local = c_vec.shape[0]
    c_bar = np.zeros((n_local, J_2), dtype=float)
    for j, gamma in enumerate(GAMMA_LIST_2):
        c_bar[:, j] = c_star + gamma * c_vec
    return c_bar


def solve_dd_face(
    A_mat: np.ndarray,
    b_vecs: np.ndarray,
    c_vec: np.ndarray,
    h_vec: np.ndarray,
    x_hat: np.ndarray,
    c_bar: np.ndarray,
    alpha: float,
    beta: float,
    time_limit: int,
    start_solution: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[float],
    Optional[float],
    float,
    Optional[float],
    int,
]:
    K_local, _ = b_vecs.shape
    m_local, n_local = A_mat.shape
    _, J_local = c_bar.shape

    model = gp.Model("dd_face")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.Seed = 0
    model.Params.NumericFocus = 1
    model.Params.NonConvex = 2

    X = model.addMVar((K_local, n_local), lb=0.0, name="X")
    Y = model.addMVar((K_local, m_local), lb=0.0, name="Y")
    c = model.addMVar(n_local, lb=-GRB.INFINITY, name="c")
    z = model.addMVar((n_local, J_local), vtype=GRB.BINARY, name="z")
    v = model.addMVar(K_local, lb=-GRB.INFINITY, name="v")
    v_ave = model.addVar(lb=-GRB.INFINITY, name="v_ave")
    v_max = model.addVar(lb=1.0, name="v_max")

    if start_solution is not None:
        X_start, Y_start, c_start = start_solution
        X.Start = X_start
        Y.Start = Y_start
        c.Start = c_start

    for k in range(K_local):
        model.addConstr(A_mat @ X[k] >= b_vecs[k], name=f"Ax_ge_b_{k}")
    for k in range(K_local):
        model.addConstr(A_mat.T @ Y[k] <= c, name=f"ATy_le_c_{k}")
    model.addConstr(X.sum(axis=0) <= h_vec, name="capacity_h")

    for i in range(n_local):
        model.addConstr(z[i].sum() == 1, name=f"z_onehot_{i}")
        model.addConstr(
            c[i] == gp.quicksum(c_bar[i, j] * z[i, j] for j in range(J_local)),
            name=f"c_def_{i}",
        )
    model.addConstr(c >= LAMBDA1 * c_vec, name="c_lower_bound")
    model.addConstr(c <= LAMBDA2 * c_vec, name="c_upper_bound")

    cost_exprs = []
    for k in range(K_local):
        expr = gp.quicksum(c[i] * X[k, i] for i in range(n_local))
        cost_exprs.append(expr)
        model.addConstr(b_vecs[k] @ Y[k] == expr, name=f"strong_duality_{k}")

    for k in range(K_local):
        denom = float(c_vec @ x_hat[k])
        model.addConstr(v[k] ==cost_exprs[k] / denom, name=f"v_def_{k}")
        model.addConstr(v_max >= v[k], name=f"vmax_ge_v_{k}")
    model.addConstr(
        v_ave == (1.0 / K_local) * gp.quicksum(v[k] for k in range(K_local)),
        name="v_average",
    )

    price_term = gp.quicksum(
        ((c[i] - c_vec[i]) / c_vec[i])
        * ((c[i] - c_vec[i]) / c_vec[i])
        for i in range(n_local)
    )
    fairness_term1 = gp.quicksum((v[k] - v_ave) * (v[k] - v_ave) for k in range(K_local))
    fairness_term2 = (v_max - 1) ** 2
    objective = (
        (1 - alpha - beta) * (1.0 / n_local) * price_term
        + alpha * (1.0 / K_local) * fairness_term1
        + beta * fairness_term2
    )
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
        return X.X, Y.X, c.X, v.X, v_ave.X, v_max.X, runtime, gap, status
    return None, None, None, None, None, None, runtime, gap, status


def compute_metrics(
    c_sol: np.ndarray,
    X_sol: np.ndarray,
    c_vec: np.ndarray,
    x_hat: np.ndarray,
    alpha: float,
    beta: float,
) -> Tuple[float, float, float, float, float, float]:
    K_local = X_sol.shape[0]
    n_local = c_vec.shape[0]
    v_values = np.array(
        [
            float(c_sol @ X_sol[k]) / float(c_vec @ x_hat[k])
            for k in range(K_local)
        ],
        dtype=float,
    )
    v_ave = float(np.mean(v_values))
    v_max = float(max(1.0, np.max(v_values)))
    price_term = ((1/n_local) * float(np.sum(((c_sol - c_vec) / c_vec) ** 2))) ** 0.5
    fairness_term1 = (float(np.sum((v_values - v_ave) ** 2)) / K_local) ** 0.5
    fairness_term2 = float(v_max - 1.0)
    G = price_term + fairness_term1 + fairness_term2
    objective_val = (
        (1 - alpha - beta) * (price_term ** 2)
        + alpha * (fairness_term1 ** 2)
        + beta * (fairness_term2 ** 2)
    )
    return price_term, fairness_term1, fairness_term2, G, objective_val, v_ave


def parse_only_list(only: Optional[Sequence[str]]) -> Optional[Sequence[Tuple[float, float]]]:
    if not only:
        return None
    parsed: list[Tuple[float, float]] = []
    for item in only:
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 2:
            raise ValueError(f"--only must be 'alpha,beta' format: {item}")
        alpha, beta = (float(p) for p in parts)
        parsed.append((alpha, beta))
    return parsed


def experiment_iterator() -> Iterable[Tuple[float, float]]:
    for alpha, beta in ALPHA_BETA_PAIRS:
        yield alpha, beta


def run_experiments(
    output_csv: str,
    only_list: Optional[Sequence[Tuple[float, float]]] = None,
    limit_1: int = LIMIT_1,
    limit_2: int = LIMIT_2,
) -> None:
    b_vecs = np.stack([b1, b2], axis=0)
    x_hat = solve_original_LP(A, b_vecs, c_hat)
    c_hat_x_hat = [float(c_hat @ x_hat[k]) for k in range(x_hat.shape[0])]
    print(
        "c_hat@x_hat[1]={:.6f}, c_hat@x_hat[2]={:.6f}".format(
            c_hat_x_hat[0], c_hat_x_hat[1]
        )
    )

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "alpha",
                "beta",
                "objective",
                "dd_face1_time",
                "dd_face1_gap",
                "dd_face1_status",
                "dd_face2_time",
                "dd_face2_gap",
                "dd_face2_status",
                "G",
                "price_term",
                "fairness_term1",
                "fairness_term2",
            ]
        )

        experiment_items = only_list if only_list is not None else experiment_iterator()
        for alpha, beta in experiment_items:
            c_bar_1 = build_c_bar_1(c_hat)
            (
                X_1,
                Y_1,
                c_1,
                _,
                _,
                _,
                dd1_runtime,
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
                limit_1,
            )
            print(
                "D-D-FACEP_1 Done "
                f"alpha={alpha}, beta={beta}, status={status_label(dd1_status)}, t={dd1_runtime:.2f}"
            )

            if X_1 is None or c_1 is None:
                writer.writerow(
                    [
                        alpha,
                        beta,
                        np.nan,
                        dd1_runtime,
                        dd1_gap,
                        status_label(dd1_status),
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
            (
                X_2,
                Y_2,
                c_2,
                _,
                _,
                _,
                dd2_runtime,
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
                limit_2,
                start_solution=(X_1, Y_1, c_1),
            )
            print(
                "D-D-FACEP_2 Done "
                f"alpha={alpha}, beta={beta}, status={status_label(dd2_status)}, t={dd2_runtime:.2f}"
            )

            if X_2 is None or c_2 is None:
                writer.writerow(
                    [
                        alpha,
                        beta,
                        np.nan,
                        dd1_runtime,
                        dd1_gap,
                        status_label(dd1_status),
                        dd2_runtime,
                        dd2_gap,
                        status_label(dd2_status),
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ]
                )
                continue

            price_term, fairness_term1, fairness_term2, G, objective_val, _ = compute_metrics(
                c_2, X_2, c_hat, x_hat, alpha, beta
            )
            c_x = [float(c_2 @ X_2[k]) for k in range(X_2.shape[0])]
            print(
                "c@x[1]={:.6f}, c@x[2]={:.6f}".format(c_x[0], c_x[1])
            )
            x_opt = solve_original_LP(A, b_vecs, c_2)
            c_x_opt = [float(c_2 @ x_opt[k]) for k in range(x_opt.shape[0])]
            for k in range(x_opt.shape[0]):
                print(f"original_LP x[{k+1}]={x_opt[k]}")
                print(f"original_LP c@x[{k+1}]={c_x_opt[k]:.6f}")
            writer.writerow(
                [
                    alpha,
                    beta,
                    objective_val,
                    dd1_runtime,
                    dd1_gap,
                    status_label(dd1_status),
                    dd2_runtime,
                    dd2_gap,
                    status_label(dd2_status),
                    G,
                    price_term,
                    fairness_term1,
                    fairness_term2,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-data D-D-FACE (discrete-discrete) experiments.")
    parser.add_argument("--output", default="real_D_D.csv", help="Output CSV path.")
    parser.add_argument(
        "--only",
        action="append",
        help="Run only a specific pair in 'alpha,beta' format. Can be repeated.",
    )
    parser.add_argument("--limit1", type=int, default=LIMIT_1, help="Time limit for D-D-FACE_1 (seconds).")
    parser.add_argument("--limit2", type=int, default=LIMIT_2, help="Time limit for D-D-FACE_2 (seconds).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(
        output_csv=args.output,
        only_list=parse_only_list(args.only),
        limit_1=args.limit1,
        limit_2=args.limit2,
    )


if __name__ == "__main__":
    main()

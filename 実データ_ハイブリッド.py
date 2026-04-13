import argparse
import csv
from typing import Optional, Sequence, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from 実データ_2段階 import (
    A,
    ALPHA_BETA_PAIRS,
    LAMBDA1,
    LAMBDA2,
    LIMIT_1,
    LIMIT_2,
    b1,
    b2,
    build_c_bar_1,
    build_c_bar_2,
    c_hat,
    compute_metrics,
    h,
    solve_dd_face,
    solve_original_LP,
    status_label,
)


HYBRID_LIMIT = 5
FEAS_TOL = 1e-6


def solve_hybrid_face(
    A_mat: np.ndarray,
    b_vecs: np.ndarray,
    c_vec: np.ndarray,
    h_vec: np.ndarray,
    x_hat: np.ndarray,
    X_fixed: np.ndarray,
    alpha: float,
    beta: float,
    time_limit: int = HYBRID_LIMIT,
    start_solution: Optional[Tuple[np.ndarray, np.ndarray]] = None,
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
    K_local, _ = b_vecs.shape
    m_local, n_local = A_mat.shape

    model = gp.Model("hybrid_face")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.NonConvex = 2

    Y = model.addMVar((K_local, m_local), lb=0.0, name="Y")
    c = model.addMVar(n_local, lb=-GRB.INFINITY, name="c")
    v = model.addMVar(K_local, lb=-GRB.INFINITY, name="v")
    v_ave = model.addVar(lb=-GRB.INFINITY, name="v_ave")
    v_max = model.addVar(lb=1.0, name="v_max")

    if start_solution is not None:
        Y_start, c_start = start_solution
        Y.Start = Y_start
        c.Start = c_start

    if X_fixed.shape != (K_local, n_local):
        raise ValueError(f"X_fixed must be shape {(K_local, n_local)}, got {X_fixed.shape}")

    lhs = A_mat @ X_fixed.T
    if np.max(b_vecs - lhs.T) > FEAS_TOL:
        return None, None, None, None, None, 0.0, None, GRB.INFEASIBLE
    if np.max(X_fixed.sum(axis=0) - h_vec) > FEAS_TOL:
        return None, None, None, None, None, 0.0, None, GRB.INFEASIBLE

    for k in range(K_local):
        model.addConstr(A_mat.T @ Y[k] <= c, name=f"ATy_le_c_{k}")

    model.addConstr(c >= LAMBDA1 * c_vec, name="c_lower_bound")
    model.addConstr(c <= LAMBDA2 * c_vec, name="c_upper_bound")

    cost_exprs = []
    for k in range(K_local):
        expr = gp.quicksum(c[i] * X_fixed[k, i] for i in range(n_local))
        cost_exprs.append(expr)
        model.addConstr(b_vecs[k] @ Y[k] == expr, name=f"strong_duality_{k}")

    for k in range(K_local):
        denom = float(c_vec @ x_hat[k])
        model.addConstr(v[k] == cost_exprs[k] / denom, name=f"v_def_{k}")
        model.addConstr(v_max >= v[k], name=f"vmax_ge_v_{k}")
    model.addConstr(
        v_ave == (1.0 / K_local) * gp.quicksum(v[k] for k in range(K_local)),
        name="v_average",
    )

    price_term = gp.quicksum(
        ((c[i] - c_vec[i]) / c_vec[i]) * ((c[i] - c_vec[i]) / c_vec[i])
        for i in range(n_local)
    )
    fairness_term1 = gp.quicksum((v[k] - v_ave) * (v[k] - v_ave) for k in range(K_local))
    fairness_term2 = (v_max-1) ** 2
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
        except (gp.GurobiError, AttributeError):
            gap = 0.0
        return X_fixed, c.X, v.X, v_ave.X, v_max.X, runtime, gap, status

    return None, None, None, None, None, runtime, gap, status


def run_experiments(
    output_csv: str,
    only_list: Optional[Sequence[Tuple[float, float]]] = None,
    limit_1: int = LIMIT_1,
    limit_2: int = LIMIT_2,
    hybrid_limit: int = HYBRID_LIMIT,
) -> None:
    b_vecs = np.stack([b1, b2], axis=0)
    x_hat = solve_original_LP(A, b_vecs, c_hat)
    K_local = b_vecs.shape[0]
    n_local = c_hat.shape[0]
    m_local = A.shape[0]
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
                "dd_face1_time",
                "dd_face1_gap",
                "dd_face1_status",
                "dd_face2_time",
                "dd_face2_gap",
                "dd_face2_status",
            ]
        )

        experiment_items = only_list if only_list is not None else ALPHA_BETA_PAIRS
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
                        K_local,
                        n_local,
                        m_local,
                        alpha,
                        beta,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        "",
                        dd1_runtime,
                        dd1_gap,
                        status_label(dd1_status),
                        np.nan,
                        np.nan,
                        "",
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
                        K_local,
                        n_local,
                        m_local,
                        alpha,
                        beta,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        "",
                        dd1_runtime,
                        dd1_gap,
                        status_label(dd1_status),
                        dd2_runtime,
                        dd2_gap,
                        status_label(dd2_status),
                    ]
                )
                continue

            (
                X_h,
                c_h,
                v_h,
                v_ave_h,
                v_max_h,
                h_runtime,
                h_gap,
                h_status,
            ) = solve_hybrid_face(
                A,
                b_vecs,
                c_hat,
                h,
                x_hat,
                X_2,
                alpha,
                beta,
                time_limit=hybrid_limit,
            )

            print(f"alpha: {alpha}, beta: {beta}")
            if X_h is None or c_h is None or v_h is None or v_ave_h is None or v_max_h is None:
                print("No feasible solution found.")
                print(f"計算時間: {h_runtime:.2f}s")
                print(f"gap: {h_gap}")
                writer.writerow(
                    [
                        K_local,
                        n_local,
                        m_local,
                        alpha,
                        beta,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        h_runtime,
                        h_gap,
                        status_label(h_status),
                        dd1_runtime,
                        dd1_gap,
                        status_label(dd1_status),
                        dd2_runtime,
                        dd2_gap,
                        status_label(dd2_status),
                    ]
                )
                continue

            price_term, fairness_term1, fairness_term2, G, objective_val, _ = compute_metrics(
                c_h, X_h, c_hat, x_hat, alpha, beta
            )

            print(f"評価指標 G: {G}")
            print(f"price_term: {price_term}")
            print(f"fairness_term: {fairness_term1}")
            print(f"v_max^2: {fairness_term2}")
            print(f"計算時間: {h_runtime:.2f}s")
            print(f"gap: {h_gap}")
            print("c =", c_h)
            for k in range(X_h.shape[0]):
                print(f"x[{k + 1}] =", X_h[k])
            c_x = [float(c_h @ X_h[k]) for k in range(X_h.shape[0])]
            print("c@x[1]={:.6f}, c@x[2]={:.6f}".format(c_x[0], c_x[1]))
            x_opt = solve_original_LP(A, b_vecs, c_h)
            c_x_opt = [float(c_h @ x_opt[k]) for k in range(x_opt.shape[0])]
            for k in range(x_opt.shape[0]):
                print(f"original_LP x[{k+1}]={x_opt[k]}")
                print(f"original_LP c@x[{k+1}]={c_x_opt[k]:.6f}")
            print("-" * 40)

            writer.writerow(
                [
                    K_local,
                    n_local,
                    m_local,
                    alpha,
                    beta,
                    G,
                    price_term,
                    fairness_term1,
                    fairness_term2,
                    objective_val,
                    h_runtime,
                    h_gap,
                    status_label(h_status),
                    dd1_runtime,
                    dd1_gap,
                    status_label(dd1_status),
                    dd2_runtime,
                    dd2_gap,
                    status_label(dd2_status),
                ]
            )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-data hybrid experiments.")
    parser.add_argument("--output", default="real_hybrid.csv", help="Output CSV path.")
    parser.add_argument(
        "--only",
        action="append",
        help="Run only a specific pair in 'alpha,beta' format. Can be repeated.",
    )
    parser.add_argument("--limit1", type=int, default=LIMIT_1, help="Time limit for D-D-FACE_1 (seconds).")
    parser.add_argument("--limit2", type=int, default=LIMIT_2, help="Time limit for D-D-FACE_2 (seconds).")
    parser.add_argument("--hybrid-limit", type=int, default=HYBRID_LIMIT, help="Time limit for hybrid stage.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiments(
        output_csv=args.output,
        only_list=parse_only_list(args.only),
        limit_1=args.limit1,
        limit_2=args.limit2,
        hybrid_limit=args.hybrid_limit,
    )


if __name__ == "__main__":
    main()

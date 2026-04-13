import argparse
import csv
from typing import Optional, Sequence, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from 乱数データ_2段階 import (
    ALPHA_BETA_LIST_DEFAULT,
    LAMBDA1,
    LAMBDA2,
    SEED_LIST_DEFAULT,
    build_c_bar_1,
    build_c_bar_2,
    build_h_vector,
    experiment_iterator,
    generate_random_matrices,
    parse_alpha_beta_list,
    parse_only_list,
    size_label,
    solve_dd_face,
    solve_original_LP,
    status_label,
)
from 乱数データ_双線形 import compute_metrics


TIME_LIMIT_DEFAULT = 5
LIMIT_1_DEFAULT = 50
LIMIT_2_DEFAULT = 10
FEAS_TOL = 1e-6


def solve_hybrid_face(
    A: np.ndarray,
    b_vecs: np.ndarray,
    c_hat: np.ndarray,
    h: np.ndarray,
    x_hat: np.ndarray,
    X_fixed: np.ndarray,
    alpha: float,
    beta: float,
    time_limit: int = TIME_LIMIT_DEFAULT,
    start_solution: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    iis_path: Optional[str] = None,
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
    Optional[str],
]:
    K, _ = b_vecs.shape
    m, n = A.shape

    model = gp.Model("hybrid_face")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.NonConvex = 2
    model.Params.Seed = 0

    Y = model.addMVar((K, m), lb=0.0, name="Y")
    c = model.addMVar(n, lb=0.0, name="c")
    v = model.addMVar(K, lb=-GRB.INFINITY, name="v")
    v_ave = model.addVar(lb=-GRB.INFINITY, name="v_ave")
    v_max = model.addVar(lb=1.0, name="v_max")

    if start_solution is not None:
        Y_start, c_start = start_solution
        Y.Start = Y_start
        c.Start = c_start
        denom_eps = 1e-8
        v_start = np.array(
            [
                float(c_start @ X_fixed[k]) / float(c_hat @ x_hat[k] + denom_eps)
                for k in range(K)
            ],
            dtype=float,
        )
        v.Start = v_start
        v_ave.Start = float(np.mean(v_start))
        v_max.Start = float(max(1.0, np.max(v_start)))

    if X_fixed.shape != (K, n):
        raise ValueError(f"X_fixed must be shape {(K, n)}, got {X_fixed.shape}")
    lhs = A @ X_fixed.T
    ax_max_vio = float(np.max(b_vecs - lhs.T))
    if ax_max_vio > FEAS_TOL:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            0.0,
            None,
            GRB.INFEASIBLE,
            f"Ax_lt_b({ax_max_vio:.2e})",
        )
    if np.any(X_fixed.sum(axis=0) > h + FEAS_TOL):
        return None, None, None, None, None, None, 0.0, None, GRB.INFEASIBLE, "capacity_exceeded"

    for k in range(K):
        model.addConstr(A.T @ Y[k] <= c, name=f"ATy_le_c_{k}")

    model.addConstr(c >= LAMBDA1 * c_hat, name="c_lower_bound")
    model.addConstr(c <= LAMBDA2 * c_hat, name="c_upper_bound")

    cost_exprs = []
    for k in range(K):
        expr = gp.quicksum(c[i] * X_fixed[k, i] for i in range(n))
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

    model.optimize()

    runtime = model.Runtime
    status = model.Status
    gap = None
    if model.SolCount > 0:
        try:
            gap = model.MIPGap
        except (gp.GurobiError, AttributeError):
            gap = 0.0
        return X_fixed, Y.X, c.X, v.X, v_ave.X, v_max.X, runtime, gap, status, None

    reason = "model_infeasible" if status == GRB.INFEASIBLE else None
    if reason == "model_infeasible" and iis_path:
        model.computeIIS()
        model.write(iis_path)
    return None, None, None, None, None, None, runtime, gap, status, reason


def run_experiments(
    output_csv: str,
    seed_list: Sequence[int] = SEED_LIST_DEFAULT,
    only_list: Optional[Sequence[Tuple[int, int, int]]] = None,
    alpha_beta_list: Sequence[Tuple[float, float]] = ALPHA_BETA_LIST_DEFAULT,
    time_limit: int = TIME_LIMIT_DEFAULT,
    limit_1: int = LIMIT_1_DEFAULT,
    limit_2: int = LIMIT_2_DEFAULT,
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
                "dd_face1_time",
                "dd_face1_status",
                "dd_face2_time",
                "dd_face2_status",
                "hybrid_time",
                "hybrid_status",
                "objective",
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
                    (
                        X_1,
                        Y_1,
                        c_1,
                        _,
                        _,
                        _,
                        _,
                        dd1_runtime,
                        _,
                        _,
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
                        numeric_focus=2,
                        mip_focus=3,
                        heuristics=0.5,
                    )
                    if X_1 is None or Y_1 is None or c_1 is None:
                        print(
                            "Done "
                            f"seed={seed_value}, K={K}, m={m}, n={n}, alpha={alpha}, beta={beta}, "
                            f"dd1={status_label(dd1_status)}",
                            flush=True,
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
                                dd1_runtime,
                                status_label(dd1_status),
                                np.nan,
                                "",
                                np.nan,
                                "",
                                np.nan,
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
                        _,
                        dd2_runtime,
                        _,
                        _,
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
                        numeric_focus=2,
                        mip_focus=3,
                        heuristics=0.5,
                        start_solution=(X_1, Y_1, c_1),
                    )
                    if X_2 is None or Y_2 is None or c_2 is None:
                        print(
                            "Done "
                            f"seed={seed_value}, K={K}, m={m}, n={n}, alpha={alpha}, beta={beta}, "
                            f"dd1={status_label(dd1_status)}, dd2={status_label(dd2_status)}",
                            flush=True,
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
                                dd1_runtime,
                                status_label(dd1_status),
                                dd2_runtime,
                                status_label(dd2_status),
                                np.nan,
                                "",
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                            ]
                        )
                        continue

                    iis_path = None
                    if write_iis:
                        iis_path = f"hybrid_iis_seed{seed_value}_K{K}_m{m}_n{n}.ilp"
                    (
                        X_h,
                        Y_h,
                        c_h,
                        _,
                        _,
                        _,
                        h_runtime,
                        _,
                        h_status,
                        h_reason,
                    ) = solve_hybrid_face(
                        A,
                        b_vecs,
                        c_hat,
                        h,
                        x_hat,
                        X_2,
                        alpha,
                        beta,
                        time_limit=time_limit,
                        start_solution=(Y_2, c_2),
                        iis_path=iis_path,
                    )
                    if X_h is None or Y_h is None or c_h is None:
                        print(
                            "Done "
                            f"seed={seed_value}, K={K}, m={m}, n={n}, alpha={alpha}, beta={beta}, "
                            f"dd1={status_label(dd1_status)}, dd2={status_label(dd2_status)}, "
                            f"hybrid={status_label(h_status)}, reason={h_reason}",
                            flush=True,
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
                                dd1_runtime,
                                status_label(dd1_status),
                                dd2_runtime,
                                status_label(dd2_status),
                                h_runtime,
                                status_label(h_status),
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                            ]
                        )
                        continue

                    ax_violation = float(max(0.0, np.max(b_vecs - (A @ X_h.T).T)))
                    sd_violation = float(
                        np.max(
                            [
                                abs((c_h @ X_h[k]) - (b_vecs[k] @ Y_h[k]))
                                for k in range(K)
                            ]
                        )
                    )
                    (
                        _,
                        _,
                        _,
                        price_term,
                        fairness_term1,
                        fairness_term2,
                        G,
                        objective_val,
                    ) = compute_metrics(c_h, X_h, c_hat, x_hat, alpha, beta)
                    writer.writerow(
                        [
                            seed_value,
                            K,
                            m,
                            n,
                            size,
                            alpha,
                            beta,
                            dd1_runtime,
                            status_label(dd1_status),
                            dd2_runtime,
                            status_label(dd2_status),
                            h_runtime,
                            status_label(h_status),
                            objective_val,
                            G,
                            price_term,
                            fairness_term1,
                            fairness_term2,
                        ]
                    )
                    print(
                        "Done "
                        f"seed={seed_value}, K={K}, m={m}, n={n}, alpha={alpha}, beta={beta}, "
                        f"dd1={status_label(dd1_status)}, dd2={status_label(dd2_status)}, "
                        f"hybrid={status_label(h_status)}, t={h_runtime:.2f}, "
                        f"ax_vio={ax_violation:.2e}, sd_vio={sd_violation:.2e}",
                        flush=True,
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hybrid experiments: discrete-discrete followed by continuous FACE."
    )
    parser.add_argument("--output", default="random_hybrid.csv", help="Output CSV path.")
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
    parser.add_argument("--time-limit", type=int, default=TIME_LIMIT_DEFAULT)
    parser.add_argument("--limit-1", type=int, default=LIMIT_1_DEFAULT)
    parser.add_argument("--limit-2", type=int, default=LIMIT_2_DEFAULT)
    parser.add_argument("--iis", action="store_true", help="Write IIS for infeasible hybrid runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    only_list = parse_only_list(args.only)
    run_experiments(
        output_csv=args.output,
        seed_list=args.seed or SEED_LIST_DEFAULT,
        only_list=only_list,
        alpha_beta_list=parse_alpha_beta_list(args.alpha_beta) or ALPHA_BETA_LIST_DEFAULT,
        time_limit=args.time_limit,
        limit_1=args.limit_1,
        limit_2=args.limit_2,
        write_iis=args.iis,
    )


if __name__ == "__main__":
    main()

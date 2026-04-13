import argparse
import csv
from typing import Optional, Sequence, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from 乱数データ_2段階 import (
    K_LIST,
    LAMBDA1,
    LAMBDA2,
    N_LIST,
    RHO_LIST,
    SEED_LIST_DEFAULT,
    build_c_bar_1,
    build_h_vector,
    generate_random_matrices,
    solve_original_LP,
)

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


def experiment_iterator() -> Sequence[Tuple[int, int, float]]:
    items: list[Tuple[int, int, float]] = []
    for K in K_LIST:
        for n in N_LIST:
            for rho in RHO_LIST:
                items.append((K, n, rho))
    return items


def solve_dd_face_feas(
    A: np.ndarray,
    b_vecs: np.ndarray,
    c_hat: np.ndarray,
    h: np.ndarray,
    x_hat: np.ndarray,
    c_bar: np.ndarray,
    time_limit: int,
    numeric_focus: int,
) -> Tuple[Optional[float], float, int]:
    K, _ = b_vecs.shape
    m, n = A.shape
    _, J_local = c_bar.shape

    model = gp.Model("dd_face_feas")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.Seed = 0
    model.Params.NumericFocus = 1
    model.Params.NonConvex = 2
    model.Params.Heuristics = 1.0
    model.Params.MIPFocus = 1

    X = model.addMVar((K, n), lb=0.0, name="X")
    Y = model.addMVar((K, m), lb=0.0, name="Y")
    c = model.addMVar(n, lb=-GRB.INFINITY, name="c")
    z = model.addMVar((n, J_local), vtype=GRB.BINARY, name="z")
    v = model.addMVar(K, lb=-GRB.INFINITY, name="v")
    v_ave = model.addVar(lb=-GRB.INFINITY, name="v_ave")
    v_max = model.addVar(lb=0.0, name="v_max")

    for k in range(K):
        model.addConstr(A @ X[k] >= b_vecs[k])
    for k in range(K):
        model.addConstr(A.T @ Y[k] <= c)
    model.addConstr(X.sum(axis=0) <= h)

    for i in range(n):
        model.addConstr(z[i].sum() == 1)
        model.addConstr(c[i] == gp.quicksum(c_bar[i, j] * z[i, j] for j in range(J_local)))
    lower = LAMBDA1 * c_hat
    upper = LAMBDA2 * c_hat
    for i in range(n):
        for j in range(J_local):
            if c_bar[i, j] < lower[i] or c_bar[i, j] > upper[i]:
                model.addConstr(z[i, j] == 0)

    cost_exprs = []
    for k in range(K):
        expr = gp.quicksum(c[i] * X[k, i] for i in range(n))
        cost_exprs.append(expr)
        model.addConstr(b_vecs[k] @ Y[k] == expr)

    for k in range(K):
        denom = float(c_hat @ x_hat[k])
        model.addConstr(v[k] == 100 * cost_exprs[k] / denom)
        model.addConstr(v_max >= v[k] - 100)
    model.addConstr(v_ave == (1.0 / K) * gp.quicksum(v[k] for k in range(K)))

    model.setObjective(0.0, GRB.MINIMIZE)

    first_feas_time = [None]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL and first_feas_time[0] is None:
            first_feas_time[0] = model.cbGet(GRB.Callback.RUNTIME)

    model.optimize(cb)

    return first_feas_time[0], model.Runtime, model.Status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure time to first feasible solution for stage 1.")
    parser.add_argument("--time-limit", type=int, default=300, help="Time limit (seconds).")
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
    parser.add_argument("--output", help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    only_list = parse_only_list(args.only)
    if only_list:
        experiment_items = [(K, n, float(m) / float(n)) for K, m, n in only_list]
    else:
        experiment_items = experiment_iterator()
    seed_list = args.seed or SEED_LIST_DEFAULT
    rows = []

    for seed_value in seed_list:
        for K, n, rho in experiment_items:
            m = int(round(n * rho))
            rng = np.random.default_rng(np.random.SeedSequence([seed_value, K, n, m]))
            A, b_vecs, c_hat = generate_random_matrices(rng, K, m, n)
            x_hat = solve_original_LP(A, b_vecs, c_hat)
            h = build_h_vector(rng, x_hat)
            c_bar_1 = build_c_bar_1(c_hat)

            first_feas_time, runtime, status = solve_dd_face_feas(
                A,
                b_vecs,
                c_hat,
                h,
                x_hat,
                c_bar_1,
                time_limit=args.time_limit,
                numeric_focus=1,
            )
            rows.append([seed_value, K, m, n, first_feas_time, runtime, status])
            print(
                f"seed={seed_value}, K={K}, n={n}, m={m}, "
                f"first_feas_time={first_feas_time}, runtime={runtime:.2f}, status={status}"
            )
    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "K", "m", "n", "first_feas_time", "runtime", "status"])
            writer.writerows(rows)


if __name__ == "__main__":
    main()

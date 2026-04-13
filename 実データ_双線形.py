"""
FACE model solver for the fixed dataset in `要件定義書.md` (continuous price version).
"""
import csv
import time
from typing import Optional, Sequence, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

# Problem data from the requirements document
# A = np.array(
A = np.array(
    [
        [392, 343, 142, 133, 127],  # energy
        [6.1, 12.6, 12.2, 21.3, 22.5],  # protein
        [0.9, 1.7, 10.2, 5.9, 4.5],  # fat
    ]
)

b_vecs = np.array(
    [
        [8000, 415, 213.6],  # b1
        [7040, 600, 192.8],  # b2
    ],
    dtype=float,
)

# Problem data from the requirements document
# A = np.array(
#     [
#         [7.84/160*100, 6.86/160*100, 2.84/160*100, 2.66/160*100, 2.54/160*100],  # energy
#         [6.1/415*100, 12.6/415*100, 12.2/415*100, 21.3/415*100, 22.5/415*100],  # protein
#         [0.9/213.6*100, 1.7/213.6*100, 10.2/213.6*100, 5.9/213.6*100, 4.5/213.6*100],  # fat
#     ],
#     dtype=float,
# )

# b_vecs = np.array(
#     [
#         [160/160*100, 415/415*100, 213.6/213.6*100],  # b1
#         [140.8/160*100, 600/415*100, 192.8/213.6*100],  # b2
#     ],
#     dtype=float,
# )

c_hat = np.array([50, 40, 90, 70, 100], dtype=float)
h = np.array([100, 100, 20, 100, 100], dtype=float)

foods = ["米", "小麦", "卵", "鶏肉", "魚"]
K = b_vecs.shape[0]
n = len(c_hat)
m = A.shape[0]
lambda_1 = 0.1
lambda_2 = 1.9
TIME_LIMIT = 10  # seconds
ALPHA_BETA_PAIRS_DEFAULT = [(0.0, 0.0), (0.25, 0.25), (0.0, 0.5), (0.5, 0.0)]
# ALPHA_BETA_PAIRS_DEFAULT = [(0.0, 0.0)]
OUTPUT_CSV_DEFAULT = "real_data.csv"


def make_progress_callback(report_interval: float = 30.0):
    """Gurobi callback that prints elapsed time every `report_interval` seconds."""
    last_report = {"time": 0.0}

    def callback(model: gp.Model, where: int) -> None:
        if where not in {
            GRB.Callback.MIP,
            GRB.Callback.MIPNODE,
            GRB.Callback.MIPSOL,
            GRB.Callback.SIMPLEX,
            GRB.Callback.BARRIER,
        }:
            return
        try:
            runtime = model.cbGet(GRB.Callback.RUNTIME)
        except gp.GurobiError:
            return
        if runtime - last_report["time"] >= report_interval:
            print(f"[progress] elapsed: {runtime:.1f}s")
            last_report["time"] = runtime

    return callback


def solve_original_LP(A_mat: np.ndarray, b_mat: np.ndarray, c_vec: np.ndarray) -> np.ndarray:
    """
    Solve the base LP:
        min sum_k c_vec @ x_k
        s.t. A x_k >= b_k, x_k >= 0
    Returns the optimal x_hat with shape (K, n).
    """
    K_local, _ = b_mat.shape
    _, n_local = A_mat.shape
    model = gp.Model("original_lp")
    model.Params.OutputFlag = 0

    x = model.addMVar((K_local, n_local), lb=0.0, name="x")
    for k in range(K_local):
        model.addConstr(A_mat @ x[k] >= b_mat[k], name=f"Ax_ge_b_{k}")

    objective = gp.quicksum(c_vec @ x[k] for k in range(K_local))
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Original LP not optimal. Status: {model.Status}")
    return x.X


def solve_FACE(
    A_mat: np.ndarray,
    b_mat: np.ndarray,
    c_vec: np.ndarray,
    h_vec: np.ndarray,
    x_hat: np.ndarray,
    alpha: float,
    beta: float,
    time_limit: int = TIME_LIMIT,
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
    Solve the FACE model with continuous prices.
    Returns X, c, v, v_ave, v_max, runtime, gap.
    Any of X/c/v/v_ave/v_max may be None if no feasible solution was found.
    """
    K_local, _ = b_mat.shape
    _, n_local = A_mat.shape

    model = gp.Model("face_continuous")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.NonConvex = 2  # needed for bilinear terms c[i] * X[k, i]

    progress_cb = make_progress_callback()

    X = model.addMVar((K_local, n_local), lb=0.0, name="X")
    Y = model.addMVar((K_local, m), lb=0.0, name="Y")
    c = model.addMVar(n_local, lb=0.0, name="c")
    v = model.addMVar(K_local, lb=-GRB.INFINITY, name="v")
    v_ave = model.addVar(lb=-GRB.INFINITY, name="v_ave")
    v_max = model.addVar(lb=1.0, name="v_max")

    # Primal feasibility constraints
    for k in range(K_local):
        model.addConstr(A_mat @ X[k] >= b_mat[k], name=f"Ax_ge_b_{k}")

    # Dual feasibility constraints
    for k in range(K_local):
        model.addConstr(A_mat.T @ Y[k] <= c, name=f"ATy_le_c_{k}")

    # Capacity constraint
    model.addConstr(X.sum(axis=0) <= h_vec, name="capacity_h")

    # Bounds on c
    model.addConstr(c >= lambda_1 * c_vec, name="c_lower_bound")
    model.addConstr(c <= lambda_2 * c_vec, name="c_upper_bound")

    # Strong duality and fairness variables
    primal_cost_exprs = []
    for k in range(K_local):
        cost_expr = gp.quicksum(c[i] * X[k, i] for i in range(n_local))
        primal_cost_exprs.append(cost_expr)
        model.addConstr(b_mat[k] @ Y[k] == cost_expr, name=f"strong_duality_{k}")

    for k in range(K_local):
        denom = float(c_vec @ x_hat[k])
        model.addConstr(v[k] == primal_cost_exprs[k] / denom, name=f"v_def_{k}")
        model.addConstr(v_max >= v[k], name=f"vmax_ge_v_{k}")
    model.addConstr(v_ave == (1.0 / K_local) * gp.quicksum(v[k] for k in range(K_local)), name="v_average")

    price_term = gp.quicksum(((c[i] - c_vec[i]) / c_vec[i]) * ((c[i] - c_vec[i]) / c_vec[i]) for i in range(n_local))
    fairness_term1 = gp.quicksum((v[k] - v_ave) * (v[k] - v_ave) for k in range(K_local))
    fairness_term2 = (v_max - 1) ** 2
    objective = (1 - alpha - beta) * (1.0 / n_local) * price_term + alpha * (1.0 / K_local) * fairness_term1 + beta * fairness_term2
    model.setObjective(objective, GRB.MINIMIZE)

    start_time = time.time()
    model.optimize(callback=progress_cb)
    runtime = time.time() - start_time
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
            float(c_sol @ X_sol[k]) / float(c_hat @ x_hat[k])
            for k in range(K_local)
        ],
        dtype=float,
    )
    v_ave = float(np.mean(v_values))
    v_max = float(max(1.0, np.max(v_values)))
    price_term = (float(np.sum(((c_sol - c_hat) / c_hat) ** 2)) / n_local) ** 0.5
    fairness_term1 = (float(np.sum((v_values - v_ave) ** 2)) / K_local) ** 0.5
    fairness_term2 = float(v_max - 1.0)
    G = price_term + fairness_term1 + fairness_term2
    objective_val = (
        (1 - alpha - beta) * (price_term ** 2)
        + alpha * (fairness_term1 ** 2)
        + beta * (fairness_term2 ** 2)
    )
    return v_values, v_ave, v_max, price_term, fairness_term1, fairness_term2, G, objective_val


def run_all(
    alpha_beta_pairs: Sequence[Tuple[float, float]] = ALPHA_BETA_PAIRS_DEFAULT,
    output_csv: str = OUTPUT_CSV_DEFAULT,
) -> None:
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

        for alpha, beta in alpha_beta_pairs:
                X_sol, c_sol, v_sol, v_ave_sol, v_max_sol, runtime, gap, status = solve_FACE(
                    A, b_vecs, c_hat, h, x_hat, alpha, beta, time_limit=TIME_LIMIT
                )
                print(f"alpha: {alpha}, beta: {beta}")
                if X_sol is None or c_sol is None or v_sol is None or v_ave_sol is None or v_max_sol is None:
                    print("No feasible solution found.")
                    print(f"計算時間: {runtime:.2f}s")
                    print(f"gap: {gap}")
                    writer.writerow([K, n, m, alpha, beta, np.nan, np.nan, np.nan, np.nan, np.nan, runtime, gap, status])
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

                print(f"評価指標 G: {G}")
                print(f"price_term: {price_term}")
                print(f"fairness_term: {fairness_term1}")
                print(f"v_max^2: {fairness_term2}")
                print(f"計算時間: {runtime:.2f}s")
                print(f"gap: {gap}")
                print("c =", c_sol)
                for k in range(K):
                    print(f"x[{k + 1}] =", X_sol[k])
                c_x = [float(c_sol @ X_sol[k]) for k in range(X_sol.shape[0])]
                print(
                    "c@x[1]={:.6f}, c@x[2]={:.6f}".format(c_x[0], c_x[1])
                )
                x_opt = solve_original_LP(A, b_vecs, c_sol)
                c_x_opt = [float(c_sol @ x_opt[k]) for k in range(x_opt.shape[0])]
                for k in range(x_opt.shape[0]):
                    print(f"original_LP x[{k+1}]={x_opt[k]}")
                    print(f"original_LP c@x[{k+1}]={c_x_opt[k]:.6f}")
                print("-" * 40)

                writer.writerow([K, n, m, alpha, beta, G, price_term, fairness_term1, fairness_term2, objective_val, runtime, gap, status])


if __name__ == "__main__":
    run_all()

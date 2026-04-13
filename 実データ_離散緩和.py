"""
FACE model solver for the fixed dataset described in `要件定義書_c離散化.md`.
"""
import csv
import time
from typing import Dict, List, Optional, Sequence, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

# A = np.array(
#     [
#         [7.84, 6.86, 2.84, 2.66, 2.54],  # energy
#         [6.1, 12.6, 12.2, 21.3, 22.5],  # protein
#         [0.9, 1.7, 10.2, 5.9, 4.5],  # fat
#     ],
#     dtype=float,
# )

# b_vecs = np.array(
#     [
#         [160, 415, 213.6],  # b1
#         [140.8, 600, 192.8],  # b2
#     ],
#     dtype=float,
# )

A = np.array(
    [
        [7.84/160*100, 6.86/160*100, 2.84/160*100, 2.66/160*100, 2.54/160*100],  # energy
        [6.1/415*100, 12.6/415*100, 12.2/415*100, 21.3/415*100, 22.5/415*100],  # protein
        [0.9/213.6*100, 1.7/213.6*100, 10.2/213.6*100, 5.9/213.6*100, 4.5/213.6*100],  # fat
    ],
    dtype=float,
)

b_vecs = np.array(
    [
        [160/160*100, 415/415*100, 213.6/213.6*100],  # b1
        [140.8/160*100, 600/415*100, 192.8/213.6*100],  # b2
    ],
    dtype=float,
)


c_hat = np.array([50, 40, 90, 70, 100], dtype=float)
h = np.array([100, 100, 25, 10, 100], dtype=float)

foods = ["米", "小麦", "卵", "鶏肉", "魚"]
K = b_vecs.shape[0]
n = len(c_hat)
m = A.shape[0]
lambda_1 = 0.5
lambda_2 = 1.5
M_BIG = 10 ** 2
TIME_LIMIT = 60  # seconds
ALPHA_LIST_DEFAULT = [0.0, 0.25, 0.5, 0.75, 1.0]
BETA_LIST_DEFAULT = [0.0, 0.25, 0.5, 0.75, 1.0]
OUTPUT_CSV_DEFAULT = "c_real_data.csv"
start_value = 0.005
cutsize=21
stepsize = (0.5/start_value) ** (2/(cutsize-3))


def make_D_values(c_base, start_value, stepsize, cutsize: int) -> List[np.ndarray]:
    """
    Build D_i from multiplicative tweaks around c_hat[i]:
    {1 + 0.0016 * 5^l for l=0..4} U {1 - 0.0016 * 5^l for l=1..4}, scaled by c_hat[i].
    """
    D_values: List[np.ndarray] = []
    plus_factors = [1 + start_value * (stepsize ** l) for l in range(cutsize//2)]
    minus_factors = [1 - start_value * (stepsize ** l) for l in range(cutsize//2)]
    factors = plus_factors + minus_factors + [1.0]
    for val in c_base:
        candidates = sorted({val * f for f in factors if val * f > 0})
        D_values.append(np.array(candidates, dtype=float))
    return D_values


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


def make_progress_callback(report_interval: float = 30.0):
    """
    Gurobi callback that prints elapsed time every `report_interval` seconds.
    """
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


def solve_FACE(
    A_mat: np.ndarray,
    b_mat: np.ndarray,
    c_vec: np.ndarray,
    h_vec: np.ndarray,
    x_hat: np.ndarray,
    alpha: float,
    beta: float,
    time_limit: int = TIME_LIMIT,
    D_values: Optional[List[np.ndarray]] = None,
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
    Solve the FACE model with discrete prices.
    Returns X, c, v, v_ave, v_max, runtime, gap.
    Any of X/c/v/v_ave/v_max may be None if no feasible solution was found.
    """
    K_local, _ = b_mat.shape
    m_local, n_local = A_mat.shape

    if D_values is None:
        D_values = make_D_values(c_vec)

    model = gp.Model("face_c_discrete")
    model.Params.OutputFlag = 1  # 計算ログをターミナルに出力
    model.Params.TimeLimit = time_limit
    model.setParam("NumericFocus", 0)

    progress_cb = make_progress_callback()

    X = model.addMVar((K_local, n_local), lb=0.0, name="X")
    Y = model.addMVar((K_local, m_local), lb=0.0, name="Y")
    c = model.addMVar(n_local, lb=0.0, name="c")
    Z: Dict[Tuple[int, int], gp.Var] = {}
    U: Dict[Tuple[int, int, int], gp.Var] = {}
    for i, d_list in enumerate(D_values):
        for j_idx, _ in enumerate(d_list):
            Z[(i, j_idx)] = model.addVar(vtype=GRB.BINARY, name=f"Z_{i}_{j_idx}")
            for k in range(K_local):
                U[(k, i, j_idx)] = model.addVar(lb=0.0, name=f"U_{k}_{i}_{j_idx}")
    v = model.addMVar(K_local, lb=-GRB.INFINITY, name="v")
    v_ave = model.addVar(lb=-GRB.INFINITY, name="v_ave")
    v_max = model.addVar(lb=0.0, name="v_max")

    # Primal feasibility constraints
    for k in range(K_local):
        model.addConstr(A_mat @ X[k] >= b_mat[k], name=f"Ax_ge_b_{k}")

    # Dual feasibility constraints
    for k in range(K_local):
        model.addConstr(A_mat.T @ Y[k] <= c, name=f"ATy_le_c_{k}")

    # Capacity constraint
    model.addConstr(X.sum(axis=0) <= h_vec, name="capacity_h")

    # Linking c_i with z_{i,j}
    for i in range(n_local):
        model.addConstr(gp.quicksum(Z[(i, j_idx)] for j_idx in range(len(D_values[i]))) == 1, name=f"z_onehot_{i}")
        model.addConstr(
            c[i] == gp.quicksum(D_values[i][j_idx] * Z[(i, j_idx)] for j_idx in range(len(D_values[i]))),
            name=f"c_def_{i}",
        )

    # u_{k,i,j} linearization with x_{k,i} and z_{i,j}
    for k in range(K_local):
        for i in range(n_local):
            for j_idx in range(len(D_values[i])):
                model.addConstr(U[(k, i, j_idx)] <= h_vec[i] * Z[(i, j_idx)], name=f"u_bigM_pos_{k}_{i}_{j_idx}")
                model.addConstr(U[(k, i, j_idx)] <= X[k, i], name=f"u_le_x_{k}_{i}_{j_idx}")
                model.addConstr(
                    U[(k, i, j_idx)] >= X[k, i] - h_vec[i] * (1 - Z[(i, j_idx)]),
                    name=f"u_ge_x_bigM_{k}_{i}_{j_idx}",
                )

    # Duality and fairness-related expressions
    dual_exprs = []
    for k in range(K_local):
        expr = gp.quicksum(
            D_values[i][j_idx] * U[(k, i, j_idx)] for i in range(n_local) for j_idx in range(len(D_values[i]))
        )
        dual_exprs.append(expr)
        model.addConstr(b_mat[k] @ Y[k] == expr, name=f"strong_duality_{k}")

    for k in range(K_local):
        denom = float(c_vec @ x_hat[k])
        model.addConstr(v[k] == 100 * dual_exprs[k] / denom, name=f"v_def_{k}")
        model.addConstr(v_max >= v[k]-100, name=f"vmax_ge_v_{k}")
    model.addConstr(v_ave == (1.0 / K_local) * gp.quicksum(v[k] for k in range(K_local)), name="v_average")

    price_term = gp.quicksum(10000 * ((c[i] - c_vec[i]) / c_vec[i]) * ((c[i] - c_vec[i]) / c_vec[i]) for i in range(n_local))
    fairness_term1 = gp.quicksum((v[k] - v_ave) * (v[k] - v_ave) for k in range(K_local))
    fairness_term2 = v_max * v_max
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
            100.0 * float(c_sol @ X_sol[k]) / float(c_hat @ x_hat[k] )
            for k in range(K_local)
        ],
        dtype=float,
    )
    v_ave = float(np.mean(v_values))
    v_max = float(max(0.0, np.max(v_values)-100.0))
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


def run_all(
    alpha_list: Sequence[float] = ALPHA_LIST_DEFAULT,
    beta_list: Sequence[float] = BETA_LIST_DEFAULT,
    output_csv: str = OUTPUT_CSV_DEFAULT,
    start_value: float = start_value,
    stepsize: float = stepsize,
    cutsize: int = cutsize,
) -> None:
    x_hat = solve_original_LP(A, b_vecs, c_hat)
    D_values = make_D_values(c_hat, start_value, stepsize, cutsize)

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

        for alpha in alpha_list:
            for beta in beta_list:
                if alpha + beta > 1:
                    continue
                X_sol, c_sol, v_sol, v_ave_sol, v_max_sol, runtime, gap, status = solve_FACE(
                    A, b_vecs, c_hat, h, x_hat, alpha, beta, time_limit=TIME_LIMIT, D_values=D_values
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
                print("-" * 40)

                writer.writerow([K, n, m, alpha, beta, G, price_term, fairness_term1, fairness_term2, objective_val, runtime, gap, status])


if __name__ == "__main__":
    run_all()

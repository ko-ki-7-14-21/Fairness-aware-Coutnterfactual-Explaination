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

# Problem data from the requirements document
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
TIME_LIMIT = 60  # seconds
ALPHA_LIST_DEFAULT = [0.0, 0.25, 0.5, 0.75, 1.0]
BETA_LIST_DEFAULT = [0.0, 0.25, 0.5, 0.75, 1.0]
# ALPHA_LIST_DEFAULT = [0.25]
# BETA_LIST_DEFAULT = [0.25]
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

if __name__ == "__main__":
    x_opt = solve_original_LP(A, b_vecs, c_hat)
    for k in range(K):
        print(x_opt[k])
        total_cost = c_hat @ x_opt[k]
        print(f"Total cost: {total_cost:.2f}\n")
    print("sum_x", np.sum(x_opt, axis=0))
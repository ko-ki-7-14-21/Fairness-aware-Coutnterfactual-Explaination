import numpy as np
import gurobipy as gp
from gurobipy import GRB

A = np.array(
    [
        [392, 343, 142, 133, 127],
        [6.1, 12.6, 12.2, 21.3, 22.5],
        [0.9, 1.7, 10.2, 5.9, 4.5],
    ]
)

b_vecs = np.array(
    [
        [8000, 415, 213.6],
        [7040, 600, 192.8],
    ],
    dtype=float,
)

# c = np.array([38.55413077, 48.52591589, 88.20668052, 72.12120559, 64.48080829], dtype=float)
c = np.array([50, 40, 90, 70, 100], dtype=float)
h = np.array([100, 100, 25, 10, 100], dtype=float)

K = b_vecs.shape[0]
n = A.shape[1]

model = gp.Model("lp_check")
model.Params.OutputFlag = 0

x = model.addMVar((K, n), lb=0.0, name="x")

for k in range(K):
    model.addConstr(A @ x[k] >= b_vecs[k], name=f"Ax_ge_b_{k}")

#model.addConstr(x.sum(axis=0) <= h, name="sum_x_le_h")

objective = gp.quicksum(c @ x[k] for k in range(K))
model.setObjective(objective, GRB.MINIMIZE)
model.optimize()

if model.Status != GRB.OPTIMAL:
    raise RuntimeError(f"LP not optimal. Status: {model.Status}")

print(x[0].X)
print(x[1].X)
print(x[0].X + x[1].X)
print(c @ x[0].X)
print(c @ x[1].X)

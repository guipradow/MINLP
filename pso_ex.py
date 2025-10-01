import numpy as np
from pyswarm import pso


def model_obj(x):
    x[0] = int(x[0])
    con1 = (-x[0] + 2*x[1]*x[0] <= 8)
    con2 = (2*x[0] + x[1] <= 14)
    con3 = (2*x[0] - x[1] <= 10)
    pen = 0 if (con1 and con2 and con3) else 1e6
    return -(x[0] + x[1]*x[0]) + pen

lb = [0, 0]
ub = [10, 10]
x0 = [0, 0]

def cons(x):
    return []


xopt, fopt = pso(model_obj, lb, ub, x0, cons)

print(f'x = {xopt[0]}')
print(f'y = {xopt[1]}')

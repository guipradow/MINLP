import numpy as np
from geneticalgorithm import geneticalgorithm as ga

def f(x):
    pen = 0
    if not -x[0]+2*x[1]*x[0] <= 8:
        pen = np.inf
    if not 2*x[0]+x[1]<=14:
        pen = np.inf
    if not 2*x[0]-x[1]<=10:
        pen = np.inf
    return -(x[0] + x[1]*x[0]) + pen
 
varbounds = np.array([[0, 10], [0, 10]])
vartype = np.array([['int'], ['real']])

model = ga(function=f, dimension=2, variable_type_mixed=vartype,
           variable_boundaries=varbounds)

model.run()

'''
http://www.mathematik.uni-dortmund.de/~kuzmin/cfdintro/lecture4.pdf
'''
import os 
import shutil
import numpy as np

from CUR_GRID_FDM.DiscreteSchemes import OperatorFDM3D, CalCoeff
from CUR_GRID_FDM.Geometry import DonutMesh
from Solver.LaplaceSolver_donut2D import SolverLaplace, NodeType

# ===============================================================
# Setting Parameters
# ===============================================================
dir_name = 'OUTOUT'

# ===============================================================
# Biuld Mesh
# ===============================================================

# create rectangular mesh
myMesh = DonutMesh(3, 8, 30, 10)

# Calculate coefficients for curvilinear coordinates
myCoeff = CalCoeff(myMesh)

# define boundary type
BCtype = np.zeros_like(myMesh.X_flatten)

BCtype[myMesh.get_node_index_list(i_front = True,i_end = True)] = NodeType.DIRICHLET


# ===============================================================
# Biuld model for simulation
# ===============================================================
# Create folder for output data
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)
else:
    shutil.rmtree(dir_name)
    os.makedirs(dir_name)

myMesh.plot_grid(BCtype)

# create solver
mySolver = SolverLaplace(myMesh, myCoeff, BCtype, OperatorFDM3D, dir_name)

mySolver.start_solve()



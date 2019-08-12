'''
http://www.mathematik.uni-dortmund.de/~kuzmin/cfdintro/lecture4.pdf
'''
import os
import shutil
import numpy as np

from CUR_GRID_FDM.DiscreteSchemes import OperatorFDM3D, CalCoeff
from CUR_GRID_FDM.Geometry import CurveRectangularMesh
from Solver.LaplaceSolver_rec_all_NuemannBC2D import SolverLaplace, NodeType

# ===============================================================
# Setting Parameters
# ===============================================================
Lx = 2
Ly = 1

nx = 40
ny = 20

dir_name = 'OUTOUT'

# ===============================================================
# Biuld Mesh
# ===============================================================

# create rectangular mesh
myMesh = CurveRectangularMesh(Lx, nx, Ly, ny)

# # Calculate coefficients for curvilinear coordinates
myCoeff = CalCoeff(myMesh)

# define boundary type
BCtype = np.zeros_like(myMesh.X_flatten)

BCtype[myMesh.get_node_index_list(i_end = True, j_end = True, i_front = True, j_front = True)] = NodeType.WALL
BCtype[myMesh.get_node_index_list(i_front = True)] = NodeType.INLET
BCtype[myMesh.get_node_index_list(i_end = True)[10:20]] = NodeType.OUTLET

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
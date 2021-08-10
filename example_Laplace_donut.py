'''
http://www.mathematik.uni-dortmund.de/~kuzmin/cfdintro/lecture4.pdf
'''
import os
import shutil
import numpy as np

from CUR_GRID_FDM.Geometry.BaseMesh import NODELOC
from CUR_GRID_FDM.DiscreteSchemes import OperatorFDM3D, CalCoeff
from CUR_GRID_FDM.Geometry import DonutMesh
from Solver.LaplaceSolver_donut import SolverLaplace, NodeType

# ===============================================================
# Setting Parameters
# ===============================================================
dirName = 'output'

# ===============================================================
# Biuld Mesh
# ===============================================================

# create rectangular mesh
myMesh = DonutMesh(3.0, 6.0, 7, 20, 1.0, 10)

# Calculate coefficients for curvilinear coordinates
myCoeff = CalCoeff(myMesh)
myOperator = OperatorFDM3D(myMesh)

# define boundary type
BCtype = np.zeros_like(myMesh.x_flatten())

BCtype[myMesh.get_node_index_list(NODELOC.INTERIOR, True)] = NodeType.DIRICHLET


# ===============================================================
# Biuld model for simulation
# ===============================================================
# Create folder for output data
if not os.path.isdir(dirName):
    os.makedirs(dirName)
else:
    shutil.rmtree(dirName)
    os.makedirs(dirName)

myMesh.plot_grid(BCtype)

# create solver
mySolver = SolverLaplace(myMesh, myCoeff, BCtype, myOperator, dirName)

mySolver.start_solve()

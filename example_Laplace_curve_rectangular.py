'''
http://www.mathematik.uni-dortmund.de/~kuzmin/cfdintro/lecture4.pdf
'''
import os
import shutil
import numpy as np

from CUR_GRID_FDM.Geometry.BaseMesh import NODELOC
from CUR_GRID_FDM.DiscreteSchemes import OperatorFDM3D, CalCoeff
from CUR_GRID_FDM.Geometry import CurveRectangularMesh
from Solver.Laplace_rectangular import SolverLaplace, NodeType

# ===============================================================
# Setting Parameters
# ===============================================================
Lx = 1
Ly = 1
Lz = 1

nx = 10
ny = 12
nz = 13

dirName = 'output'

# ===============================================================
# Biuld Mesh
# ===============================================================

# create rectangular mesh
myMesh = CurveRectangularMesh(Lx, nx, Ly, ny, Lz, nz)

# # Calculate coefficients for curvilinear coordinates
myCoeff = CalCoeff(myMesh)
myOperator = OperatorFDM3D(myMesh)

# # define boundary type
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

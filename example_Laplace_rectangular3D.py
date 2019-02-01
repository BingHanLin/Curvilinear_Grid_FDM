'''
http://www.mathematik.uni-dortmund.de/~kuzmin/cfdintro/lecture4.pdf
'''
import os
import shutil
import numpy as np
from Geometry.RectangularMesh import RectangularMesh
from DiscreteSchemes.OperatorFDM3D import OperatorFDM3D
from DiscreteSchemes.CalCoeff import CalCoeff
from Solver.Laplace_rectangular3D import SolverLaplace, NodeType
# ===============================================================
# Setting Parameters
# ===============================================================
Lx = 1
Ly = 1
Lz = 1

nx = 10
ny = 10
nz = 10

dir_name = 'OUTOUT'

# ===============================================================
# Biuld Mesh
# ===============================================================

# create rectangular mesh
myMesh = RectangularMesh(Lx, nx, Ly, ny, Lz, nz)

# # Calculate coefficients for curvilinear coordinates
myCoeff = CalCoeff(myMesh)

# # define boundary type
BCtype = np.zeros_like(myMesh.X_flatten)

BCtype[myMesh.get_node_index_list(i_front = True,i_end = True, 
                                  j_front = True ,j_end = True, 
                                  k_front = True ,k_end = True)] = NodeType.DIRICHLET


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
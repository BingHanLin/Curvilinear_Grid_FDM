'''
http://www.mathematik.uni-dortmund.de/~kuzmin/cfdintro/lecture4.pdf
'''
import os
import shutil
import numpy as np
from Geometry.UserDataBaseMesh import UserDataBaseMesh, DataBase
from DiscreteSchemes.OperatorFDM3D import OperatorFDM3D
from DiscreteSchemes.CalCoeff import CalCoeff
from Solver.LaplaceSolver_UserDataBaseMesh import SolverLaplace, NodeType
# ===============================================================
# Setting Parameters
# ===============================================================
dir_name = 'OUTOUT'

# ===============================================================
# Biuld Mesh
# ===============================================================

# create rectangular mesh
myDataBase = DataBase()
myMesh = UserDataBaseMesh(myDataBase)

# # Calculate coefficients for curvilinear coordinates
myCoeff = CalCoeff(myMesh)

# define boundary type
BCtype = np.zeros_like(myMesh.X_flatten)

BCtype[myMesh.get_node_index_list(i_front = True,i_end = True, j_front = True ,j_end = True)] = NodeType.DIRICHLET

# # ===============================================================
# # Biuld model for simulation
# # ===============================================================

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
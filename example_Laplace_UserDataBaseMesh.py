'''
http://www.mathematik.uni-dortmund.de/~kuzmin/cfdintro/lecture4.pdf
'''
import os
import shutil
import numpy as np

from CUR_GRID_FDM.DiscreteSchemes import OperatorFDM3D, CalCoeff
from CUR_GRID_FDM.Geometry import UserDataBaseMesh, DataBase
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

BCtype[myMesh.get_node_index_list(i_front = True)] = NodeType.BOTTOMWALL_1
BCtype[myMesh.get_node_index_list(j_front = True)] = NodeType.BOTTOMWALL_2
BCtype[myMesh.get_node_index_list(j_end = True)] = NodeType.TOPWALL
BCtype[myMesh.get_node_index_list(j_end = True)[:16]] = NodeType.INFLOW
BCtype[myMesh.get_node_index_list(i_end = True)] = NodeType.OUTFLOW

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

# print (myMesh.out_norm[0, :, :, :])

# create solver
# mySolver = SolverLaplace(myMesh, myCoeff, BCtype, OperatorFDM3D, dir_name)

# mySolver.start_solve()
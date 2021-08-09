import numpy as np
import math
import enum
import os
import shutil
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab, spsolve, gmres, lgmres
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class NodeType(enum.IntEnum):
    INTERIOR = 0
    DIRICHLET = 1


class SolverLaplace:

    def __init__(self, mesh, coeff, BCtype, opertor, dir_name):

        self._mesh = mesh
        self._coeff = coeff
        self._BCtype = BCtype
        self._opertor = opertor
        self._dir_name = dir_name
        self._assmemble()

        print('Laplace solver is created')

    def _assmemble(self):

        self._systemMatrix = sp.csr_matrix(
            self._opertor.der_2('i').multiply(self._coeff.get_inv_metric_tensor(0, 0)) +
            self._opertor.der_2('j').multiply(
                self._coeff.get_inv_metric_tensor(1, 1))
        )

        temp_matrix = self._opertor.no_operation()

        self._opertor.csr_zero_rows(self._systemMatrix, np.where(
            self._BCtype == NodeType.DIRICHLET))
        self._opertor.csr_zero_rows(temp_matrix, np.where(
            self._BCtype != NodeType.DIRICHLET))

        self._systemMatrix = self._systemMatrix + temp_matrix

        print('System matrix is created')

    # # ============================================
    # # Print data to file
    # # ============================================
    def printDate(self, DirName):

        FileName = 'Data.txt'

        np.savetxt('./' + DirName + '/' + FileName,
                   np.column_stack(
                       (self._mesh.X_flatten, self._mesh.Y_flatten, self._phi)),
                   fmt="%2.5f", delimiter=" , ")

    # ============================================
    # Solving processing
    # ============================================
    def start_solve(self):

        B = np.zeros_like(self._mesh.x_flatten().T)

        for i in range(len(self._mesh.x_flatten())):
            if self._mesh.Y_flatten[i] == 1:
                B[i] = 100

        self._phi = lgmres(self._systemMatrix, B)[0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cm = plt.cm.get_cmap('rainbow')
        pnt3d = ax.scatter(self._mesh.x_flatten(), self._mesh.y_flatten(),
                           self._mesh.z_flatten(), c=self._phi, cmap=cm)

        plt.show()

        self.printDate(self._dir_name)
        print('Calculation Completed!!!')

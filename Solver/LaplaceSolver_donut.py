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

    def __init__(self, mesh, coeff, BCtype, opertor, dirName):

        self._mesh = mesh
        self._coeff = coeff
        self._BCtype = BCtype
        self._opertor = opertor
        self._dirName = dirName
        self._assmemble()

    def _assmemble(self):

        IJK = ['i', 'j', 'k']

        sum = np.zeros(
            (self._mesh.node_number(), self._mesh.node_number()))

        # for i in range(3):
        #     for j in range(3):
        #         for k in range(3):
        #             sum -= self._opertor.der_1(IJK[i]).multiply(self._coeff.get_inv_metric_tensor(
        #                 j, k) * self._coeff.get_christoffel_symbol(i, j, k))

        self._SystemMatrix = sp.csr_matrix(
            self._opertor.der_2('i').multiply(self._coeff.get_inv_metric_tensor(0, 0)) +
            self._opertor.der_2('j').multiply(self._coeff.get_inv_metric_tensor(1, 1)) +
            self._opertor.der_2('k').multiply(self._coeff.get_inv_metric_tensor(2, 2)) +
            2*self._opertor.der_11('ij').multiply(self._coeff.get_inv_metric_tensor(0, 1)) +
            2*self._opertor.der_11('ik').multiply(self._coeff.get_inv_metric_tensor(0, 2)) +
            2*self._opertor.der_11('jk').multiply(self._coeff.get_inv_metric_tensor(1, 2)) -
            self._opertor.der_1('i').multiply(self._coeff.get_inv_metric_tensor(0, 0) * self._coeff.get_christoffel_symbol(0, 0, 0)) -
            self._opertor.der_1('i').multiply(self._coeff.get_inv_metric_tensor(0, 1) * self._coeff.get_christoffel_symbol(0, 0, 1)) -
            self._opertor.der_1('i').multiply(self._coeff.get_inv_metric_tensor(0, 2) * self._coeff.get_christoffel_symbol(0, 0, 2)) -
            self._opertor.der_1('i').multiply(self._coeff.get_inv_metric_tensor(1, 0) * self._coeff.get_christoffel_symbol(0, 1, 0)) -
            self._opertor.der_1('i').multiply(self._coeff.get_inv_metric_tensor(1, 1) * self._coeff.get_christoffel_symbol(0, 1, 1)) -
            self._opertor.der_1('i').multiply(self._coeff.get_inv_metric_tensor(1, 2) * self._coeff.get_christoffel_symbol(0, 1, 2)) -
            self._opertor.der_1('i').multiply(self._coeff.get_inv_metric_tensor(2, 0) * self._coeff.get_christoffel_symbol(0, 2, 0)) -
            self._opertor.der_1('i').multiply(self._coeff.get_inv_metric_tensor(2, 1) * self._coeff.get_christoffel_symbol(0, 2, 1)) -
            self._opertor.der_1('i').multiply(self._coeff.get_inv_metric_tensor(2, 2) * self._coeff.get_christoffel_symbol(0, 2, 2)) -
            self._opertor.der_1('j').multiply(self._coeff.get_inv_metric_tensor(0, 0) * self._coeff.get_christoffel_symbol(1, 0, 0)) -
            self._opertor.der_1('j').multiply(self._coeff.get_inv_metric_tensor(0, 1) * self._coeff.get_christoffel_symbol(1, 0, 1)) -
            self._opertor.der_1('j').multiply(self._coeff.get_inv_metric_tensor(0, 2) * self._coeff.get_christoffel_symbol(1, 0, 2)) -
            self._opertor.der_1('j').multiply(self._coeff.get_inv_metric_tensor(1, 0) * self._coeff.get_christoffel_symbol(1, 1, 0)) -
            self._opertor.der_1('j').multiply(self._coeff.get_inv_metric_tensor(1, 1) * self._coeff.get_christoffel_symbol(1, 1, 1)) -
            self._opertor.der_1('j').multiply(self._coeff.get_inv_metric_tensor(1, 2) * self._coeff.get_christoffel_symbol(1, 1, 2)) -
            self._opertor.der_1('j').multiply(self._coeff.get_inv_metric_tensor(2, 0) * self._coeff.get_christoffel_symbol(1, 2, 0)) -
            self._opertor.der_1('j').multiply(self._coeff.get_inv_metric_tensor(2, 1) * self._coeff.get_christoffel_symbol(1, 2, 1)) -
            self._opertor.der_1('j').multiply(self._coeff.get_inv_metric_tensor(2, 2) * self._coeff.get_christoffel_symbol(1, 2, 2)) -
            self._opertor.der_1('k').multiply(self._coeff.get_inv_metric_tensor(0, 0) * self._coeff.get_christoffel_symbol(2, 0, 0)) -
            self._opertor.der_1('k').multiply(self._coeff.get_inv_metric_tensor(0, 1) * self._coeff.get_christoffel_symbol(2, 0, 1)) -
            self._opertor.der_1('k').multiply(self._coeff.get_inv_metric_tensor(0, 2) * self._coeff.get_christoffel_symbol(2, 0, 2)) -
            self._opertor.der_1('k').multiply(self._coeff.get_inv_metric_tensor(1, 0) * self._coeff.get_christoffel_symbol(2, 1, 0)) -
            self._opertor.der_1('k').multiply(self._coeff.get_inv_metric_tensor(1, 1) * self._coeff.get_christoffel_symbol(2, 1, 1)) -
            self._opertor.der_1('k').multiply(self._coeff.get_inv_metric_tensor(1, 2) * self._coeff.get_christoffel_symbol(2, 1, 2)) -
            self._opertor.der_1('k').multiply(self._coeff.get_inv_metric_tensor(2, 0) * self._coeff.get_christoffel_symbol(2, 2, 0)) -
            self._opertor.der_1('k').multiply(self._coeff.get_inv_metric_tensor(2, 1) * self._coeff.get_christoffel_symbol(2, 2, 1)) -
            self._opertor.der_1('k').multiply(self._coeff.get_inv_metric_tensor(
                2, 2) * self._coeff.get_christoffel_symbol(2, 2, 2))
        )

        temp_matrix = self._opertor.no_operation()

        self._opertor.csr_zero_rows(self._SystemMatrix, np.where(
            self._BCtype == NodeType.DIRICHLET))

        self._opertor.csr_zero_rows(temp_matrix, np.where(
            self._BCtype != NodeType.DIRICHLET))

        self._SystemMatrix = self._SystemMatrix + temp_matrix

        print('System matrix is created')

    # # ============================================
    # # Print data to file
    # # ============================================
    def printDate(self, dirName):

        fileName = 'Data.txt'

        np.savetxt('./' + dirName + '/' + fileName,
                   np.column_stack(
                       (self._mesh.x_flatten(), self._mesh.y_flatten(), self._phi)),
                   fmt="%2.5f", delimiter=" , ")

    # ============================================
    # Solving processing
    # ============================================
    def start_solve(self):

        B = np.zeros_like(self._mesh.x())

        print(B[:, :, 0].shape)
        # B[:, 0, :] = 100
        # B[:, -1, :] = 50
        B[0, :, :] = 100
        B[-1, :, :] = 50
        B = np.reshape(B, self._mesh.node_number(), order='F')

        self._phi = lgmres(self._SystemMatrix, B)[0]

        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        cm = plt.cm.get_cmap('rainbow')
        pnt3d = ax.scatter(self._mesh.x_flatten(), self._mesh.y_flatten(),
                           self._mesh.z_flatten(), c=self._phi, cmap=cm)
        cbar = plt.colorbar(pnt3d)

        plt.show()

        self.printDate(self._dirName)
        print('Calculation Completed!!!')

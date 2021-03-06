import numpy as np
import math
import enum
import os
import shutil
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab, spsolve, gmres, lgmres

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class NodeType(enum.IntEnum):
    INTERIOR = 0
    DIRICHLET = 1
    NEUMANN = 2

    
# http://mragheb.com/NPRE%20498MC%20Monte%20Carlo%20Simulations%20in%20Engineering/Mixed%20Boundary%20Value%20Problems.pdf

class SolverLaplace:

    def __init__(self, MMesh, CCoeff, BBCtype, OOperatorFDM3D, ddir_name):
        
        self._Mesh = MMesh
        self._CCoeff = CCoeff
        self._BCtype = BBCtype
        self._Opertor = OOperatorFDM3D(MMesh)
        self._dir_name = ddir_name
        self._assmemble()
        
        print ('Laplace solver is created')


    def _assmemble(self):

        self._SystemMatrix = sp.csr_matrix(
                              self._Opertor.der_2('i').multiply(self._CCoeff.get_inv_metric_tensor(0,0)) + \
                              self._Opertor.der_2('j').multiply(self._CCoeff.get_inv_metric_tensor(1,1)) + \
                              2*self._Opertor.der_11('ij').multiply(self._CCoeff.get_inv_metric_tensor(1,2)) - \
                              self._Opertor.der_1('i').multiply( self._CCoeff.get_inv_metric_tensor(0,0) * self._CCoeff.get_christoffel_symbol(0,0,0)) - \
                              self._Opertor.der_1('j').multiply( self._CCoeff.get_inv_metric_tensor(0,0) * self._CCoeff.get_christoffel_symbol(1,0,0)) - \
                              self._Opertor.der_1('i').multiply( self._CCoeff.get_inv_metric_tensor(1,1) * self._CCoeff.get_christoffel_symbol(0,1,1)) - \
                              self._Opertor.der_1('j').multiply( self._CCoeff.get_inv_metric_tensor(1,1) * self._CCoeff.get_christoffel_symbol(1,1,1)) - \
                              self._Opertor.der_1('i').multiply( self._CCoeff.get_inv_metric_tensor(0,1) * self._CCoeff.get_christoffel_symbol(0,0,1)) - \
                              self._Opertor.der_1('j').multiply( self._CCoeff.get_inv_metric_tensor(0,1) * self._CCoeff.get_christoffel_symbol(1,0,1))
                                )
        
        self._Opertor.csr_zero_rows( self._SystemMatrix, np.where( self._BCtype != NodeType.INTERIOR ))


        # DIRICHLET
        temp_matrix1  = self._Opertor.no_operation()
        self._Opertor.csr_zero_rows( temp_matrix1, np.where(self._BCtype != NodeType.DIRICHLET))

    
        # NEUMANN
        outter_norm  = self._CCoeff.get_con_basis(0) / np.linalg.norm(self._CCoeff.get_con_basis(0), axis=1)[:,None]

        temp_matrix2 = (
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(0,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(0),outter_norm)).transpose() +
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(1,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(1),outter_norm)).transpose() +
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(2,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(2),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(0,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(0),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(1,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(1),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(2,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(2),outter_norm)).transpose() 
        )

        outter_norm  = (-self._CCoeff.get_con_basis(1) / np.linalg.norm(self._CCoeff.get_con_basis(1), axis=1)[:,None])

        temp_matrix3 = (
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(0,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(0),outter_norm)).transpose() +
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(1,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(1),outter_norm)).transpose() +
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(2,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(2),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(0,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(0),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(1,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(1),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(2,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(2),outter_norm)).transpose() 
        )

        # be careful of the intersection of two boudnary
        self._Opertor.csr_zero_rows( temp_matrix2, np.where( (self._Mesh.X_flatten != 0)))
        self._Opertor.csr_zero_rows( temp_matrix2, np.where( (self._Mesh.X_flatten == 0) & (self._Mesh.Y_flatten == 0) ))
        self._Opertor.csr_zero_rows( temp_matrix3, np.where( (self._Mesh.Y_flatten != 0)))


        self._SystemMatrix = self._SystemMatrix + temp_matrix1 + temp_matrix2 + temp_matrix3

        print ('System matrix is created')

    # # ============================================
    # # Print data to file
    # # ============================================
    def printDate(self, DirName):
        
        FileName = 'Data.txt'

        np.savetxt('./'+ DirName + '/' + FileName,
                   np.column_stack( (self._Mesh.X_flatten, self._Mesh.Y_flatten, self._Phi) ),
                   fmt="%2.5f", delimiter=" , " )
    # ============================================
    # Solving processing
    # ============================================
    def start_solve(self):
        
        B = np.zeros_like(self._Mesh.X_flatten)

        for i in range(len(self._Mesh.X_flatten)):
            if self._BCtype[i] == NodeType.DIRICHLET and self._Mesh.Y_flatten[i] == 1:
                B[i] = 1
            elif self._BCtype[i] == NodeType.DIRICHLET and self._Mesh.X_flatten[i] == 1:
                B[i] = 0

        self._Phi = lgmres(self._SystemMatrix, B)[0]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        my_cm = plt.cm.get_cmap('rainbow')
        pnt3d = ax.scatter( self._Mesh.X_flatten, self._Mesh.Y_flatten, self._Phi,c = self._Phi, cmap=my_cm)
        cbar=plt.colorbar(pnt3d)


        plt.show()
        
        self.printDate(self._dir_name)

        print ('Calculation Completed!!!')
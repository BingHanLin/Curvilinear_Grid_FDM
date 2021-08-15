import numpy as np
import math
import enum
import os
import shutil
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, cg, bicg, bicgstab, spsolve ,gmres ,lgmres
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class NodeType(enum.IntEnum):
    INTERIOR = 0
    TOPWALL = 1
    BOTTOMWALL_1 = 2
    BOTTOMWALL_2 = 3    
    INFLOW = 4
    OUTFLOW = 5


class SolverLaplace:

    def __init__(self, MMesh, CCoeff, BBCtype, OOperatorFDM3D, ddir_name):
        
        self._Mesh = MMesh
        self._CCoeff = CCoeff
        self._BCtype = BBCtype
        self._Opertor = OOperatorFDM3D(MMesh)
        self._dir_name = ddir_name
        self._assmemble()
        
        print ('Laplace solver is created')

    def flatten(self, data):
        return np.hstack(np.hstack(data))
        
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
                              self._Opertor.der_1('j').multiply( self._CCoeff.get_inv_metric_tensor(0,1) * self._CCoeff.get_christoffel_symbol(1,0,1)) - \
                              self._Opertor.der_1('i').multiply( self._CCoeff.get_inv_metric_tensor(1,0) * self._CCoeff.get_christoffel_symbol(0,1,0)) - \
                              self._Opertor.der_1('j').multiply( self._CCoeff.get_inv_metric_tensor(1,0) * self._CCoeff.get_christoffel_symbol(1,1,0))
                                )
        
        self._Opertor.csr_zero_rows( self._SystemMatrix, np.where( self._BCtype != NodeType.INTERIOR ))

        # NEUMANN
        temp_matrix1 = (
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(0,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(0), self._Mesh.out_norm)).transpose() +
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(1,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(1), self._Mesh.out_norm)).transpose() +
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(2,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(2), self._Mesh.out_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(0,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(0), self._Mesh.out_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(1,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(1), self._Mesh.out_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(2,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(2), self._Mesh.out_norm)).transpose() 
        )

        self._Opertor.csr_zero_rows( temp_matrix1, np.where( self._BCtype == NodeType.INTERIOR ))

        self._SystemMatrix = self._SystemMatrix + temp_matrix1


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
        B[np.where( (self._BCtype == NodeType.INFLOW))] = -5.0
        B[np.where( (self._BCtype == NodeType.OUTFLOW))] = 5.0

        self._Phi = lgmres(self._SystemMatrix, B)[0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cm = plt.cm.get_cmap('rainbow')
        pnt3d = ax.scatter( self._Mesh.X_flatten, self._Mesh.Y_flatten, self._Mesh.Z_flatten,c = self._Phi, cmap=cm)
        
        plt.show()
        
        self.printDate(self._dir_name)

        print ('Calculation Completed!!!')
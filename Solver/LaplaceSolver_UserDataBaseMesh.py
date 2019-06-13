import numpy as np
import math
import enum
import os
import shutil
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab, spsolve ,gmres ,lgmres
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
                              self._Opertor.der_1('j').multiply( self._CCoeff.get_inv_metric_tensor(0,1) * self._CCoeff.get_christoffel_symbol(1,0,1))
                                )
        
        self._Opertor.csr_zero_rows( self._SystemMatrix, np.where( self._BCtype != NodeType.INTERIOR ))

        # INFLOW(NEUMANN)
        outter_norm  = self._CCoeff.get_con_basis(1) / np.linalg.norm(self._CCoeff.get_con_basis(1), axis=1)[:,None]
        print (outter_norm[np.where(self._BCtype == NodeType.INFLOW)])
        temp_matrix1 = (
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(0,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(0),outter_norm)).transpose() +
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(1,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(1),outter_norm)).transpose() +
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(2,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(2),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(0,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(0),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(1,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(1),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(2,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(2),outter_norm)).transpose() 
        )

        # OUTFLOW(NEUMANN)
        outter_norm  = self._CCoeff.get_con_basis(0) / np.linalg.norm(self._CCoeff.get_con_basis(0), axis=1)[:,None]

        temp_matrix2 = (
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(0,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(0),outter_norm)).transpose() +
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(1,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(1),outter_norm)).transpose() +
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(2,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(2),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(0,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(0),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(1,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(1),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(2,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(2),outter_norm)).transpose() 
        )

        # BOTTOMWALL_1(NEUMANN)
        outter_norm  = -self._CCoeff.get_con_basis(0) / np.linalg.norm(self._CCoeff.get_con_basis(0), axis=1)[:,None]

        temp_matrix3 = (
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(0,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(0),outter_norm)).transpose() +
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(1,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(1),outter_norm)).transpose() +
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(2,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(2),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(0,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(0),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(1,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(1),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(2,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(2),outter_norm)).transpose() 
        )

        # BOTTOMWALL_2(NEUMANN)
        outter_norm  = -self._CCoeff.get_con_basis(1) / np.linalg.norm(self._CCoeff.get_con_basis(1), axis=1)[:,None]

        temp_matrix4 = (
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(0,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(0),outter_norm)).transpose() +
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(1,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(1),outter_norm)).transpose() +
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(2,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(2),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(0,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(0),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(1,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(1),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(2,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(2),outter_norm)).transpose() 
        )

        # TOPWALL(NEUMANN)
        outter_norm  = self._CCoeff.get_con_basis(1) / np.linalg.norm(self._CCoeff.get_con_basis(1), axis=1)[:,None]

        temp_matrix5 = (
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(0,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(0),outter_norm)).transpose() +
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(1,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(1),outter_norm)).transpose() +
        self._Opertor.der_1('i').transpose().multiply( self._CCoeff.get_inv_metric_tensor(2,0)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(2),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(0,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(0),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(1,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(1),outter_norm)).transpose() +
        self._Opertor.der_1('j').transpose().multiply( self._CCoeff.get_inv_metric_tensor(2,1)*np.einsum('ij, ij->i', self._CCoeff.get_co_basis(2),outter_norm)).transpose() 
        )

        self._Opertor.csr_zero_rows( temp_matrix1, np.where( (self._BCtype != NodeType.INFLOW)))
        self._Opertor.csr_zero_rows( temp_matrix2, np.where( (self._BCtype != NodeType.OUTFLOW)))
        self._Opertor.csr_zero_rows( temp_matrix3, np.where( (self._BCtype != NodeType.BOTTOMWALL_1)))
        self._Opertor.csr_zero_rows( temp_matrix4, np.where( (self._BCtype != NodeType.BOTTOMWALL_2)))
        self._Opertor.csr_zero_rows( temp_matrix5, np.where( (self._BCtype != NodeType.TOPWALL)))

        self._SystemMatrix = self._SystemMatrix + temp_matrix1 + temp_matrix2 + temp_matrix3 + temp_matrix4 + temp_matrix5

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
        B[np.where( (self._BCtype == NodeType.INFLOW))] = 5.0
        B[np.where( (self._BCtype == NodeType.OUTFLOW))] = 1.0

        self._Phi = lgmres(self._SystemMatrix, B)[0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cm = plt.cm.get_cmap('rainbow')
        pnt3d = ax.scatter( self._Mesh.X_flatten, self._Mesh.Y_flatten, self._Mesh.Z_flatten,c = self._Phi, cmap=cm)
        
        plt.show()
        
        self.printDate(self._dir_name)
        print ('Calculation Completed!!!')
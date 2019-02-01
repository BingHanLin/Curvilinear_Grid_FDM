import numpy as np
import math
import enum
import os
import shutil
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab, spsolve ,gmres ,lgmres

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

class NodeType(enum.IntEnum):
    INTERIOR = 0
    DIRICHLET = 1
    NEUMANN = 2

    


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
                              self._Opertor.der_2('i').multiply(self.flatten(self._CCoeff.inv_metric_tensor[...,0,0])) + \
                              self._Opertor.der_2('j').multiply(self.flatten(self._CCoeff.inv_metric_tensor[...,1,1])) 
                              )

        # DIRICHLET
        temp_matrix1  = self._Opertor.no_operation()

        self._Opertor.csr_zero_rows( self._SystemMatrix, np.where(self._BCtype == NodeType.DIRICHLET))
        self._Opertor.csr_zero_rows( temp_matrix1, np.where(self._BCtype != NodeType.DIRICHLET))
      

        co_basis_1_flatten = self._CCoeff.co_basis_1[...,:].reshape(-1,self._CCoeff.co_basis_1[...,:].shape[-1])
        co_basis_2_flatten = self._CCoeff.co_basis_2[...,:].reshape(-1,self._CCoeff.co_basis_2[...,:].shape[-1])
        co_basis_3_flatten = self._CCoeff.co_basis_3[...,:].reshape(-1,self._CCoeff.co_basis_3[...,:].shape[-1])

        con_basis_1_flatten = self._CCoeff.con_basis_1[...,:].reshape(-1,self._CCoeff.con_basis_1[...,:].shape[-1])
        con_basis_2_flatten = self._CCoeff.con_basis_2[...,:].reshape(-1,self._CCoeff.con_basis_2[...,:].shape[-1])
        con_basis_3_flatten = self._CCoeff.con_basis_3[...,:].reshape(-1,self._CCoeff.con_basis_3[...,:].shape[-1])
        
        
        temp_matrix2 = np.zeros((self._Mesh.node_number,self._Mesh.node_number))
        temp_matrix3 = np.zeros((self._Mesh.node_number,self._Mesh.node_number))

        for i in range(len(self._Mesh.X_flatten)):

            if self._BCtype[i] == NodeType.NEUMANN and self._Mesh.X_flatten[i] == 0:
                outter_norm  = con_basis_1_flatten[i] / np.linalg.norm(con_basis_1_flatten[i])

                temp_matrix2[i,:] =  self._CCoeff.inv_metric_tensor[...,0,0].flatten()[:, None][i]*self._Opertor.der_1('i')[i,:]*np.dot(co_basis_1_flatten[i],outter_norm) + \
                                            self._CCoeff.inv_metric_tensor[...,1,0].flatten()[:, None][i]*self._Opertor.der_1('i')[i,:]*np.dot(co_basis_2_flatten[i],outter_norm) + \
                                            self._CCoeff.inv_metric_tensor[...,2,0].flatten()[:, None][i]*self._Opertor.der_1('i')[i,:]*np.dot(co_basis_3_flatten[i],outter_norm) + \
                                            self._CCoeff.inv_metric_tensor[...,0,1].flatten()[:, None][i]*self._Opertor.der_1('j')[i,:]*np.dot(co_basis_1_flatten[i],outter_norm) + \
                                            self._CCoeff.inv_metric_tensor[...,1,1].flatten()[:, None][i]*self._Opertor.der_1('j')[i,:]*np.dot(co_basis_2_flatten[i],outter_norm) + \
                                            self._CCoeff.inv_metric_tensor[...,2,1].flatten()[:, None][i]*self._Opertor.der_1('j')[i,:]*np.dot(co_basis_3_flatten[i],outter_norm) 
                
                # self.__SystemMatrix[i,:] =  self._Opertor.der_1st('i')[i,:]*np.dot(con_basis_1_flatten[i],outter_norm) + \
                #                             self._Opertor.der_1st('j')[i,:]*np.dot(con_basis_2_flatten[i],outter_norm)

      
            elif self._BCtype[i] == NodeType.NEUMANN and self._Mesh.Y_flatten[i] == 0:
                outter_norm  = -con_basis_2_flatten[i] / np.linalg.norm(con_basis_2_flatten[i])
                
                temp_matrix3[i,:] =  self._CCoeff.inv_metric_tensor[...,0,0].flatten()[:, None][i]*self._Opertor.der_1('i')[i,:]*np.dot(co_basis_1_flatten[i],outter_norm) + \
                                            self._CCoeff.inv_metric_tensor[...,1,0].flatten()[:, None][i]*self._Opertor.der_1('i')[i,:]*np.dot(co_basis_2_flatten[i],outter_norm) + \
                                            self._CCoeff.inv_metric_tensor[...,2,0].flatten()[:, None][i]*self._Opertor.der_1('i')[i,:]*np.dot(co_basis_3_flatten[i],outter_norm) + \
                                            self._CCoeff.inv_metric_tensor[...,0,1].flatten()[:, None][i]*self._Opertor.der_1('j')[i,:]*np.dot(co_basis_1_flatten[i],outter_norm) + \
                                            self._CCoeff.inv_metric_tensor[...,1,1].flatten()[:, None][i]*self._Opertor.der_1('j')[i,:]*np.dot(co_basis_2_flatten[i],outter_norm) + \
                                            self._CCoeff.inv_metric_tensor[...,2,1].flatten()[:, None][i]*self._Opertor.der_1('j')[i,:]*np.dot(co_basis_3_flatten[i],outter_norm) 
        
                # self.__SystemMatrix[i,:] =  self._Opertor.der_1st('i')[i,:]*np.dot(con_basis_1_flatten[i],outter_norm) + \
                #                             self._Opertor.der_1st('j')[i,:]*np.dot(con_basis_2_flatten[i],outter_norm)

        temp_matrix2 = sp.csr_matrix(temp_matrix2)
        temp_matrix3 = sp.csr_matrix(temp_matrix3)
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
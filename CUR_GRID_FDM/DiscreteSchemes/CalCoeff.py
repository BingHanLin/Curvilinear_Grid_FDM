import numpy as np

class CalCoeff:
    '''
    Calculate essential coefficients for curvilinear coordinates
    '''

    def __init__(self, Mesh):

        self._MESH = Mesh
        self.update_Coeff()
        self._cal_out_norm()

    def update_Coeff(self):

        self._cal_co_basis_vector()
        self._cal_jacobian()
        self._cal_con_basis_vector()
        self._cal_metric_tensor()
        self._cal_christoffel_symbol()


    def _cal_co_basis_vector(self):
        
        # create covariant basis vectors
        self._co_basis_1 = np.zeros((self._MESH.mesh_size + (3,)))
        self._co_basis_2 = np.zeros_like(self._co_basis_1)
        self._co_basis_3 = np.zeros_like(self._co_basis_1)
        
        # get gradient along i, j, k direction
        if self._MESH.mesh_size[2] != 1:
            self._co_basis_1[...,0], self._co_basis_2[...,0], self._co_basis_3[...,0]  = np.gradient(self._MESH.X)       
            self._co_basis_1[...,1], self._co_basis_2[...,1], self._co_basis_3[...,1]  = np.gradient(self._MESH.Y)       
            self._co_basis_1[...,2], self._co_basis_2[...,2], self._co_basis_3[...,2]  = np.gradient(self._MESH.Z)       
        
        else:
            self._co_basis_1[...,0], self._co_basis_2[...,0]  = np.gradient(self._MESH.X, axis = (0,1))       
            self._co_basis_1[...,1], self._co_basis_2[...,1]  = np.gradient(self._MESH.Y, axis = (0,1))       
            self._co_basis_1[...,2], self._co_basis_2[...,2]  = np.gradient(self._MESH.Z, axis = (0,1))       

            self._co_basis_3[...,2].fill(1.0)


    def _cal_jacobian(self):
        
        # calculate jacobian of each basis
        self._jacobian = np.cross(self._co_basis_1, self._co_basis_2)
        self._jacobian = np.einsum( 'ijkl,ijkl->ijk',self._jacobian, self._co_basis_3 )


    def _cal_con_basis_vector(self):
        
        # create contravariant basis vectors
        self._con_basis_1 = np.cross(self._co_basis_2, self._co_basis_3) / self._jacobian[...,None]
        self._con_basis_2 = np.cross(self._co_basis_3, self._co_basis_1) / self._jacobian[...,None]
        self._con_basis_3 = np.cross(self._co_basis_1, self._co_basis_2) / self._jacobian[...,None]


    def _cal_metric_tensor(self):
        
        # create metric_tensor
        self._metric_tensor = np.zeros((self._MESH.mesh_size + (3,3)))

        # https://zhuanlan.zhihu.com/p/27739282 如何理解和使用NumPy.einsum？
        self._metric_tensor[...,0,0] = np.einsum( 'ijkl,ijkl->ijk',self._co_basis_1, self._co_basis_1 )
        self._metric_tensor[...,1,1] = np.einsum( 'ijkl,ijkl->ijk',self._co_basis_2, self._co_basis_2 )
        self._metric_tensor[...,2,2] = np.einsum( 'ijkl,ijkl->ijk',self._co_basis_3, self._co_basis_3 )
        self._metric_tensor[...,0,1] = np.einsum( 'ijkl,ijkl->ijk',self._co_basis_1, self._co_basis_2 )
        self._metric_tensor[...,0,2] = np.einsum( 'ijkl,ijkl->ijk',self._co_basis_1, self._co_basis_3 )
        self._metric_tensor[...,1,2] = np.einsum( 'ijkl,ijkl->ijk',self._co_basis_2, self._co_basis_3 )

        self._metric_tensor[...,1,0] = np.copy( self._metric_tensor[...,0,1] )
        self._metric_tensor[...,2,0] = np.copy( self._metric_tensor[...,0,2] )
        self._metric_tensor[...,2,1] = np.copy( self._metric_tensor[...,1,2] )

        # create inv_metric_tensor g^ij
        self._inv_metric_tensor = np.zeros((self._MESH.mesh_size + (3,3)))
        
        self._inv_metric_tensor[...,0,0] = np.einsum( 'ijkl,ijkl->ijk',self._con_basis_1, self._con_basis_1 )
        self._inv_metric_tensor[...,1,1] = np.einsum( 'ijkl,ijkl->ijk',self._con_basis_2, self._con_basis_2 )
        self._inv_metric_tensor[...,2,2] = np.einsum( 'ijkl,ijkl->ijk',self._con_basis_3, self._con_basis_3 )
        self._inv_metric_tensor[...,0,1] = np.einsum( 'ijkl,ijkl->ijk',self._con_basis_1, self._con_basis_2 )
        self._inv_metric_tensor[...,0,2] = np.einsum( 'ijkl,ijkl->ijk',self._con_basis_1, self._con_basis_3 )
        self._inv_metric_tensor[...,1,2] = np.einsum( 'ijkl,ijkl->ijk',self._con_basis_2, self._con_basis_3 )

        self._inv_metric_tensor[...,1,0] = np.copy( self._inv_metric_tensor[...,0,1] )
        self._inv_metric_tensor[...,2,0] = np.copy( self._inv_metric_tensor[...,0,2] )
        self._inv_metric_tensor[...,2,1] = np.copy( self._inv_metric_tensor[...,1,2] )



    def _cal_christoffel_symbol(self):
        # https://johnkerl.org/gdg/gdgprojnotes.pdf
        # create christoffel symbols
        self._christoffel_symbol = np.zeros((self._MESH.mesh_size + (3,3,3)))

        # get gradient along i, j, k direction
        D_metric_tensor = np.zeros((self._metric_tensor.shape + (3,)))

        for i in range(3):
            for j in range(3):
                if self._MESH.mesh_size[2] != 1:
                    D_metric_tensor[...,i,j,0], D_metric_tensor[...,i,j,1], D_metric_tensor[...,i,j,2] = \
                    np.gradient(self._metric_tensor[...,i,j])
                else:
                    D_metric_tensor[...,i,j,0], D_metric_tensor[...,i,j,1] = \
                    np.gradient(self._metric_tensor[...,i,j], axis = (0,1))

                    D_metric_tensor[...,i,j,2].fill(0.0)   


        # calculate christoffel symbols
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for m in range(3):
                        self._christoffel_symbol[...,i,j,k] = \
                            self._christoffel_symbol[...,i,j,k] + \
                            0.5*self._inv_metric_tensor[...,i,m]*( 
                                D_metric_tensor[...,m,j,k] +
                                D_metric_tensor[...,k,m,j] -
                                D_metric_tensor[...,j,k,m] )

        # print (self._christoffel_symbol)

    def _cal_out_norm(self):

        self._MESH.out_norm = np.zeros((self._MESH.mesh_size + (3,)))
        print ( self._con_basis_1[...,:])
        print ( np.linalg.norm(self._con_basis_1[...,:], axis=-1)[:,None])
        print ( self._con_basis_1[...,:]/np.linalg.norm(self._con_basis_1[...,:], axis=-1)[:,None] ) 

        # self._MESH.out_norm[0, :, :, :] = self._con_basis_1[...,:]
        # # -self.get_con_basis(0) / np.linalg.norm(self.get_con_basis(0), axis=1)[:,None]

    def get_co_basis(self, idx):

        if idx == 0:
            return np.reshape(self._co_basis_1[...,:], (self._MESH.node_number,3), order='F')
        elif idx == 1:
            return np.reshape(self._co_basis_2[...,:], (self._MESH.node_number,3), order='F')
        elif idx == 2:
            return np.reshape(self._co_basis_3[...,:], (self._MESH.node_number,3), order='F')


    def get_con_basis(self, idx):

        if idx == 0:
            return np.reshape(self._con_basis_1[...,:], (self._MESH.node_number,3), order='F')
        elif idx == 1:
            return np.reshape(self._con_basis_2[...,:], (self._MESH.node_number,3), order='F')
        elif idx == 2:
            return np.reshape(self._con_basis_3[...,:], (self._MESH.node_number,3), order='F')
        

    def get_metric_tensor(self, idx1, idx2):

        return np.reshape(self._metric_tensor[...,idx1, idx2], self._MESH.node_number, order='F') 
    

    def get_inv_metric_tensor(self, idx1, idx2):

        return np.reshape(self._inv_metric_tensor[...,idx1, idx2], self._MESH.node_number, order='F')  


    def get_christoffel_symbol(self, idx1, idx2, idx3):

        return np.reshape(self._christoffel_symbol[...,idx1, idx2, idx3], self._MESH.node_number, order='F')  

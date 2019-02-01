import numpy as np

class CalCoeff:
    '''
    Calculate essential coefficients for curvilinear coordinates
    '''

    def __init__(self, Mesh):

        self.__MESH = Mesh
        self.update_Coeff()

    def update_Coeff(self):

        self.__cal_co_basis_vector()
        self.__cal_jacobian()
        self.__cal_con_basis_vector()
        self.__cal_metric_tensor()
        self.__cal_christoffel_symbol()


    def __cal_co_basis_vector(self):
        
        # create covariant basis vectors
        self.co_basis_1 = np.zeros((self.__MESH.mesh_size + (3,)))
        self.co_basis_2 = np.zeros_like(self.co_basis_1)
        self.co_basis_3 = np.zeros_like(self.co_basis_1)
        
        # get gradient along i, j, k direction
        if self.__MESH.mesh_size[2] != 1:
            self.co_basis_1[...,0], self.co_basis_2[...,0], self.co_basis_3[...,0]  = np.gradient(self.__MESH.X)       
            self.co_basis_1[...,1], self.co_basis_2[...,1], self.co_basis_3[...,1]  = np.gradient(self.__MESH.Y)       
            self.co_basis_1[...,2], self.co_basis_2[...,2], self.co_basis_3[...,2]  = np.gradient(self.__MESH.Z)       
        
        else:
            self.co_basis_1[...,0], self.co_basis_2[...,0]  = np.gradient(self.__MESH.X, axis = (0,1))       
            self.co_basis_1[...,1], self.co_basis_2[...,1]  = np.gradient(self.__MESH.Y, axis = (0,1))       
            self.co_basis_1[...,2], self.co_basis_2[...,2]  = np.gradient(self.__MESH.Z, axis = (0,1))       

            self.co_basis_3[...,2].fill(1.0)


    def __cal_jacobian(self):
        
        # calculate jacobian of each basis
        self.jacobian = np.cross(self.co_basis_1, self.co_basis_2)
        self.jacobian = np.einsum( 'ijkl,ijkl->ijk',self.jacobian, self.co_basis_3 )


    def __cal_con_basis_vector(self):
        
        # create contravariant basis vectors
        self.con_basis_1 = np.cross(self.co_basis_2, self.co_basis_3) / self.jacobian[...,None]
        self.con_basis_2 = np.cross(self.co_basis_3, self.co_basis_1) / self.jacobian[...,None]
        self.con_basis_3 = np.cross(self.co_basis_1, self.co_basis_2) / self.jacobian[...,None]


    def __cal_metric_tensor(self):
        
        # create metric_tensor
        self.metric_tensor = np.zeros((self.__MESH.mesh_size + (3,3)))

        # https://zhuanlan.zhihu.com/p/27739282 如何理解和使用NumPy.einsum？
        self.metric_tensor[...,0,0] = np.einsum( 'ijkl,ijkl->ijk',self.co_basis_1, self.co_basis_1 )
        self.metric_tensor[...,1,1] = np.einsum( 'ijkl,ijkl->ijk',self.co_basis_2, self.co_basis_2 )
        self.metric_tensor[...,2,2] = np.einsum( 'ijkl,ijkl->ijk',self.co_basis_3, self.co_basis_3 )
        self.metric_tensor[...,0,1] = np.einsum( 'ijkl,ijkl->ijk',self.co_basis_1, self.co_basis_2 )
        self.metric_tensor[...,0,2] = np.einsum( 'ijkl,ijkl->ijk',self.co_basis_1, self.co_basis_3 )
        self.metric_tensor[...,1,2] = np.einsum( 'ijkl,ijkl->ijk',self.co_basis_2, self.co_basis_3 )

        self.metric_tensor[...,1,0] = np.copy( self.metric_tensor[...,0,1] )
        self.metric_tensor[...,2,0] = np.copy( self.metric_tensor[...,0,2] )
        self.metric_tensor[...,2,1] = np.copy( self.metric_tensor[...,1,2] )

        # create inv_metric_tensor g^ij
        self.inv_metric_tensor = np.zeros((self.__MESH.mesh_size + (3,3)))
        
        self.inv_metric_tensor[...,0,0] = np.einsum( 'ijkl,ijkl->ijk',self.con_basis_1, self.con_basis_1 )
        self.inv_metric_tensor[...,1,1] = np.einsum( 'ijkl,ijkl->ijk',self.con_basis_2, self.con_basis_2 )
        self.inv_metric_tensor[...,2,2] = np.einsum( 'ijkl,ijkl->ijk',self.con_basis_3, self.con_basis_3 )
        self.inv_metric_tensor[...,0,1] = np.einsum( 'ijkl,ijkl->ijk',self.con_basis_1, self.con_basis_2 )
        self.inv_metric_tensor[...,0,2] = np.einsum( 'ijkl,ijkl->ijk',self.con_basis_1, self.con_basis_3 )
        self.inv_metric_tensor[...,1,2] = np.einsum( 'ijkl,ijkl->ijk',self.con_basis_2, self.con_basis_3 )

        self.inv_metric_tensor[...,1,0] = np.copy( self.inv_metric_tensor[...,0,1] )
        self.inv_metric_tensor[...,2,0] = np.copy( self.inv_metric_tensor[...,0,2] )
        self.inv_metric_tensor[...,2,1] = np.copy( self.inv_metric_tensor[...,1,2] )



    def __cal_christoffel_symbol(self):
        # https://johnkerl.org/gdg/gdgprojnotes.pdf
        # create christoffel symbols
        self.christoffel_symbol = np.zeros((self.__MESH.mesh_size + (3,3,3)))

        # get gradient along i, j, k direction
        D_metric_tensor = np.zeros((self.metric_tensor.shape + (3,)))

        for i in range(3):
            for j in range(3):
                if self.__MESH.mesh_size[2] != 1:
                    D_metric_tensor[...,i,j,0], D_metric_tensor[...,i,j,1], D_metric_tensor[...,i,j,2] = \
                    np.gradient(self.metric_tensor[...,i,j])
                else:
                    D_metric_tensor[...,i,j,0], D_metric_tensor[...,i,j,1] = \
                    np.gradient(self.metric_tensor[...,i,j], axis = (0,1))

                    D_metric_tensor[...,i,j,2].fill(0.0)   


        # calculate christoffel symbols
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for m in range(3):
                        self.christoffel_symbol[...,i,j,k] = \
                            self.christoffel_symbol[...,i,j,k] + \
                            0.5*self.inv_metric_tensor[...,i,m]*( 
                                D_metric_tensor[...,m,j,k] +
                                D_metric_tensor[...,k,m,j] -
                                D_metric_tensor[...,j,k,m] )

        # print (self.christoffel_symbol)

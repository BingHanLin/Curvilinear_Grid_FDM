
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class CurveRectangularMesh:

    def __init__(self, Lx, nx, Ly = None, ny = None, Lz = None, nz = None):
        
        if ny == None and nz == None:
            self.dim = 1
        elif nz == None:   
            self.dim = 2
        else:
            self.dim = 3

        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        self.nx = nx
        self.ny = ny
        self.nz = nz


        if self.dim == 1:
            self.ny = 1
            self.nz = 1
        elif self.dim == 2:
            self.nz = 1


        self.node_number =  self.nx*self.ny*self.nz

        self._create_grid()


    def _create_grid(self):

        x_ = np.linspace(0, self.Lx, self.nx)

        if self.dim == 1:
            y_ = np.linspace(0, 1, 1)
        else:
            y_ = np.linspace(0, self.Ly, self.ny)

        if self.dim == 1 or self.dim == 2:
            z_ = np.linspace(0, 1, 1)
        else:
            z_ = np.linspace(0, self.Lz, self.nz)


        self.X, self.Y, self.Z = np.meshgrid(x_, y_, z_, indexing='ij')

        assert np.all(self.X[:,0,0] == x_)
        assert np.all(self.Y[0,:,0] == y_)
        assert np.all(self.Z[0,0,:] == z_)
        
        amp = self.Ly*0.2

        for i in range(len(self.Y[:,0,0])):
            self.Y[i,:,:] = self.Y[i,:,:] + amp*np.sin(self.X[i,0,0]*3.14)
        

        self.X_flatten = np.reshape(self.X, self.node_number, order='F')
        self.Y_flatten = np.reshape(self.Y, self.node_number, order='F')
        self.Z_flatten = np.reshape(self.Z, self.node_number, order='F')

        self.mesh_size = self.X.shape
        
        print ('normal mesh (nx,ny,nz) = ({}) is created'.format(self.mesh_size) )


    def get_node_index_list(self, i_front = False, i_end = False
                                , j_front = False, j_end = False
                                , k_front = False, k_end = False
                                , interior = False, inverse = False):
        '''
        i_front, i_end, j_front, j_end, k_front, k_end, interior, if not define then return all
        '''
        
        index_list = np.array([], dtype=int)
        node_num_list =  np.zeros(self.mesh_size)

        if i_front == True:
            node_num_list[0,:,:] = 1 

        if i_end == True:
            node_num_list[-1,:,:] = 1 

        if j_front == True:
           node_num_list[:,0,:] = 1 

        if j_end == True:
            node_num_list[:,-1,:] = 1 

        if k_front == True:
            node_num_list[:,:,0] = 1 

        if k_end == True:
            node_num_list[:,:,-1] = 1 

        node_num_list = np.reshape(node_num_list, self.node_number, order='F') 
        i_bool = np.where(node_num_list == 1)
        index_list = np.append(index_list, i_bool)

        if inverse == True:
            return np.setdiff1d(np.arange(self.node_number), index_list) 
        else:
            return index_list
            

            

    def plot_grid(self, BCtype=False):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        marker_size = 50

        if BCtype is False:
            ax.scatter(self.X_flatten, self.Y_flatten, self.Z_flatten, marker='o', c='g', s = marker_size, label = 'nodes')

        else:
            for nodetype in set(BCtype):
                
                mask = np.ma.masked_where(BCtype!=nodetype, BCtype)

                ax.scatter(np.ma.masked_array(self.X_flatten, mask.mask),
                            np.ma.masked_array(self.Y_flatten, mask.mask),
                            np.ma.masked_array(self.Z_flatten, mask.mask),
                            marker='o', s = marker_size, label = 'nodetype: {}'.format(int(nodetype)))


        plt.legend()
        plt.axis('equal')
        plt.show()






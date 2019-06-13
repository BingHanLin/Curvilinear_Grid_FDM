
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class DonutMesh:

    def __init__(self, R_inner, R_outer, n_theta, n_radius, Lz=None, nz=None):
        
        if nz == None:   
            self.dim = 2
        else:
            self.dim = 3

        self.R_inner = R_inner
        self.R_outer = R_outer

        self.n_theta = n_theta
        self.n_radius = n_radius

        self.Lz = Lz
        self.nz = nz

        if self.dim == 2:
            self.nz = 1

        self.node_number =  self.n_theta* self.n_radius*self.nz

        self.__create_grid()


    def __create_grid(self):

        radius_ = np.linspace(self.R_inner, self.R_outer, self.n_radius)
        theta_ = np.linspace(0, 2*np.pi-2*np.pi/self.n_theta, self.n_theta)
       
        if self.dim == 2:
            z_ = np.linspace(0, 1, 1)
        else:
            z_ = np.linspace(0, self.Lz, self.nz)

        radius_matrix, theta_matrix = np.meshgrid(radius_, theta_)

        _, _, self.Z = np.meshgrid(radius_, theta_, z_, indexing='ij')

        self.X = radius_matrix * np.cos(theta_matrix)
        self.Y = radius_matrix * np.sin(theta_matrix)

        self.X = np.dstack([self.X]*self.nz)
        self.Y = np.dstack([self.Y]*self.nz)
        self.X = self.X.transpose((1, 0, 2))
        self.Y = self.Y.transpose((1, 0, 2))

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


if __name__ == "__main__":


    MESH = DonutMesh(5, 8, 30, 5)
    print (MESH.mesh_size)
    MESH.plot_grid()


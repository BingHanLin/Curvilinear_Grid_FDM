
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from .MeshGeneratorTFI import Vinokur_Distribution

class DataBase:

    def __init__(self):

        # edges for 2D domain
        self.edge1 = {}
        self.edge2 = {}
        self.edge3 = {}
        self.edge4 = {}
        self.edge_1_list = []
        self.edge_2_list = []
        self.edge_3_list = []
        self.edge_4_list = []

        # first edge
        num_edge_1a  = 31
        
        x = [ (-4-0.1*( (i+1)-1)) for i in range(num_edge_1a)]
        y = [ 0 for i in range(num_edge_1a)]
        z = [ 0 for i in range(num_edge_1a)]

        node = np.asarray([x, y, z])
        self.edge_1_list.append({"node": node})

        # second edge
        num_edge_2a  = 51
        theta = [ 0.01*((i+1) - 1)*np.pi for i in range(num_edge_2a)]
        x = [ -3-np.cos(theta[i]) for i in range(num_edge_2a)] 
        y = [ np.sin(theta[i]) for i in range(num_edge_2a)] 
        z = [ 0 for i in range(num_edge_2a)] 

        node = np.asarray([x, y, z])
        self.edge_2_list.append({"node": node})


        num_edge_2b  = 51
        x = [ -3+0.1*((i+1)-1) for i in range(num_edge_2b)] 
        y = [ 1 for i in range(num_edge_2b)] 
        z = [ 0 for i in range(num_edge_2b)] 

        node = np.asarray([x, y, z])
        self.edge_2_list.append({"node": node})


        num_edge_2c  = 51
        theta = [ 0.01*(51-(i+1))*np.pi for i in range(num_edge_2c)]
        x = [ 2+2*np.cos(theta[i]) for i in range(num_edge_2c)] 
        y = [ np.sin(theta[i]) for i in range(num_edge_2c)] 
        z = [ 0 for i in range(num_edge_2c)] 

        node = np.asarray([x, y, z])
        self.edge_2_list.append({"node": node})

        num_edge_2d  = 51
        x = [ 4+0.1*((i+1)-1) for i in range(num_edge_2d)] 
        y = [ 0 for i in range(num_edge_2d)] 
        z = [ 0 for i in range(num_edge_2d)] 

        node = np.asarray([x, y, z])
        self.edge_2_list.append({"node": node})

        # third edge
        num_edge_3a  = 21
        x = [ 9.0 for i in range(num_edge_3a)]
        y = [ 0.2*((i+1)-1) for i in range(num_edge_3a)]
        z = [ 0 for i in range(num_edge_3a)]

        node = np.asarray([x, y, z])
        self.edge_3_list.append({"node": node})

        # fourth edge
        num_edge_4a  = 21
        x = [ -7 for i in range(num_edge_4a)]
        y = [ 0.2*((i+1)-1) for i in range(num_edge_4a)]
        z = [ 0 for i in range(num_edge_4a)]

        node = np.asarray([x, y, z])
        self.edge_4_list.append({"node": node})

        num_edge_4b  = 21
        x = [ -7+0.2*((i+1)-1) for i in range(num_edge_4b)] 
        y = [ 4.0 for i in range(num_edge_4b)] 
        z = [ 0 for i in range(num_edge_4b)] 

        node = np.asarray([x, y, z])
        self.edge_4_list.append({"node": node})

        num_edge_4c  = 26
        x = [ -3+0.2*((i+1)-1) for i in range(num_edge_4c)] 
        y = [ 4.0 for i in range(num_edge_4c)] 
        z = [ 0 for i in range(num_edge_4c)] 

        node = np.asarray([x, y, z])
        self.edge_4_list.append({"node": node})

        num_edge_4d  = 16
        x = [ 2+0.2*((i+1)-1) for i in range(num_edge_4d)] 
        y = [ 4.0 for i in range(num_edge_4d)] 
        z = [ 0 for i in range(num_edge_4d)] 

        node = np.asarray([x, y, z])
        self.edge_4_list.append({"node": node})

        num_edge_4e  = 21
        x = [ 5+0.2*((i+1)-1) for i in range(num_edge_4e)] 
        y = [ 4.0 for i in range(num_edge_4e)] 
        z = [ 0 for i in range(num_edge_4e)] 

        node = np.asarray([x, y, z])
        self.edge_4_list.append({"node": node})

        self.summary()



    def summary(self):
        print ("Database is created.")
        print ("edge 1 : {:>5} nodes".format( sum([edge['node'].shape[1] for edge in self.edge_1_list]) ))
        print ("edge 2 : {:>5} nodes".format( sum([edge['node'].shape[1] for edge in self.edge_2_list]) ))
        print ("edge 3 : {:>5} nodes".format( sum([edge['node'].shape[1] for edge in self.edge_3_list]) ))
        print ("edge 4 : {:>5} nodes".format( sum([edge['node'].shape[1] for edge in self.edge_4_list]) ))


class UserDataBaseMesh:

    def __init__(self, database):

        self.__create_grid(database)


    def __create_grid(self, database):

        database.edge1['node'] = Vinokur_Distribution(database.edge_1_list[0], 0.01, 0.4, 21)
        print (database.edge1['node'])

        # edge1['node'] = Vinokur_Distribution(database.edge_1a, 0.01, 0.4, 21)
        # edge1['sum_arclength'], edge1['arclength_ratio'] = cal_arclength(edge1)

        # edge2['node'] = Vinokur_Distribution(database.edge_2a, 0.02, 0.1, 31)
        # edge2['node'] = np.hstack((edge2['node'][:, :-1], Vinokur_Distribution(database.edge_2b, 0.1, 0.1, 41) ))
        # edge2['node'] = np.hstack((edge2['node'][:, :-1], Vinokur_Distribution(database.edge_2c, 0.1, 0.04, 31) ))
        # edge2['node'] = np.hstack((edge2['node'][:, :-1], Vinokur_Distribution(database.edge_2d, 0.05, 0.5, 25) ))
        # edge2['sum_arclength'], edge2['arclength_ratio'] = cal_arclength(edge2)

        # edge3['node'] = Vinokur_Distribution(database.edge_3a, 0.01, 0.4, 21)
        # edge3['sum_arclength'], edge3['arclength_ratio'] = cal_arclength(edge3)

        # edge4['node'] = Vinokur_Distribution(database.edge_4a, 0.1, 0.3, 16)
        # edge4['node'] = np.hstack((edge4['node'][:,:-1], Vinokur_Distribution(database.edge_4b, 0.3, 0.1, 16) ))
        # edge4['node'] = np.hstack((edge4['node'][:,:-1], Vinokur_Distribution(database.edge_4c, 0.1, 0.1, 41) ))
        # edge4['node'] = np.hstack((edge4['node'][:,:-1], Vinokur_Distribution(database.edge_4d, 0.1, 0.05, 31) ))
        # edge4['node'] = np.hstack((edge4['node'][:,:-1], Vinokur_Distribution(database.edge_4e, 0.05, 0.5, 25) ))
        # edge4['sum_arclength'], edge4['arclength_ratio'] = cal_arclength(edge4)

        # TFI(edge1, edge3, edge2, edge4)



    #     x_ = np.linspace(0, self.Lx, self.nx)

    #     if self.dim == 1:
    #         y_ = np.linspace(0, 1, 1)
    #     else:
    #         y_ = np.linspace(0, self.Ly, self.ny)

    #     if self.dim == 1 or self.dim == 2:
    #         z_ = np.linspace(0, 1, 1)
    #     else:
    #         z_ = np.linspace(0, self.Lz, self.nz)


    #     self.X, self.Y, self.Z = np.meshgrid(x_, y_, z_, indexing='ij')

    #     assert np.all(self.X[:,0,0] == x_)
    #     assert np.all(self.Y[0,:,0] == y_)
    #     assert np.all(self.Z[0,0,:] == z_)


    #     self.X_flatten = np.reshape(self.X, self.node_number, order='F')
    #     self.Y_flatten = np.reshape(self.Y, self.node_number, order='F')
    #     self.Z_flatten = np.reshape(self.Z, self.node_number, order='F')

    #     self.mesh_size = self.X.shape
        
    #     print ('normal mesh (nx,ny,nz) = ({}) is created'.format(self.mesh_size) )


    # def get_node_index_list(self, i_front = False, i_end = False
    #                             , j_front = False, j_end = False
    #                             , k_front = False, k_end = False
    #                             , interior = False, inverse = False):
    #     '''
    #     i_front, i_end, j_front, j_end, k_front, k_end, interior, if not define then return all
    #     '''
        
    #     index_list = np.array([], dtype=int)
    #     node_num_list =  np.zeros(self.mesh_size)

    #     if i_front == True:
    #         node_num_list[0,:,:] = 1 

    #     if i_end == True:
    #         node_num_list[-1,:,:] = 1 

    #     if j_front == True:
    #        node_num_list[:,0,:] = 1 

    #     if j_end == True:
    #         node_num_list[:,-1,:] = 1 

    #     if k_front == True:
    #         node_num_list[:,:,0] = 1 

    #     if k_end == True:
    #         node_num_list[:,:,-1] = 1 

    #     node_num_list = np.reshape(node_num_list, self.node_number, order='F') 
    #     i_bool = np.where(node_num_list == 1)
    #     index_list = np.append(index_list, i_bool)

    #     if inverse == True:
    #         return np.setdiff1d(np.arange(self.node_number), index_list) 
    #     else:
    #         return index_list
            

            

    # def plot_grid(self, BCtype=False):

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     marker_size = 50

    #     if BCtype is False:
    #         ax.scatter(self.X_flatten, self.Y_flatten, self.Z_flatten, marker='o', c='g', s = marker_size, label = 'nodes')

    #     else:
    #         for nodetype in set(BCtype):
                
    #             mask = np.ma.masked_where(BCtype!=nodetype, BCtype)

    #             ax.scatter(np.ma.masked_array(self.X_flatten, mask.mask),
    #                         np.ma.masked_array(self.Y_flatten, mask.mask),
    #                         np.ma.masked_array(self.Z_flatten, mask.mask),
    #                         marker='o', s = marker_size, label = 'nodetype: {}'.format(int(nodetype)))


    #     plt.legend()
    #     plt.axis('equal')
    #     plt.show()






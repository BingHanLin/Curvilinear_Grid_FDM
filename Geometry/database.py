
import numpy as np

class DataBase:

    def __init__(self):

        num_edge_1a  = 31
        
        self.edge_1a = {}

        x = [ (-4-0.1*( (i+1)-1)) for i in range(num_edge_1a)]
        y = [ 0 for i in range(num_edge_1a)]
        z = [ 0 for i in range(num_edge_1a)]

        node = np.asarray([x, y, z])

        self.edge_1a['node'] = node
        self.edge_1a['sum_arclength'], self.edge_1a['arclength_ratio'] = self.cal_arclength(self.edge_1a)  
        
        # =======================================================
        self.edge_2a = {}

        num_edge_2a  = 51
        theta = [ 0.01*((i+1) - 1)*np.pi for i in range(num_edge_2a)]
        x = [ -3-np.cos(theta[i]) for i in range(num_edge_2a)] 
        y = [ np.sin(theta[i]) for i in range(num_edge_2a)] 
        z = [ 0 for i in range(num_edge_2a)] 

        node = np.asarray([x, y, z])

        self.edge_2a['node'] = node
        self.edge_2a['sum_arclength'], self.edge_2a['arclength_ratio'] = self.cal_arclength(self.edge_2a)  

        # =======================================================
        self.edge_2b = {}
        
        num_edge_2b  = 51
        x = [ -3+0.1*((i+1)-1) for i in range(num_edge_2b)] 
        y = [ 1 for i in range(num_edge_2b)] 
        z = [ 0 for i in range(num_edge_2b)] 

        node = np.asarray([x, y, z])

        self.edge_2b['node'] = node
        self.edge_2b['sum_arclength'], self.edge_2b['arclength_ratio'] = self.cal_arclength(self.edge_2b)  

        # =======================================================
        self.edge_2c = {}

        num_edge_2c  = 51
        theta = [ 0.01*(51-(i+1))*np.pi for i in range(num_edge_2c)]
        x = [ 2+2*np.cos(theta[i]) for i in range(num_edge_2c)] 
        y = [ np.sin(theta[i]) for i in range(num_edge_2c)] 
        z = [ 0 for i in range(num_edge_2c)] 

        node = np.asarray([x, y, z])

        self.edge_2c['node'] = node
        self.edge_2c['sum_arclength'], self.edge_2c['arclength_ratio'] = self.cal_arclength(self.edge_2c)  

        # =======================================================
        self.edge_2d = {}

        num_edge_2d  = 51
        x = [ 4+0.1*((i+1)-1) for i in range(num_edge_2d)] 
        y = [ 0 for i in range(num_edge_2d)] 
        z = [ 0 for i in range(num_edge_2d)] 

        node = np.asarray([x, y, z])

        self.edge_2d['node'] = node
        self.edge_2d['sum_arclength'], self.edge_2d['arclength_ratio'] = self.cal_arclength(self.edge_2d)  

        # =======================================================
        self.edge_3a = {}

        num_edge_3a  = 21
        x = [ 9.0 for i in range(num_edge_3a)]
        y = [ 0.2*((i+1)-1) for i in range(num_edge_3a)]
        z = [ 0 for i in range(num_edge_3a)]

        node = np.asarray([x, y, z])

        self.edge_3a['node'] = node
        self.edge_3a['sum_arclength'], self.edge_3a['arclength_ratio'] = self.cal_arclength(self.edge_3a)  

        # =======================================================
        
        self.edge_4a = {}

        num_edge_4a  = 21
        x = [ -7 for i in range(num_edge_4a)]
        y = [ 0.2*((i+1)-1) for i in range(num_edge_4a)]
        z = [ 0 for i in range(num_edge_4a)]

        node = np.asarray([x, y, z])
        self.edge_4a['node'] = node
        self.edge_4a['sum_arclength'], self.edge_4a['arclength_ratio'] = self.cal_arclength(self.edge_4a)  

        # =======================================================
        self.edge_4b = {}

        num_edge_4b  = 21
        x = [ -7+0.2*((i+1)-1) for i in range(num_edge_4b)] 
        y = [ 4.0 for i in range(num_edge_4b)] 
        z = [ 0 for i in range(num_edge_4b)] 

        node = np.asarray([x, y, z])
        self.edge_4b['node'] = node
        self.edge_4b['sum_arclength'], self.edge_4b['arclength_ratio'] = self.cal_arclength(self.edge_4b)  
        # =======================================================
        self.edge_4c = {}

        num_edge_4c  = 26
        x = [ -3+0.2*((i+1)-1) for i in range(num_edge_4c)] 
        y = [ 4.0 for i in range(num_edge_4c)] 
        z = [ 0 for i in range(num_edge_4c)] 

        node = np.asarray([x, y, z])
        self.edge_4c['node'] = node
        self.edge_4c['sum_arclength'], self.edge_4c['arclength_ratio']  = self.cal_arclength(self.edge_4c)
        # =======================================================
        self.edge_4d = {}

        num_edge_4d  = 16
        x = [ 2+0.2*((i+1)-1) for i in range(num_edge_4d)] 
        y = [ 4.0 for i in range(num_edge_4d)] 
        z = [ 0 for i in range(num_edge_4d)] 

        node = np.asarray([x, y, z])
        self.edge_4d['node'] = node
        self.edge_4d['sum_arclength'], self.edge_4d['arclength_ratio'] = self.cal_arclength(self.edge_4d)
        # =======================================================
        self.edge_4e = {}

        num_edge_4e  = 21
        x = [ 5+0.2*((i+1)-1) for i in range(num_edge_4e)] 
        y = [ 4.0 for i in range(num_edge_4e)] 
        z = [ 0 for i in range(num_edge_4e)] 

        node = np.asarray([x, y, z])
        self.edge_4e['node'] = node
        self.edge_4e['sum_arclength'], self.edge_4e['arclength_ratio'] = self.cal_arclength(self.edge_4e)

    def cal_arclength(self, edge):

        num_nodes = len(edge['node'][0, :])
        
        arclength_ratio = np.zeros(num_nodes)
        sum_arclength = 0

        for i in range(num_nodes):
            if i == 0:
                sum_arclength = 0
                arclength_ratio[i] = sum_arclength
            
            else:    
                sum_arclength += np.linalg.norm(edge['node'][:, i] - edge['node'][:, i-1] )
                arclength_ratio[i] = sum_arclength

        arclength_ratio = arclength_ratio/sum_arclength

        return sum_arclength, arclength_ratio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

def Vinokur_Distribution(edge, d_s1, d_s2, num_node):

    xi_min, xi_max  = 1, num_node
   
    # create vector for xi
    xi_list = np.linspace(xi_min, xi_max, num_node, endpoint=True)

    # calculate arclength from database
    edge['sum_arclength'], edge['arclength_ratio'] = cal_arclength(edge)

    s_max = edge['sum_arclength']

    # calculate uniform spacing between xi_min and xi_max
    d_s_str = s_max / (xi_max-xi_min)

    # calculate constants 
    s0 =  d_s_str / d_s1
    s1 =  d_s_str / d_s2

    A = np.sqrt(s0/s1)
    B = np.sqrt(s0*s1)

    d_z = root(func_sin, 1,  args=(B,) ).x[0]

    xi_bar = ( xi_list-xi_min ) / ( xi_max-xi_min )

    if B < 1:
        u = 0.5 + np.tan(d_z*(xi_bar-0.5)) / (2*np.tan(d_z*0.5))
    elif B > 1:
        u = 0.5 + np.tanh(d_z*(xi_bar-0.5)) / (2*np.tanh(d_z*0.5))
    

    s = u / (A + (1-A)*u)

    position = np.zeros((3, num_node))
    
    for i, xi in  enumerate(xi_list):
        position[:,i] = interpolate_position(edge, s[i])

    return position


def interpolate_position(edge, s):

    for i in range(len(edge['node'][0, :])):

        if s == edge['arclength_ratio'][i]:
            pos = edge['node'][:,i]
            break
        elif edge['arclength_ratio'][i] < s < edge['arclength_ratio'][i+1]:
            pos = ( (s - edge['arclength_ratio'][i])/ (edge['arclength_ratio'][i+1]-edge['arclength_ratio'][i])
                         *(edge['node'][:,i+1]-edge['node'][:,i]) + edge['node'][:,i] )
            break

    return pos


def func_sin(x, c):
    if c < 1:
        return np.sin(x) / x - c
    elif c > 1:
        return np.sinh(x) / x - c



def TFI(xi_edge_min, xi_edge_max, eta_edge_min, eta_edge_max, soni=True):

    num_xi = len(eta_edge_min['node'][0,:])
    num_eta = len(xi_edge_min['node'][0,:])

    pos_vec = np.zeros((3, num_xi, num_eta))

    for index_xi in range(num_xi):
        for index_eta in range(num_eta):

            if soni == False:
                eta_bar = index_eta / (num_eta -1)
            
            elif soni == True:
                P_coeff = 1 - (eta_edge_max['arclength_ratio'][index_xi]-eta_edge_min['arclength_ratio'][index_xi])*(xi_edge_max['arclength_ratio'][index_eta]- xi_edge_min['arclength_ratio'][index_eta])
                eta_bar = xi_edge_min['arclength_ratio'][index_eta] + eta_edge_min['arclength_ratio'][index_xi]*(xi_edge_max['arclength_ratio'][index_eta]-
                                                                                                             xi_edge_min['arclength_ratio'][index_eta])/P_coeff

            pos_vec[:,index_xi, index_eta] = (1-eta_bar)*eta_edge_min['node'][:, index_xi] + eta_bar*eta_edge_max['node'][:, index_xi]
  

    err_xi_min = np.zeros((3,num_eta))
    err_xi_max = np.zeros((3,num_eta))
    for index_eta in range(num_eta):
        err_xi_min[:,index_eta] = xi_edge_min['node'][:, index_eta] - pos_vec[:,0, index_eta]
        err_xi_max[:,index_eta] = xi_edge_max['node'][:, index_eta] - pos_vec[:,num_xi -1, index_eta]


    for index_eta in range(num_eta):
        for index_xi in range(num_xi):
            xi_bar = index_xi / (num_xi -1)
            pos_vec[:,index_xi, index_eta] += (1-xi_bar)*err_xi_min[:,index_eta] + xi_bar*err_xi_max[:,index_eta]

    return np.expand_dims(pos_vec[0,:,:], axis=-1), np.expand_dims(pos_vec[1,:,:], axis=-1), np.expand_dims(pos_vec[2,:,:], axis=-1)



def cal_arclength(edge):

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


        

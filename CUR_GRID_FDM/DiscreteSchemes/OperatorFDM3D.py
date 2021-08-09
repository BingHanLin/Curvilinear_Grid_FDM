# http://www.legi.grenoble-inp.fr/people/Pierre.Augier/how-to-finite-differences-with-python.html
# http://olsthoorn.readthedocs.io/en/latest/03_fdm_as_python_func.html
from CUR_GRID_FDM.Geometry.BaseMesh import BaseMesh, NODELOC

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
import scipy.sparse as sp


class OperatorFDM3D:
    '''
    Create coefficient matrix for finite difference method in rectangular domain.    
    '''

    def __init__(self, Mesh: BaseMesh):
        self._mesh = Mesh
        self._axes = ('i', 'j', 'k')
        self._node_num = dict(zip(self._axes, self._mesh.mesh_size()))
        self._matrix_len = self._node_num['i'] * \
            self._node_num['j']*self._node_num['k']

    def no_operation(self):
        '''
        Return a coefficient matrix for no operation.
        '''
        return sp.csr_matrix(sp.eye(self._matrix_len))

# 權重可參考findiff
# https://github.com/maroba/findiff/tree/master/findiff
# https://blog.csdn.net/bitcarmanlee/article/details/52668477
# https://math.stackexchange.com/questions/1053751/3d-finite-difference-matrix
# http://olsthoorn.readthedocs.io/en/latest/03_fdm_as_python_func.html

    def der_1(self, axis='i'):
        '''
        Return a coefficient matrix for 1st derivative discretization with 2nd order accuracy.
        '''

        Ii, Ij, Ik = sp.eye(self._node_num['i']), sp.eye(
            self._node_num['j']), sp.eye(self._node_num['k'])

        di_1D = sp.diags([-0.5, 0.5], [-1, 1],
                         shape=(self._node_num['i'], self._node_num['i']))
        dj_1D = sp.diags([-0.5, 0.5], [-1, 1],
                         shape=(self._node_num['j'], self._node_num['j']))
        dk_1D = sp.diags([-0.5, 0.5], [-1, 1],
                         shape=(self._node_num['k'], self._node_num['k']))

        if axis == 'i':
            gap = 1
        elif axis == 'j':
            gap = self._node_num['i']
        elif axis == 'k':
            gap = self._node_num['i']*self._node_num['j']

        front_bound_matrix = sp.csr_matrix(sp.diags([-1.5, 2, -0.5], [0, gap, 2*gap],
                                                    shape=(self._matrix_len, self._matrix_len)))

        end_bound_matrix = sp.csr_matrix(sp.diags([0.5, -2, 1.5], [-2*gap, -gap, 0],
                                                  shape=(self._matrix_len, self._matrix_len)))

        if axis == 'i':
            der_1_3D = sp.csr_matrix(sp.kron(Ik, sp.kron(Ij, di_1D)))
            self.csr_zero_rows(der_1_3D, self._mesh.get_node_index_list(
                NODELOC.I_START | NODELOC.I_END))
            self.csr_zero_rows(front_bound_matrix,  self._mesh.get_node_index_list(
                NODELOC.I_START, True))
            self.csr_zero_rows(end_bound_matrix, self._mesh.get_node_index_list(
                NODELOC.I_END, True))

        elif axis == 'j':
            der_1_3D = sp.csr_matrix(sp.kron(Ik, sp.kron(dj_1D, Ii)))
            self.csr_zero_rows(der_1_3D, self._mesh.get_node_index_list(
                NODELOC.J_START | NODELOC.J_END))
            self.csr_zero_rows(front_bound_matrix,  self._mesh.get_node_index_list(
                NODELOC.J_START, True))
            self.csr_zero_rows(end_bound_matrix, self._mesh.get_node_index_list(
                NODELOC.J_END, True))

        elif axis == 'k':
            der_1_3D = sp.csr_matrix(sp.kron(dk_1D, sp.kron(Ii, Ij)))
            self.csr_zero_rows(der_1_3D, self._mesh.get_node_index_list(
                NODELOC.K_START | NODELOC.K_END))
            self.csr_zero_rows(front_bound_matrix,  self._mesh.get_node_index_list(
                NODELOC.K_START, True))
            self.csr_zero_rows(end_bound_matrix, self._mesh.get_node_index_list(
                NODELOC.K_END, True))

        return der_1_3D + front_bound_matrix + end_bound_matrix

    def der_2(self, axis='i'):
        '''
        Return a coefficient matrix for 2nd derivative discretization with 2nd order accuracy.
        '''

        Ii, Ij, Ik = sp.eye(self._node_num['i']), sp.eye(
            self._node_num['j']), sp.eye(self._node_num['k'])

        dii_1D = sp.diags([1, -2, 1], [-1, 0, 1],
                          shape=(self._node_num['i'], self._node_num['i']))
        djj_1D = sp.diags([1, -2, 1], [-1, 0, 1],
                          shape=(self._node_num['j'], self._node_num['j']))
        dkk_1D = sp.diags([1, -2, 1], [-1, 0, 1],
                          shape=(self._node_num['k'], self._node_num['k']))

        if axis == 'i':
            gap = 1
        elif axis == 'j':
            gap = self._node_num['i']
        elif axis == 'k':
            gap = self._node_num['i']*self._node_num['j']

        front_bound_matrix = sp.csr_matrix(sp.diags([2, -5, 4, -1], [0, gap, 2*gap, 3*gap],
                                                    shape=(self._matrix_len, self._matrix_len)))

        end_bound_matrix = sp.csr_matrix(sp.diags([-1, 4, -5, 2], [-3*gap, -2*gap, -gap, 0],
                                                  shape=(self._matrix_len, self._matrix_len)))

        if axis == 'i':
            der_2_3D = sp.csr_matrix(sp.kron(Ik, sp.kron(Ij, dii_1D)))
            self.csr_zero_rows(der_2_3D, self._mesh.get_node_index_list(
                NODELOC.I_START | NODELOC.I_END))
            self.csr_zero_rows(front_bound_matrix,  self._mesh.get_node_index_list(
                NODELOC.I_START, True))
            self.csr_zero_rows(end_bound_matrix, self._mesh.get_node_index_list(
                NODELOC.I_END, True))

        elif axis == 'j':
            der_2_3D = sp.csr_matrix(sp.kron(Ik, sp.kron(djj_1D, Ii)))
            self.csr_zero_rows(der_2_3D, self._mesh.get_node_index_list(
                NODELOC.J_START | NODELOC.J_END))
            self.csr_zero_rows(front_bound_matrix,  self._mesh.get_node_index_list(
                NODELOC.J_START, True))
            self.csr_zero_rows(end_bound_matrix, self._mesh.get_node_index_list(
                NODELOC.J_END, True))

        elif axis == 'k':
            der_2_3D = sp.csr_matrix(sp.kron(dkk_1D, sp.kron(Ii, Ij)))
            self.csr_zero_rows(der_2_3D, self._mesh.get_node_index_list(
                NODELOC.K_START | NODELOC.K_END))
            self.csr_zero_rows(front_bound_matrix,  self._mesh.get_node_index_list(
                NODELOC.K_START, True))
            self.csr_zero_rows(end_bound_matrix, self._mesh.get_node_index_list(
                NODELOC.K_END, True))

        return der_2_3D + front_bound_matrix + end_bound_matrix

    def der_11(self, axis='ij'):
        '''
        Return a coefficient matrix for double derivative discretization with 2nd order accuracy.
        '''

        Ii, Ij, Ik = sp.eye(self._node_num['i']), sp.eye(
            self._node_num['j']), sp.eye(self._node_num['k'])

        di_1D = sp.diags([-0.5, 0.5], [-1, 1],
                         shape=(self._node_num['i'], self._node_num['i']))
        dj_1D = sp.diags([-0.5, 0.5], [-1, 1],
                         shape=(self._node_num['j'], self._node_num['j']))
        dk_1D = sp.diags([-0.5, 0.5], [-1, 1],
                         shape=(self._node_num['k'], self._node_num['k']))

        if axis == 'ij' or axis == 'ji':
            der_11_3D = sp.kron(Ik, sp.kron(dj_1D, di_1D))

        elif axis == 'ik' or axis == 'ki':
            der_11_3D = sp.kron(dk_1D, sp.kron(Ij, di_1D))

        elif axis == 'jk' or axis == 'kj':
            der_11_3D = sp.kron(dk_1D, sp.kron(dj_1D, Ii))

        return der_11_3D

    # https://stackoverflow.com/questions/19784868/what-is-most-efficient-way-of-setting-row-to-zeros-for-a-sparse-scipy-matrix
    def csr_zero_rows(self, csr, rows_to_zero):
        rows, cols = csr.shape
        mask = np.ones((rows,), dtype=np.bool)
        mask[rows_to_zero] = False
        nnz_per_row = np.diff(csr.indptr)

        mask = np.repeat(mask, nnz_per_row)
        nnz_per_row[rows_to_zero] = 0
        csr.data = csr.data[mask]
        csr.indices = csr.indices[mask]
        csr.indptr[1:] = np.cumsum(nnz_per_row)

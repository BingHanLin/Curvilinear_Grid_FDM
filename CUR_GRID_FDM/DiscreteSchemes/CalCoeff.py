from CUR_GRID_FDM.Geometry.BaseMesh import BaseMesh
import numpy as np


class CalCoeff:
    '''
    Calculate essential coefficients for curvilinear coordinates
    '''

    def __init__(self, Mesh: BaseMesh):
        self._mesh = Mesh
        self.update_Coeff()

    def update_Coeff(self):
        self._cal_co_basis_vector()
        self._cal_jacobian()
        self._cal_con_basis_vector()
        self._cal_metric_tensor()
        self._cal_christoffel_symbol()

    def _cal_co_basis_vector(self):
        # create covariant basis vectors
        basis_size = (3,) + self._mesh.mesh_size() + (3,)
        self._co_basis = np.zeros(basis_size)

        self._co_basis[0, ..., 0], self._co_basis[1, ..., 0], self._co_basis[2, ..., 0] = \
            np.gradient(self._mesh.x(), axis=(0, 1, 2))

        self._co_basis[0, ..., 1], self._co_basis[1, ..., 1], self._co_basis[2, ..., 1] = \
            np.gradient(self._mesh.y(), axis=(0, 1, 2))

        self._co_basis[0, ..., 2], self._co_basis[1, ..., 2], self._co_basis[2, ..., 2] = \
            np.gradient(self._mesh.z(), axis=(0, 1, 2))

    def _cal_jacobian(self):
        # calculate jacobian of each basis
        crossVec = np.cross(self._co_basis[0], self._co_basis[1])
        self._jacobian = np.einsum(
            'ijkl,ijkl->ijk', crossVec, self._co_basis[2])

    def _cal_con_basis_vector(self):
        # create contravariant basis vectors
        basis_size = (3,) + self._mesh.mesh_size() + (3,)
        self._con_basis = np.zeros(basis_size)

        self._con_basis[0] = np.cross(
            self._co_basis[1], self._co_basis[2]) / self._jacobian[..., np.newaxis]

        self._con_basis[1] = np.cross(
            self._co_basis[2], self._co_basis[0]) / self._jacobian[..., np.newaxis]

        self._con_basis[2] = np.cross(
            self._co_basis[0], self._co_basis[1]) / self._jacobian[..., np.newaxis]

    def _cal_metric_tensor(self):

        # https://zhuanlan.zhihu.com/p/27739282 如何理解和使用NumPy.einsum

        # create metric_tensor g_ij
        self._metric_tensor = np.zeros(self._mesh.mesh_size() + (3, 3))
        for i in range(3):
            for j in range(3):
                self._metric_tensor[..., i, j] = np.einsum(
                    'ijkl,ijkl->ijk', self._co_basis[i], self._co_basis[j])

        # create inv_metric_tensor g^ij
        self._inv_metric_tensor = np.zeros(self._mesh.mesh_size() + (3, 3))
        for i in range(3):
            for j in range(3):
                self._inv_metric_tensor[..., i, j] = np.einsum(
                    'ijkl,ijkl->ijk', self._con_basis[i], self._con_basis[j])

    def _cal_christoffel_symbol(self):
        # https://johnkerl.org/gdg/gdgprojnotes.pdf
        # create christoffel symbols
        self._christoffel_symbol = np.zeros(
            (self._mesh.mesh_size() + (3, 3, 3)))

        # # get gradient along i, j, k direction
        d_metric_tensor = np.zeros((self._metric_tensor.shape + (3,)))
        for i in range(3):
            for j in range(3):
                d_metric_tensor[..., i, j, 0], d_metric_tensor[..., i, j, 1], d_metric_tensor[..., i, j, 2] = \
                    np.gradient(self._metric_tensor[..., i, j], axis=(0, 1, 2))

        # calculate christoffel symbols
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for m in range(3):
                        self._christoffel_symbol[..., i, j, k] = \
                            self._christoffel_symbol[..., i, j, k] + \
                            0.5*self._inv_metric_tensor[..., i, m]*(
                                d_metric_tensor[..., m, j, k] +
                                d_metric_tensor[..., k, m, j] -
                                d_metric_tensor[..., j, k, m])

    def get_co_basis(self, idx):
        return np.reshape(self._co_basis[idx, ..., :], (self._mesh.node_number(), 3), order='F')

    def get_con_basis(self, idx):
        return np.reshape(self._con_basis[idx, ..., :], (self._mesh.node_number(), 3), order='F')

    def get_metric_tensor(self, idx1, idx2):
        return np.reshape(self._metric_tensor[..., idx1, idx2], self._mesh.node_number(), order='F')

    def get_inv_metric_tensor(self, idx1, idx2):
        return np.reshape(self._inv_metric_tensor[..., idx1, idx2], self._mesh.node_number(), order='F')

    def get_christoffel_symbol(self, idx1, idx2, idx3):
        return np.reshape(self._christoffel_symbol[..., idx1, idx2, idx3], self._mesh.node_number(), order='F')

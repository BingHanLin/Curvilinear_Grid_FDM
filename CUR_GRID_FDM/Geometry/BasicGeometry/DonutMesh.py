from ..BaseMesh import BaseMesh
import numpy as np


class DonutMesh(BaseMesh):

    def __init__(self, r_inner: float, r_outer: float, n_theta: int, n_radius: int, lz: int, nz: int):

        self._r_inner = r_inner
        self._r_outer = r_outer

        self._n_theta = n_theta
        self._n_radius = n_radius

        self._lz = lz
        self._nz = nz

        self._create_grid()

    def _create_grid(self):

        self._node_number = self._n_theta * self._n_radius * self._nz
        self._mesh_size = (self._n_theta, self._n_radius, self._nz)

        theta = np.linspace(0, 2 * np.pi - 2 * np.pi /
                            self._n_theta, self._n_theta)
        radius = np.linspace(self._r_inner, self._r_outer, self._n_radius)

        z = np.linspace(0, self._lz, self._nz)

        theta_matrix, radius_matrix = np.meshgrid(theta, radius)

        _, _, self._z = np.meshgrid(theta, radius, z, indexing='ij')

        self._x = radius_matrix * np.cos(theta_matrix)
        self._y = radius_matrix * np.sin(theta_matrix)

        self._x = np.dstack([self._x]*self._nz)
        self._y = np.dstack([self._y]*self._nz)
        self._x = self._x.transpose((1, 0, 2))
        self._y = self._y.transpose((1, 0, 2))

        self._x_flatten = np.reshape(self._x, self._node_number, order='F')
        self._y_flatten = np.reshape(self._y, self._node_number, order='F')
        self._z_flatten = np.reshape(self._z, self._node_number, order='F')

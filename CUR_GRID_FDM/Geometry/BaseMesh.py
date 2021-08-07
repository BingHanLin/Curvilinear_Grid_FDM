import abc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class BaseMesh(abc.ABC):

    @abc.abstractmethod
    def _create_grid(self):
        pass

    def _cal_out_norm(self):
        co_basis = np.zeros((3,) + self.mesh_size() + (3,))

        co_basis[0, ..., 0], co_basis[1, ..., 0], co_basis[2, ..., 0] = \
            np.gradient(self.x(), axis=(0, 1, 2))

        co_basis[0, ..., 1], co_basis[1, ..., 1], co_basis[2, ..., 1] = \
            np.gradient(self.y(), axis=(0, 1, 2))

        co_basis[0, ..., 2], co_basis[1, ..., 2], co_basis[2, ..., 2] = \
            np.gradient(self.z(), axis=(0, 1, 2))

        self._out_norm = np.zeros((self.mesh_size() + (3,)))

        out_norm = np.cross(
            co_basis[1, 0, :, :, :], co_basis[2, 0, :, :, :])
        self._out_norm[0, :, :, :] = -out_norm / \
            np.linalg.norm(out_norm, axis=-1)[..., None]

        out_norm = np.cross(
            co_basis[1, -1, :, :, :], co_basis[2, -1, :, :, :])
        self._out_norm[-1, :, :, :] = out_norm / \
            np.linalg.norm(out_norm, axis=-1)[..., None]

        out_norm = np.cross(
            co_basis[2, :, 0, :, :], co_basis[0, :, 0, :, :])
        self._out_norm[:, 0, :, :] = - out_norm / \
            np.linalg.norm(out_norm, axis=-1)[..., None]

        out_norm = np.cross(
            co_basis[2, :, -1, :, :], co_basis[0, :, -1, :, :])
        self._out_norm[:, -1, :, :] = out_norm / \
            np.linalg.norm(out_norm, axis=-1)[..., None]

        out_norm = np.cross(
            co_basis[0, :, :, 0, :], co_basis[1, :, :, 0, :])
        self._out_norm[:, :, 0, :] = - out_norm / \
            np.linalg.norm(out_norm, axis=-1)[..., None]

        out_norm = np.cross(
            co_basis[0, :, :, -1, :], co_basis[1, :, :, -1, :])
        self._out_norm[:, :, -1, :] = out_norm / \
            np.linalg.norm(out_norm, axis=-1)[..., None]

    def mesh_size(self):
        return self._mesh_size

    def x(self):
        return self._x

    def y(self):
        return self._y

    def z(self):
        return self._z

    def x_flatten(self):
        return np.reshape(self._x, self._node_number, order='F')

    def y_flatten(self):
        return np.reshape(self._y, self._node_number, order='F')

    def z_flatten(self):
        return np.reshape(self._z, self._node_number, order='F')

    def node_number(self):
        return self._node_number

    def out_norm(self):
        if not hasattr(self, '_out_norm'):
            self._cal_out_norm()
        return self._out_norm

    def get_node_index_list(self, i_front=False, i_end=False, j_front=False, j_end=False, k_front=False, k_end=False, interior=False, inverse=False):
        '''
        i_front, i_end, j_front, j_end, k_front, k_end, interior, if not define then return all
        '''

        index_list = np.array([], dtype=int)
        node_num_list = np.zeros(self._mesh_size)

        if i_front == True:
            node_num_list[0, :, :] = 1

        if i_end == True:
            node_num_list[-1, :, :] = 1

        if j_front == True:
            node_num_list[:, 0, :] = 1

        if j_end == True:
            node_num_list[:, -1, :] = 1

        if k_front == True:
            node_num_list[:, :, 0] = 1

        if k_end == True:
            node_num_list[:, :, -1] = 1

        node_num_list = np.reshape(node_num_list, self._node_number, order='F')
        i_bool = np.where(node_num_list == 1)
        index_list = np.append(index_list, i_bool)

        if inverse == True:
            return np.setdiff1d(np.arange(self._node_number), index_list)
        else:
            return index_list

    def plot_grid(self, BCtype=False):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        marker_size = 50

        if BCtype is False:
            ax.scatter(self.x_flatten(), self._y_flatten, self._z_flatten,
                       marker='o', c='g', s=marker_size, label='nodes')

        else:
            for nodetype in set(BCtype):

                mask = np.ma.masked_where(BCtype != nodetype, BCtype)

                ax.scatter(np.ma.masked_array(self.x_flatten(), mask.mask),
                           np.ma.masked_array(self.y_flatten(), mask.mask),
                           np.ma.masked_array(self.z_flatten(), mask.mask),
                           marker='o', s=marker_size, label='nodetype: {}'.format(int(nodetype)))

        plt.legend()
        plt.axis('auto')
        plt.show()

    def plot_out_norm(self):

        u = np.reshape(self.out_norm(),
                       (self.x_flatten().shape + (3,)), order='F')[:, 0]

        v = np.reshape(self.out_norm(),
                       (self.y_flatten().shape + (3,)), order='F')[:, 1]

        w = np.reshape(self.out_norm(),
                       (self.z_flatten().shape + (3,)), order='F')[:, 2]

        ax = plt.figure().add_subplot(projection='3d')

        ax.quiver(self.x_flatten(), self.y_flatten(),
                  self.z_flatten(), u, v, w,  normalize=True)

        plt.show()

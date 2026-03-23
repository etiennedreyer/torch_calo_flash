import numpy as np
import calo_flash

class CaloBlock:

    ### Homogeneous calorimeter block with cells on Cartesian grid

    def __init__(self, config):
        self.Z = config['Z']
        self.height = config['height']
        self.width = config['width']
        self.depth = config['depth']
        self.N_cells_transverse = config['N_cells_transverse']
        self.N_cells_longitudinal = config['N_cells_longitudinal']

        self.initialize_cells()

    def initialize_cells(self):
        self.cell_size_x = self.width / self.N_cells_transverse
        self.cell_size_y = self.height / self.N_cells_transverse
        self.cell_size_z = self.depth / self.N_cells_longitudinal

        self.cell_x = np.linspace(-self.width/2, self.width/2, self.N_cells_transverse)
        self.cell_y = np.linspace(-self.height/2, self.height/2, self.N_cells_transverse)
        self.cell_z = np.linspace(0, self.depth, self.N_cells_longitudinal)
        self.cell_e = np.zeros((self.N_cells_transverse, self.N_cells_transverse, self.N_cells_longitudinal))

    def shower(self, E):

        spots = calo_flash.shoot(E, self.Z, self.cell_z)

        ### Convert to Cartesian coordinates
        x = spots['r'] * np.cos(spots['phi'])
        y = spots['r'] * np.sin(spots['phi'])
        z = spots['t']
        e = spots['E']

        ### Lookup cells
        idx_x = np.searchsorted(self.cell_x, x) - 1
        idx_y = np.searchsorted(self.cell_y, y) - 1
        idx_z = np.searchsorted(self.cell_z, z) - 1

        ### Mask for valid indices
        mask = (idx_x >= 0) & (idx_x < self.N_cells_transverse) & \
            (idx_y >= 0) & (idx_y < self.N_cells_transverse) & \
            (idx_z >= 0) & (idx_z < self.N_cells_longitudinal)

        ### Deposit energy in cells
        np.add.at(self.cell_e[:, :, :], 
                  (idx_x[mask], idx_y[mask], idx_z[mask]), e[mask])


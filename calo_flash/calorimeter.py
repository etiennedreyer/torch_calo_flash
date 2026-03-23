import numpy as np
import calo_flash

class CaloBlock:

    ### Homogeneous calorimeter block with cells on Cartesian grid

    def __init__(self, config):
        self.Z = config['Z']
        self.width = config['width']
        self.height = config.get('height', self.width)
        self.depth = config['depth']
        self.N_cells_x = config['N_cells_x']
        self.N_cells_y = config.get('N_cells_y', self.N_cells_x)
        self.N_cells_z = config['N_cells_z']
        self.N_cells = self.N_cells_x * self.N_cells_y * self.N_cells_z

        self.initialize_cells()

    def initialize_cells(self):
        self.cell_size_x = self.width / self.N_cells_x
        self.cell_size_y = self.height / self.N_cells_y
        self.cell_size_z = self.depth / self.N_cells_z

        self.cell_x = np.linspace(-self.width/2, self.width/2, self.N_cells_x)
        self.cell_y = np.linspace(-self.height/2, self.height/2, self.N_cells_y)
        self.cell_z = np.linspace(0, self.depth, self.N_cells_z)

        # Bin edges used for cell lookup (N+1 edges covering the full range)
        self.cell_x_edges = np.linspace(-self.width/2, self.width/2, self.N_cells_x + 1)
        self.cell_y_edges = np.linspace(-self.height/2, self.height/2, self.N_cells_y + 1)
        self.cell_z_edges = np.linspace(0, self.depth, self.N_cells_z + 1)
        self.cell_e = np.zeros((self.N_cells_x, self.N_cells_y, self.N_cells_z))
        self.cell_local_indices = np.indices(self.cell_e.shape).reshape(3, -1).T
        self.cell_global_indices = np.arange(self.N_cells).reshape(self.cell_e.shape)

        ### For storing flattened truth record
        self.cell_particle_src = []
        self.cell_particle_dst = []
        self.cell_particle_wgt = []
    
    def clear(self):
        self.cell_e.fill(0)
        self.cell_particle_src.clear()
        self.cell_particle_dst.clear()
        self.cell_particle_wgt.clear()

    def simulate(self, particle_Es, particle_xs=None, particle_ys=None,
                       N_spots_per_layer=None, store_truth=True):

        ### Flexible input handling
        if isinstance(particle_Es, float) or isinstance(particle_Es, int):
            particle_Es = np.array([particle_Es])

        N = len(particle_Es)
        if particle_xs is None: particle_xs = np.zeros(N)
        elif isinstance(particle_xs, float) or isinstance(particle_xs, int):
            particle_xs = np.array([particle_xs])
        if particle_ys is None: particle_ys = np.zeros(N)
        elif isinstance(particle_ys, float) or isinstance(particle_ys, int):
            particle_ys = np.array([particle_ys])

        ### Empty calo and truth
        self.clear()

        ### Calo Flash simulation
        spots = calo_flash.shoot(particle_Es, self.Z, self.cell_z_edges,
                                    N_spots_per_layer=N_spots_per_layer)

        ### Convert to Cartesian
        pidx = spots['particle_idx']
        x = spots['r'] * np.cos(spots['phi']) + particle_xs[pidx]
        y = spots['r'] * np.sin(spots['phi']) + particle_ys[pidx]
        z = spots['t']
        e = spots['E']

        ### Discard out-of-bounds spots
        mask = (x >= -self.width/2)  & (x < self.width/2) & \
               (y >= -self.height/2) & (y < self.height/2) & \
               (z >= 0)              & (z < self.depth)
        x_m, y_m, z_m, e_m, pidx_m = x[mask], y[mask], z[mask], e[mask], pidx[mask]

        ### Cell indices via floor-division (uniform bins — avoids searchsorted / histogramdd)
        ix = np.floor((x_m - self.cell_x_edges[0]) / self.cell_size_x).astype(np.intp)
        iy = np.floor((y_m - self.cell_y_edges[0]) / self.cell_size_y).astype(np.intp)
        iz = np.floor((z_m - self.cell_z_edges[0]) / self.cell_size_z).astype(np.intp)
        cell_key = ix * (self.N_cells_y * self.N_cells_z) + iy * self.N_cells_z + iz

        ### Deposit energy
        self.cell_e = np.bincount(cell_key, weights=e_m, minlength=self.N_cells).reshape(
            self.N_cells_x, self.N_cells_y, self.N_cells_z)

        if not store_truth:
            return

        ### Truth record
        combined_key = cell_key * N + pidx_m  # unique key for (cell, particle) pair
        wgt_dense = np.bincount(combined_key, weights=e_m, minlength=self.N_cells * N)
        nonzero = np.flatnonzero(wgt_dense)

        self.cell_particle_src = (nonzero // N).tolist()
        self.cell_particle_dst = (nonzero  % N).tolist()
        self.cell_particle_wgt = wgt_dense[nonzero].tolist()

class EventGenerator:

    def __init__(self, config):
        self.x_min, self.x_max = config['x_range']
        self.y_min, self.y_max = config['y_range']
        self.E_min, self.E_max = config['E_range']
        self.N_min, self.N_max = config['N_range']

    def generate(self):

        N = np.random.randint(self.N_min, self.N_max + 1)

        ### Uniform spread in x/y
        particle_xs = np.random.uniform(self.x_min, self.x_max, N)
        particle_ys = np.random.uniform(self.y_min, self.y_max, N)
        
        ### Power-law spectrum in energy
        alpha = 2.0
        r = np.random.uniform(0, 1, N)
        particle_Es = ((self.E_max**(1-alpha) - self.E_min**(1-alpha)) * r + self.E_min**(1-alpha))**(1/(1-alpha))

        return particle_Es, particle_xs, particle_ys
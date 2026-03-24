import torch
import numpy as np
from calo_flash import shoot

class CaloBlock:

    ### Homogeneous calorimeter block with cells on Cartesian grid

    def __init__(self, config, device=None):
        self.Z = config['Z']
        self.width = config['width']
        self.height = config.get('height', self.width)
        self.depth = config['depth']
        self.N_cells_x = config['N_cells_x']
        self.N_cells_y = config.get('N_cells_y', self.N_cells_x)
        self.N_cells_z = config['N_cells_z']
        self.N_cells = self.N_cells_x * self.N_cells_y * self.N_cells_z
        self.N_spots_per_layer = config.get('N_spots_per_layer', 1000)

        if device is None:
            self.get_device()

        self.initialize_cells()

    def get_device(self):
        if torch.cuda.is_available():
            print("Using CUDA")
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    def initialize_cells(self):
        self.cell_size_x = self.width / self.N_cells_x
        self.cell_size_y = self.height / self.N_cells_y
        self.cell_size_z = self.depth / self.N_cells_z

        # N+1 bin edges covering the full range
        self.cell_x_edges = torch.linspace(-self.width/2, self.width/2, self.N_cells_x + 1, device=self.device)
        self.cell_y_edges = torch.linspace(-self.height/2, self.height/2, self.N_cells_y + 1, device=self.device)
        self.cell_z_edges = torch.linspace(0, self.depth, self.N_cells_z + 1, device=self.device)
        self.cell_e = torch.zeros(self.N_cells_x, self.N_cells_y, self.N_cells_z, device=self.device)

        ### For storing flattened truth record
        self.cell_particle_src = []
        self.cell_particle_dst = []
        self.cell_particle_wgt = []
    
    def clear(self):
        self.cell_e.zero_()
        self.cell_particle_src.clear()
        self.cell_particle_dst.clear()
        self.cell_particle_wgt.clear()

    def simulate(self, particle_Es: torch.Tensor, particle_xs: torch.Tensor, particle_ys: torch.Tensor, 
                 store_truth=True, N_spots_per_layer=None):

        ### Shape
        N_particles = len(particle_Es)
        assert particle_Es.shape == particle_xs.shape == particle_ys.shape == (N_particles,), \
            "Input tensors must have shape (N_particles,)"

        ### Move to device
        particle_Es = particle_Es.to(device=self.device, dtype=torch.float32)
        particle_xs = particle_xs.to(device=self.device, dtype=torch.float32)
        particle_ys = particle_ys.to(device=self.device, dtype=torch.float32)

        ### Empty calo and truth
        self.clear()

        ### Calo Flash simulation
        if N_spots_per_layer is None:
            N_spots_per_layer = self.N_spots_per_layer
        spots = shoot(particle_Es, self.Z, self.cell_z_edges,
                        N_spots_per_layer=N_spots_per_layer)

        ### Convert to Cartesian
        pidx = spots['particle_idx'].long()
        x = spots['r'] * torch.cos(spots['phi']) + particle_xs[pidx]
        y = spots['r'] * torch.sin(spots['phi']) + particle_ys[pidx]
        z = spots['t']
        e = spots['E']

        ### Discard out-of-bounds spots
        mask = (x >= -self.width/2)  & (x < self.width/2) & \
               (y >= -self.height/2) & (y < self.height/2) & \
               (z >= 0)              & (z < self.depth)
        x_m, y_m, z_m, e_m, pidx_m = x[mask], y[mask], z[mask], e[mask], pidx[mask]

        ### Assign local index to each spot based on nearest cell edge
        ### (assumes uniform grid and positive x_m - cell_x_edges[0])
        ix = ((x_m - self.cell_x_edges[0]) / self.cell_size_x).long().clamp(0, self.N_cells_x - 1)
        iy = ((y_m - self.cell_y_edges[0]) / self.cell_size_y).long().clamp(0, self.N_cells_y - 1)
        iz = ((z_m - self.cell_z_edges[0]) / self.cell_size_z).long().clamp(0, self.N_cells_z - 1)

        ### Global cell index for flattened array
        cell_idx = ix * (self.N_cells_y * self.N_cells_z) + iy * self.N_cells_z + iz

        ### Deposit energy
        flat_e = torch.zeros(self.N_cells, device=self.device)
        flat_e.scatter_add_(0, cell_idx, e_m)
        self.cell_e = flat_e.reshape(self.N_cells_x, self.N_cells_y, self.N_cells_z)

        if not store_truth:
            ### Save some time
            return

        ### Truth record ###

        ### unique key for (cell, particle) pair
        combined_idx = cell_idx * N_particles + pidx_m

        ### Sum energy for each (cell, particle) pair
        wgt_dense = torch.zeros(self.N_cells * N_particles, device=self.device)
        wgt_dense.scatter_add_(0, combined_idx, e_m)
        nonzero = wgt_dense.nonzero(as_tuple=True)[0]

        ### Invert combined index
        self.cell_particle_src = (nonzero // N_particles).tolist()
        self.cell_particle_dst = (nonzero  % N_particles).tolist()
        self.cell_particle_wgt = wgt_dense[nonzero].tolist()

class EventGenerator:

    def __init__(self, config):
        self.x_min, self.x_max = config['x_range']
        self.y_min, self.y_max = config['y_range']
        self.E_min, self.E_max = config['E_range']
        self.N_min, self.N_max = config['N_range']

    def generate(self):

        N_particles = torch.randint(self.N_min, self.N_max + 1, (1,)).item()

        ### Uniform spread in x/y
        particle_xs = torch.rand(N_particles) * (self.x_max - self.x_min) + self.x_min
        particle_ys = torch.rand(N_particles) * (self.y_max - self.y_min) + self.y_min

        ### Power-law spectrum in energy
        alpha = 2.0
        r = torch.rand(N_particles)
        particle_Es = ((self.E_max**(1-alpha) - self.E_min**(1-alpha)) * r + self.E_min**(1-alpha))**(1/(1-alpha))

        return particle_Es, particle_xs, particle_ys
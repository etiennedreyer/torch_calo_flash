import torch

class EventGenerator:

    def __init__(self, config):
        self.x_min, self.x_max = config['x_range']
        self.y_min, self.y_max = config['y_range']
        self.E_min, self.E_max = config['E_range']
        self.N_min, self.N_max = config['N_range']
        self.power = config.get('power', 2.0)

    def generate(self, N_events=1):

        N_particles = torch.randint(self.N_min, self.N_max + 1, (N_events,)).tolist()

        N_pad = max(N_particles)

        ### Uniform spread in x/y
        particle_xs = torch.rand((N_events, N_pad)) * (self.x_max - self.x_min) + self.x_min
        particle_ys = torch.rand((N_events, N_pad)) * (self.y_max - self.y_min) + self.y_min

        ### Power-law spectrum in energy
        r = torch.rand((N_events, N_pad))
        particle_Es = ((self.E_max**(1-self.power) - self.E_min**(1-self.power)) * r + self.E_min**(1-self.power))**(1/(1-self.power))

        if N_events == 1:
            return particle_Es[0], particle_xs[0], particle_ys[0]

        ### Mask out padded particles, vectorized
        for i, N in enumerate(N_particles):
            particle_Es[i, N:] = 0.0  # zero energy means "no particle"

        return particle_Es, particle_xs, particle_ys
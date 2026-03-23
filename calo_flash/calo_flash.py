import numpy as np
import scipy
from scipy.stats import gamma as gamma_dist

### Python implementation of https://arxiv.org/pdf/hep-ex/0001020

atomic_number = {
    'lead': 82,
    'iron': 26,
    'copper': 29,
    'aluminum': 13
}

### From https://pdg.lbl.gov/2025/AtomicNuclearProperties/expert.html
critical_energy = {
    82: 7.43,
    26: 21.0,
    29: 19.0,
    13: 28.0
}

###########################
#### Fitted Parameters ####
###########################

### LONGITUDINAL ###

### Eq (2)
def longitudinal_pdf(t, alpha, beta):
    return gamma_dist.pdf(t, alpha, scale=1/beta)

def longitudinal_cdf(t, alpha, beta):
    return gamma_dist.cdf(t, alpha, scale=1/beta)

### Eq (4)
def get_beta(alpha, T):
    return (alpha - 1) / T

### Eq (7) -- default yields mean value (A.1.1)
def get_T(y, 
          t_1=0.858):

    return np.log(y) - t_1

### Eq (8) -- default yields mean value (A.1.1)
def get_alpha(y, Z, 
              a_1=0.21, 
              a_2=0.492, 
              a_3=2.38):
    return a_1 + (a_2 + a_3/Z) * np.log(y)

### Putting it all together (longitudinal)
def get_longitudinal_parameters(E, Z):

    y = E / critical_energy[Z]

    ### Mean longitudinal profile parameters (default args)
    mean_T = get_T(y)
    mean_alpha = get_alpha(y, Z)
    mean_beta = get_beta(mean_alpha, mean_T)

    ### Eq (9)
    def get_sigma(y, s1, s2):
        return 1 / (s1 + s2*np.log(y))

    ### Eq (10)
    def get_rho(y, r1, r2):
        return r1 + r2*np.log(y)

    ### A.1.2
    mean_ln_T         = np.log(get_T(y, t_1=0.812))
    sigma_ln_T        = get_sigma(y, s1=-1.4, s2=1.26)
    mean_ln_alpha     = np.log(get_alpha(y, Z, a_1=0.81, a_2=0.458, a_3=2.26))
    sigma_ln_alpha    = get_sigma(y, s1=-0.58, s2=0.86)
    rho_ln_T_ln_alpha = get_rho(y, r1=0.705, r2=-0.023)

    ### Two random variables
    if isinstance(y, np.ndarray):
        # z = np.random.randn(len(y), 2)
        z1 = np.random.randn(len(y))
        z2 = np.random.randn(len(y))
    else:
        # z = np.random.randn(2)
        z1, z2 = np.random.randn(2)

    ### Eq (11) expanded
    ln_T_i     = mean_ln_T     + sigma_ln_T     * (np.sqrt(1+rho_ln_T_ln_alpha)*z1 + np.sqrt(1-rho_ln_T_ln_alpha)*z2)/np.sqrt(2)
    ln_alpha_i = mean_ln_alpha + sigma_ln_alpha * (np.sqrt(1+rho_ln_T_ln_alpha)*z1 - np.sqrt(1-rho_ln_T_ln_alpha)*z2)/np.sqrt(2)

    ### Final parameters
    T_i = np.exp(ln_T_i)
    alpha_i = np.exp(ln_alpha_i)
    beta_i = get_beta(alpha_i, T_i)

    return {
        'mean_T': mean_T,
        'mean_alpha': mean_alpha,
        'mean_beta': mean_beta,
        'T': T_i,
        'alpha': alpha_i,
        'beta': beta_i,
        'mean_ln_T': mean_ln_T,
        'sigma_ln_T': sigma_ln_T,
        'mean_ln_alpha': mean_ln_alpha,
        'sigma_ln_alpha': sigma_ln_alpha,
        'rho_ln_T_ln_alpha': rho_ln_T_ln_alpha
    }


### RADIAL ###

def get_tau(t, T, alpha=None, mean_ln_alpha=None, fluctuate=True):

    if fluctuate:
        assert alpha is not None and mean_ln_alpha is not None, \
            "fluctuate requires alpha and mean_ln_alpha"
        beta = get_beta(alpha, T)
        mean_t = alpha / beta
        ### Eq (34)
        tau = (t / mean_t) * np.exp(mean_ln_alpha) / (np.exp(mean_ln_alpha) - 1)
    else:
        tau = t / T

    return tau

### Eq (23)
def radial_component(r, R):
    num = 2*r*R**2
    den = (r**2 + R**2)**2
    return num/den

### Eq (23)
def radial_pdf(r, p, R_core, R_tail):
    core = radial_component(r, R_core)
    tail = radial_component(r, R_tail)
    return p*core + (1-p)*tail

### Eq (24)
def get_R_core(tau, z_1, z_2):
    return z_1 + z_2 * tau

### Eq (25)
def get_R_tail(tau, k_1, k_2, k_3, k_4):
    term1 = np.exp(k_3 * (tau - k_2))
    term2 = np.exp(k_4 * (tau - k_2))
    return k_1 * (term1 + term2)

### Eq (26)
def get_p(tau, p_1, p_2, p_3):
    tau_prime = (p_2 - tau) / p_3
    p = p_1 * np.exp(tau_prime - np.exp(tau_prime))
    oob = False
    if isinstance(tau, np.ndarray):
        if (p<0).any() or (p>1).any():
            oob = True
    elif p < 0 or p > 1:
        oob = True
    if oob:
        print("WARNING: p not in [0, 1]. Clamping...")
        p = np.clip(p, 0, 1)
    return p

### Putting it all together (radial)
def get_radial_parameters(tau, E, Z):

    ### A.1.3
    lnE = np.log(E)
    z_1 = 0.0251 + 0.00319*lnE
    z_2 = 0.1162 - 0.000381*Z
    k_1 = 0.659  - 0.00309*Z
    k_2 = 0.645
    k_3 = -2.59
    k_4 = 0.3585 + 0.0421*lnE
    p_1 = 2.632  - 0.00094*Z
    p_2 = 0.401  + 0.00187*Z
    p_3 = 1.313  - 0.0686*lnE

    R_C = get_R_core(tau=tau, z_1=z_1, z_2=z_2)
    R_T = get_R_tail(tau=tau, k_1=k_1, k_2=k_2, k_3=k_3, k_4=k_4)
    p = get_p(tau=tau, p_1=p_1, p_2=p_2, p_3=p_3)

    return R_C, R_T, p

### RADIAL SAMPLING ###

### Eq (28)
def sample_radii(R_core, R_tail, p, N):
    size = N
    if isinstance(p, np.ndarray):
        assert len(p) == len(R_core) == len(R_tail), \
            "inputs must have same length"
        ### Expand dims to broadcast to (size, N)
        R_core = R_core[..., None]
        R_tail = R_tail[..., None]
        p = p[..., None]
        size = list(p.shape)
        size[-1] = N
    v = np.random.uniform(0, 1, size)
    w = np.random.uniform(0, 1, size)
    ### Note: there appears to be a typo in the paper (they do p < w)
    R_mixed = np.where(w < p, R_core, R_tail)
    return R_mixed * np.sqrt(v / (1 - v))

### Eq (31)
def get_num_spots_total(E, Z):
    N = 93 * np.log(Z) * E ** 0.876
    if isinstance(E, np.ndarray):
        N = np.clip(N.astype(int), 1, None)
    else:
        N = max(1, int(N))
    return N

### Eq (32) and (33)
def get_num_spots_layer(t_lo, t_hi, alpha, T, Z, N_total=None, E=None):

    if N_total is None:
        assert E is not None, "Either N_total or E must be provided"
        N_total = get_num_spots_total(E, Z)

    T_spot     = T     * (0.698 + 0.00212*Z)
    alpha_spot = alpha * (0.639 + 0.00334*Z)
    beta_spot  = get_beta(alpha_spot, T_spot)
    
    ### Fraction of spots in this layer
    frac = longitudinal_cdf(t_hi, alpha_spot, beta_spot) \
         - longitudinal_cdf(t_lo, alpha_spot, beta_spot)
    
    if isinstance(alpha, np.ndarray):
        N_layer = N_total * frac
        N_layer = np.clip(N_layer.astype(int), 1, None)
    else:
        N_layer = max(1, int(N_total * frac))

    return N_layer


def shoot(Es, Z, t_edges, flatten=True, N_spots_per_layer=None):

    ### Flexible input handling
    if isinstance(Es, float) or isinstance(Es, int):
        Es = np.array([Es])

    assert len(t_edges) >= 2, "t_edges must have at least 2 edges"
    assert np.all(np.diff(t_edges) > 0), "t_edges must be in ascending order"

    N_layers = len(t_edges) - 1

    ### Longitudinal parameters: each (N_particles,)
    long_params = get_longitudinal_parameters(Es, Z)

    t_lo  = t_edges[:-1]       # (N_layers,)
    t_hi  = t_edges[1:]
    t_mid = (t_lo + t_hi) / 2

    ### Broadcast to (N_particles, 1)
    alpha         = long_params['alpha'][:, None]
    beta          = long_params['beta'][:, None]
    T             = long_params['T'][:, None]
    mean_ln_alpha = long_params['mean_ln_alpha'][:, None]

    ### Energy per layer: (N_particles, N_layers)
    dE = Es[:, None] * (longitudinal_cdf(t_hi, alpha, beta)
                      - longitudinal_cdf(t_lo, alpha, beta))

    ### Radial parameters: (N_particles, N_layers)
    tau = get_tau(t_mid, T, alpha=alpha, mean_ln_alpha=mean_ln_alpha)
    R_core, R_tail, p = get_radial_parameters(tau, Es[:, None], Z)

    ### Fix spots per layer to allow vectorized ops
    if N_spots_per_layer is None:
        N_total = get_num_spots_total(Es, Z)
        N_spots_per_layer = max(1, int(np.mean(N_total) / N_layers))
        N_spots_per_layer = min(N_spots_per_layer, 100_000) # avoid OOM

    ### Sample radii and angles: (N_particles, N_layers, N_spots_per_layer)
    r   = sample_radii(R_core, R_tail, p, N_spots_per_layer)
    phi = np.random.uniform(0, 2*np.pi, r.shape)

    ### Additional information
    spot_E = dE / N_spots_per_layer
    spot_E_bc = np.broadcast_to(spot_E[:, :, None], r.shape)
    t_mid_bc = np.broadcast_to(t_mid[None, :, None], r.shape)
    particle_idx = np.arange(len(Es))
    particle_idx_bc = np.broadcast_to(particle_idx[:, None, None], r.shape)

    ### Results
    out_dict = {
        'E':   spot_E_bc,
        't':   t_mid_bc,
        'r':   r,
        'phi': phi,
        'particle_idx': particle_idx_bc
    }

    if flatten:
        out_dict = {k: v.ravel() for k, v in out_dict.items()}

    return out_dict

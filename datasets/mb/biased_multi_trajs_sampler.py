import jax
from jax import numpy as jnp
from jax import grad, random, lax, vmap
import numpy as onp
import os
import sys

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

os.environ["XLA_PYTHON_CLIENT_MEN_FRACTION"] = "0.95"

class MullerBrownLangevinMD:
    def __init__(self, dt=0.01, steps=1000, mass=1.0, 
                 gamma=0.1, temperature=1.0, seed=0,
                 bias_potential=None):
        self.dt = dt
        self.steps = steps
        self.mass = mass
        self.gamma = gamma
        self.kT = temperature
        self.noise_scale = jnp.sqrt(2 * gamma * temperature / mass)
        self.key = random.PRNGKey(seed)
        self.bias_potential = bias_potential if bias_potential is not None else (lambda xy: 0.0)

    @staticmethod
    def potential(xy):
        x, y = xy
        term1 = -17.3 * jnp.exp(-0.0039 * (x - 48)**2 - 0.0391 * (y - 8)**2)
        term2 = -8.7 * jnp.exp(-0.0039 * (x - 32)**2 - 0.0391 * (y - 16)**2)
        term3 = -14.7 * jnp.exp(-0.0254 * (x - 24)**2 + 0.043 * (x - 24) * (y - 32) - 0.0254 * (y - 32)**2)
        term4 = 1.3 * jnp.exp(0.00273 * (x - 16)**2 + 0.0023 * (x - 16) * (y - 24) + 0.00273 * (y - 24)**2)
        
        return term1 + term2 + term3 + term4

    def biased_potential(self, xy):
        bias = self.bias_potential(xy)
        return self.potential(xy) + bias

    def force(self, xy):
        return -grad(self.biased_potential)(xy)

    def simulate_single_traj(self, xy0, v0, key):
        def step_fn(carry, _):
            xy, v, key = carry
            key, subkey = random.split(key)
            f = self.force(xy)
            noise = self.noise_scale * random.normal(subkey, shape=(2,))
            v_new = v + (self.dt * f / self.mass) - self.gamma * v * self.dt + noise * jnp.sqrt(self.dt)
            xy_new = xy + v_new * self.dt
            e = self.potential(xy_new)
            return (xy_new, v_new, key), (xy_new, v_new, f, e)

        init_state = (jnp.array(xy0), jnp.array(v0), key)
        _, (positions, velocities, forces, energies) = lax.scan(step_fn, init_state, None, length=self.steps)
        return positions, velocities, forces, energies

    def run_multiple(self, n_traj=10):
        self.key, subkey = random.split(self.key)
        keys = random.split(subkey, n_traj)
        self.key, pos_key, vel_key = random.split(self.key, 3)
        xy0s = random.uniform(pos_key, shape=(n_traj, 2), minval=10.0, maxval=50.0)
        v0s = random.normal(vel_key, shape=(n_traj, 2)) * 0.1

        batched_sim = vmap(self.simulate_single_traj, in_axes=(0, 0, 0))
        positions, velocities, forces, energies = batched_sim(xy0s, v0s, keys)
        return positions, velocities, forces, energies

    def save_multiple(self, filename="langevin_multi_trajs.npz", **kwargs):
        onp.savez(filename, **{k: onp.array(v) for k, v in kwargs.items()})

def compute_unbiased_forces(positions):
    force_fn = lambda xy: -grad(MullerBrownLangevinMD.potential)(xy)
    return vmap(vmap(force_fn))(positions)

def compute_importance_weights(positions, bias_potential, temperature):
    beta = 1.0 / temperature
    bias_fn = vmap(vmap(bias_potential))
    bias_energies = bias_fn(positions)
    weights = jnp.exp(beta * bias_energies)
    return weights


if __name__ == "__main__":
    # def gaussian_bump_bias(x):
    #     return 3.0 * jnp.exp(-0.5 * ((x - 2.0) / 0.5)**2)
    # def nonlinear_bias(xy):
    #     x, y = xy
    #     x0 = 25.0
    #     y0 = 25.0
    #     width = 5.0
    #     height = 10.0
    #     return height * jnp.exp(-((x - x0)**2 + (y - y0)**2) / (2 * width**2))

    # bias only in x direction
    def remove_barrier(xy):
        x, y = xy
        x0 = 32.0
        width = 5
        height = -4
        return height * jnp.exp(-((x - x0)**2) / (2 * width**2))
    
    
    # bias only in y direction
    # def nonlinear_bias(xy):
    #     x, y = xy
    #     y0 = 30.0
    #     width = 4.0
    #     height = 6.0
    #     return height * jnp.exp(-((y - y0)**2) / (2 * width**2))

    steps = 10000000
    bias_potential = remove_barrier
    sim = MullerBrownLangevinMD(dt=0.1, steps=steps, bias_potential=bias_potential)
    positions, velocities, forces_biased, energies = sim.run_multiple(n_traj=10)


    importance_weights = compute_importance_weights(positions, bias_potential, sim.kT)
    forces_unbiased = compute_unbiased_forces(positions)
    sim.save_multiple(f"biased_langevin_multi_trajs_{steps}_along_x.npz",
                      positions=positions,
                    #   velocities=velocities,
                      energies=energies,
                      forces_biased=forces_biased,
                      forces_unbiased=forces_unbiased,
                      importance_weights=importance_weights)

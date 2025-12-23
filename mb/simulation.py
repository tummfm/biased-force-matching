import jax
import jax.numpy as jnp
from jax import random, grad, lax
import numpy as onp
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from flax.core import FrozenDict
from models.rbf_mlp import RBFMLP

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

os.environ["XLA_PYTHON_CLIENT_MEN_FRACTION"] = "0.95"

class Langevin1D_NNP:
    def __init__(self, model, params, dt=0.01, steps=1000, mass=1.0, gamma=1.0, temperature=1.0, seed=0):
        self.model = model
        self.params = params
        self.dt = dt
        self.steps = steps
        self.mass = mass
        self.gamma = gamma
        self.kT = temperature
        self.noise_scale = jnp.sqrt(2 * gamma * temperature / mass)
        self.key = random.PRNGKey(seed)

    def potential(self, x):
            # Apply a prior to stable simulation
            nnp_potential = self.model.apply(self.params, x.reshape(-1, 1)).squeeze()
            wall_strength = 1000.0  # High spring constant for the wall
            lower_bound = 0.0
            upper_bound = 60.0

            wall_left = jnp.where(x < lower_bound, 0.5 * wall_strength * (x - lower_bound)**2, 0.0)
            wall_right = jnp.where(x > upper_bound, 0.5 * wall_strength * (x - upper_bound)**2, 0.0)

            return nnp_potential + wall_left + wall_right

    def force(self, x):
        return -grad(self.potential)(x)

    def simulate(self, x0, v0):
        def step_fn(carry, _):
            x, v, key = carry
            key, subkey = random.split(key)
            f = self.force(x)
            noise = self.noise_scale * random.normal(subkey, shape=())
            v_new = v + (self.dt * f / self.mass) - self.gamma * v * self.dt + noise * jnp.sqrt(self.dt)
            x_new = x + v_new * self.dt
            e = self.potential(x_new)
            return (x_new, v_new, key), (x_new, v_new, f, e)

        init_state = (x0, v0, self.key)
        _, (positions, velocities, forces, energies) = lax.scan(step_fn, init_state, None, length=self.steps)
        return positions, velocities, forces, energies

    def run_multiple(self, n_traj=10):
        self.key, subkey = random.split(self.key)
        keys = random.split(subkey, n_traj)
        self.key, pos_key, vel_key = random.split(self.key, 3)
        x0s = random.uniform(pos_key, shape=(n_traj,), minval=10.0, maxval=50.0)
        v0s = random.normal(vel_key, shape=(n_traj,)) * 0.1

        batched_sim = jax.vmap(self.simulate, in_axes=(0, 0))
        positions, velocities, forces, energies = batched_sim(x0s, v0s)
        return positions, velocities, forces, energies

    def save_multiple(self, filename="langevin_nnp_trajs.npz", **kwargs):
        onp.savez(filename, **{k: onp.array(v) for k, v in kwargs.items()})

def load_model_and_params():
    with open("mb/biased_weights/model_biased_200000_seed2.pkl", "rb") as f:
        params = onp.load(f, allow_pickle=True).item()
    model = RBFMLP([128, 128, 128, 128])
    return model, FrozenDict(params)

if __name__ == "__main__":
    model, params = load_model_and_params()
    sim = Langevin1D_NNP(model=model, params=params, dt=0.1, steps=1500000, mass=1.0, gamma=1.0, temperature=1.0, seed=11)
    positions, velocities, forces, energies = sim.run_multiple(n_traj=10)

    sim.save_multiple("biased_xy_langevin_nnp_trajs.npz",
                      positions=positions,
                      velocities=velocities,
                      forces=forces,
                      energies=energies)

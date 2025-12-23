import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

def loss_fn(params, apply_fn, x, fx_true):
    def energy_fn(x):
        return apply_fn(params, x)
    _, g_vjp = jax.vjp(energy_fn, x)
    fx_pred = -g_vjp(jax.numpy.ones_like(x))[0]
    return jnp.mean((fx_pred - fx_true) ** 2)

@jax.jit
def train_step(state, x, fx_true):
    grads = jax.grad(loss_fn)(state.params, state.apply_fn, x, fx_true)
    return state.apply_gradients(grads=grads)

def create_train_state(rng, model, x_example, learning_rate):
    params = model.init({"params": rng}, x_example)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

def evaluate_loss(state, dataloader):
    total_loss = 0.0
    for x_batch, fx_batch in dataloader:
        x_batch = jnp.array(x_batch.numpy())
        fx_batch = jnp.array(fx_batch.numpy())
        loss_val = loss_fn(state.params, state.apply_fn, x_batch, fx_batch)
        total_loss += loss_val
    return total_loss / len(dataloader)
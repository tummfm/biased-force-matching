import jax
import jax.numpy as jnp
import optax
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import argparse
from mb.train.train_utils import train_step, create_train_state, evaluate_loss
import numpy as onp
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

os.environ["XLA_PYTHON_CLIENT_MEN_FRACTION"] = "0.95"

from mb.models.mlp import MLP
from mb.models.rbf_mlp import RBFMLP
from mb.data.mb_data import get_mb_dataloader
from collections import OrderedDict

def get_default_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("device", type=str, default="-1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    print(f"Running on device: {args.device}")

    return OrderedDict(
        dataset=OrderedDict(
            num_samples=200000,
        ),
        model=OrderedDict(
            features=[128, 128, 128, 128],
        ),
        optimizer=OrderedDict(
            name="adam",
            learning_rate=1e-4,
            weight_decay=0.0,
            epochs=args.epochs,
            batch=args.batch,
        ),
    )

def train():
    config = get_default_config()
    rng = jax.random.PRNGKey(0)
    dataloader = get_mb_dataloader(batch_size=config["optimizer"]["batch"],
                                   num_samples=config["dataset"]["num_samples"])
    model = RBFMLP(
        hidden_layers=config["model"]["features"],
        num_rbf_centers=100,  
        sigma=5.0            
    )

    state = create_train_state(rng, model, jnp.ones((config["optimizer"]["batch"], 1)), config["optimizer"]["learning_rate"])

    for epoch in range(config["optimizer"]["epochs"]):
        for x_batch, fx_batch in dataloader:
            x_batch = jnp.array(x_batch.numpy())
            fx_batch = jnp.array(fx_batch.numpy())
            state = train_step(state, x_batch, fx_batch)
        train_loss = evaluate_loss(state, dataloader)
        print(f"Epoch {epoch}/{config['optimizer']['epochs']}, Loss: {train_loss:.4f}")

    with open(f"unbiased_weights_retrain/test_model_biased_{config['dataset']['num_samples']}.pkl", "wb") as f:
        onp.save(f, state.params)
        
    print("Model saved to model.pkl")

    return state, model

def inference(state, model):
    x = jnp.linspace(10, 50, 500).reshape(-1, 1)
    def energy_fn(x):
        return model.apply(state.params, x)
    energy = energy_fn(x)
    _, g_vjp = jax.vjp(energy_fn, x)
    fx_pred = -g_vjp(jax.numpy.ones_like(x))[0]
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(x, energy)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("Energy")
    axs[0].set_title("Energy vs x")
    axs[1].plot(x, fx_pred)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("Force")
    axs[1].set_title("Force vs x")
    plt.tight_layout()
    plt.savefig("energy_force_biased_only_x.png")
    with open("biased_only_x_energy_force.pkl", "wb") as f:
        onp.save(f, (x, energy, fx_pred))

if __name__ == "__main__":
    state, model = train()
    inference(state, model)
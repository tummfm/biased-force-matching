import argparse
import os

# Parse minimal args and set environment before heavy imports
parser = argparse.ArgumentParser()
parser.add_argument(
    '--device',
    type=str,
    help='GPU or MIG UUID to use for training'
)
args, _ = parser.parse_known_args()

if args.device:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'

import json
from typing import Any

from chemutils.datasets import pepsol
from chemutils.models import mace
from chemtrain import trainers
from chemtrain.data import preprocessing
from utils import plot_predictions, plot_convergence

import numpy as onp
from jax import numpy as jnp, random, tree_util
from jax_md import partition, space
import optax

def parse_args() -> argparse.Namespace:
    """
    Parse full command-line arguments for training.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with 'device'.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        type=str,
        help='GPU or MIG UUID to use for training'
    )
    return parser.parse_args()

def load_and_split_data(
    data_path: str,
    scale_R: float = 1.0,
    scale_U: float = 1.0,
    fractional: bool = True,
    train_frac: float = 1.0,
    seed: int = 11
) -> dict[str, dict[str, Any]]:
    """
    Load and split NPZ dataset into train/validation/test, with scaling.

    Parameters
    ----------
    data_path : str
        Path to NPZ file.
    scale_R : float
        Coordinate scaling factor.
    scale_U : float
        Energy scaling factor.
    fractional : bool
        Use fractional coordinates.
    train_frac : float
        Fraction of training data to keep.
    seed : int
        RNG seed for splitting.

    Returns
    -------
    dict[str, dict[str, Any]]
        Split datasets with keys 'training', 'validation', 'testing'.
    """
    raw = onp.load(data_path, allow_pickle=True)
    data = dict(raw)

    train_data, val_data, test_data = preprocessing.train_val_test_split(
        data, train_ratio=0.9, val_ratio=0.1, shuffle=True, shuffle_seed=seed
    )
    splits = {'training': train_data, 'validation': val_data, 'testing': test_data}

    for key, subset in splits.items():
        splits[key] = pepsol.scale_dataset(
            subset, scale_R=scale_R, scale_U=scale_U, fractional=fractional
        )

    # Reduce training size
    n_train = splits['training']['R'].shape[0]
    keep = int(train_frac * n_train)
    for field in splits['training']:
        splits['training'][field] = splits['training'][field][:keep]

    return splits


def setup_neighborlist(
    dataset_split: dict[str, Any],
    cutoff: float,
    box: jnp.ndarray,
    batch_size: int,
    fractional: bool = True
) -> tuple[Any, tuple[int, int, float]]:
    """
    Allocate neighbor list for the training split.

    Parameters
    ----------
    dataset_split : dict[str, Any]
        Dataset dict with 'R' and 'mask'.
    cutoff : float
        Neighbor cutoff radius.
    box : jnp.ndarray
        Simulation box.
    batch_size : int
        Batch size for preallocation.
    fractional : bool
        Use fractional coordinates.

    Returns
    -------
    tuple containing neighbor_fn and stats (max_neighbors, max_edges, avg_neighbors).
    """
    disp_fn, _ = space.periodic_general(box=box, fractional_coordinates=fractional)
    return preprocessing.allocate_neighborlist(
        dataset_split,
        disp_fn,
        box,
        r_cutoff=cutoff,
        mask_key='mask',
        box_key='box',
        format=partition.Sparse,
        batch_size=batch_size
    )


def build_model(
    displacement_fn: Any,
    hidden_irreps: str,
    readout_irreps: str,
    output_irreps: str,
    max_ell: int,
    num_interactions: int,
    correlation: int,
    cutoff: float,
    n_species: int,
    max_edges: int,
    avg_neighbors: float,
    seed: int = 0
) -> tuple[Any, Any]:
    """
    Build MACE model initializer and energy function.

    Returns
    -------
    tuple containing init_fn and energy_fn.
    """
    init_fn, gnn_energy_fn = mace.mace_neighborlist_pp(
        displacement_fn,
        r_cutoff=cutoff,
        n_species=n_species,
        max_edges=max_edges,
        per_particle=False,
        avg_num_neighbors=avg_neighbors,
        mode='energy',
        hidden_irreps=hidden_irreps,
        max_ell=max_ell,
        num_interactions=num_interactions,
        correlation=correlation,
        readout_mlp_irreps=readout_irreps,
        output_irreps=output_irreps,
    )

    def energy_fn_template(energy_params):
        def energy_fn(pos, neighbor, mode=None, **dynamic_kwargs):
            assert 'species' in dynamic_kwargs.keys(), 'species not in dynamic_kwargs'

            if "mask" not in dynamic_kwargs:
                print(f"Add defaul all-positive mask.")
                dynamic_kwargs["mask"] = jnp.ones(pos.shape[0], dtype=jnp.bool_)

            if "box" in dynamic_kwargs:
                print(f"Found box in energy kwargs")

            return gnn_energy_fn(
                energy_params, pos, neighbor, **dynamic_kwargs
            )
        return energy_fn

    return init_fn, energy_fn_template


def train_model(
    init_params: Any,
    energy_fn: Any,
    neighbor_fn: Any,
    dataset: dict[str, dict[str, Any]],
    output_dir: str,
    batch_size: int,
    num_epochs: int,
    init_lr: float,
    decay_rate: float
) -> trainers.ForceMatching:
    """
    Train the model with force matching.

    Returns
    -------
    trainers.ForceMatching
    """
    num_samples = dataset['training']['R'].shape[0]
    total_steps = (num_epochs * num_samples) // batch_size
    scheduler = optax.exponential_decay(
        init_value=init_lr,
        transition_steps=total_steps,
        decay_rate=decay_rate
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(scheduler),
        optax.scale(-1.0)
    )

    trainer = trainers.ForceMatching(
        init_params,
        optimizer,
        energy_fn,
        neighbor_fn,
        log_file=os.path.join(output_dir, 'force_matching.log'),
        batch_per_device=batch_size,
    )
    trainer.set_dataset(dataset['training'], stage='training')
    trainer.set_dataset(dataset['validation'], stage='validation', include_all=True)
    trainer.set_dataset(dataset['testing'], stage='testing', include_all=True)

    trainer.train(num_epochs)
    return trainer


def save_results(
    trainer: trainers.ForceMatching,
    dataset: dict[str, dict[str, Any]],
    output_dir: str,
    MACE_CONFIG: dict[str, Any],
    TRAIN_CONFIG: dict[str, Any]
) -> None:
    """
    Save models, params, and predictions.
    """
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_trainer(os.path.join(output_dir, 'trainer.pkl'), format='.pkl')
    trainer.save_energy_params(os.path.join(output_dir, 'best_params.pkl'), '.pkl', best=True)
    trainer.save_energy_params(os.path.join(output_dir, 'final_params.pkl'), '.pkl', best=False)

    preds_val = trainer.predict(dataset['validation'], trainer.best_params, batch_size=TRAIN_CONFIG['batch_size'])
    preds_test = trainer.predict(dataset['testing'], trainer.best_params, batch_size=TRAIN_CONFIG['batch_size'])

    onp.savez(os.path.join(output_dir, 'predictions_val.npz'), **tree_util.tree_map(onp.asarray, preds_val))
    onp.savez(os.path.join(output_dir, 'predictions_test.npz'), **tree_util.tree_map(onp.asarray, preds_test))

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(MACE_CONFIG, f, indent=4)
    with open(os.path.join(output_dir, 'train_config.json'), 'w') as f:
        json.dump(TRAIN_CONFIG, f, indent=4)


def main() -> None:
    """
    Execute training pipeline end-to-end.
    """
    args = parse_args()
    if args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    MEM_FRACTION = 1.0
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(MEM_FRACTION)

    MACE_CONFIG = {
        "hidden_irreps": "32x0e+32x1o", 
        "readout_mlp_irreps": "16x0e", 
        "output_irreps": "1x0e", 
        "max_ell": 3,
        "num_interactions": 2,
        "correlation": 2, 
        "r_cutoff": 0.5,
        "mol": "ala2",
        "CG_map": "heavyOnly",
        "type": "CG",  # "AT" or "CG"
        "PRNGKey_seed": 22,
        "data_path": "/ala2/welltemp_6.npz",
        "train_frac": 1.0,  
    }
    TRAIN_CONFIG = {
        "batch_size": 256,
        "num_epochs": 401,
        "init_lr": 0.001,
        "decay_rate": 0.9,  # Decay rate for learning rate
        "optimizer": "adam+decay",
    }

    prefix = prefix + "_37a" if '37a' in MACE_CONFIG['data_path'] else prefix

    MACE_CONFIG['output_dir'] = (
        f"model_out/pub_{prefix}_frac={MACE_CONFIG['train_frac']}_"
        f"mol={MACE_CONFIG['mol']}_map={MACE_CONFIG['CG_map']}_"
        f"batch={TRAIN_CONFIG['batch_size']}_"
        f"hidden_irreps={MACE_CONFIG['hidden_irreps']}_"
        f"epochs={str(TRAIN_CONFIG['num_epochs'])}"
    )

    if os.path.exists(MACE_CONFIG['output_dir']):
        raise FileExistsError(f"Output directory {MACE_CONFIG['output_dir']} already exists. "
                              "Please change the output directory or remove the existing one.")
    os.makedirs(MACE_CONFIG['output_dir'], exist_ok=True)    
    
    dataset = load_and_split_data(
        MACE_CONFIG['data_path'],
        scale_R=1,
        scale_U=1,
        fractional=True,
        train_frac=MACE_CONFIG['train_frac'],
        seed=MACE_CONFIG['PRNGKey_seed']
    )

    # Neighbor list
    box = jnp.asarray(dataset['training']['box'][0])
    neighbor_fn, stats = setup_neighborlist(
        dataset['training'],
        cutoff=MACE_CONFIG['r_cutoff'],
        box=box,
        batch_size=TRAIN_CONFIG['batch_size']
    )

    # Model init
    n_species = len(dataset['training']['species'][0])
    init_fn, energy_fn_template = build_model(
        displacement_fn=space.periodic_general(box=box, fractional_coordinates=True)[0],
        hidden_irreps=MACE_CONFIG['hidden_irreps'],
        readout_irreps=MACE_CONFIG['readout_mlp_irreps'],
        output_irreps=MACE_CONFIG['output_irreps'],
        max_ell=MACE_CONFIG['max_ell'],
        num_interactions=MACE_CONFIG['num_interactions'],
        correlation=MACE_CONFIG['correlation'],
        cutoff=MACE_CONFIG['r_cutoff'],
        n_species=n_species,
        max_edges=stats[1],
        avg_neighbors=stats[2],
        seed=MACE_CONFIG['PRNGKey_seed']
    )
    key = random.PRNGKey(MACE_CONFIG['PRNGKey_seed'])
    init_params = init_fn(
        key,
        jnp.asarray(dataset['training']['R'][0]),
        neighbor_fn,
        species=jnp.asarray(dataset['training']['species'][0]),
        mask=jnp.asarray(dataset['training']['mask'][0])
    )

    # Train
    trainer = train_model(
        init_params,
        energy_fn_template,
        neighbor_fn,
        dataset,
        output_dir=MACE_CONFIG['output_dir'],
        batch_size=TRAIN_CONFIG['batch_size'],
        num_epochs=TRAIN_CONFIG['num_epochs'],
        init_lr=TRAIN_CONFIG['init_lr'],
        decay_rate=TRAIN_CONFIG['decay_rate']
    )

    save_results(trainer, dataset, MACE_CONFIG['output_dir'], MACE_CONFIG, TRAIN_CONFIG)
    plot_predictions(
        trainer.predict(dataset['validation'], trainer.best_params, batch_size=TRAIN_CONFIG['batch_size']),
        dataset['validation'],
        MACE_CONFIG['output_dir'],
        name='preds_validation'
    )
    plot_predictions(
        trainer.predict(dataset['testing'], trainer.best_params, batch_size=TRAIN_CONFIG['batch_size']),
        dataset['testing'],
        MACE_CONFIG['output_dir'],
        name='preds_testing'
    )
    plot_convergence(trainer, MACE_CONFIG['output_dir'])


if __name__ == '__main__':
    main()

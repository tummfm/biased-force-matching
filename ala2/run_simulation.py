import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, help='GPU or MIG UUID')
parser.add_argument('--model', type=str, help='Model path', required=True)
parser.add_argument('--ens', type=str, help='Ensemble', required=True)
args = parser.parse_args()

# Set device
if args.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import json
import pickle
import numpy as onp
from jax import numpy as jnp, tree_util
import jax

from chemtrain.data import preprocessing
from chemtrain import quantity, util
from chemutils.models import mace
from jax import random
from chemtrain.ensemble import sampling
from jax_md import partition, space, simulate
from jax_md_mod import custom_quantity
from chemutils.datasets import pepsol



# -------------------------
# Configuration handling
# -------------------------
# load training config if available
model_path = args.model
base_dir = os.path.dirname(model_path)
mace_config_path = os.path.join(base_dir, "config.json")    
if os.path.exists(mace_config_path):
    with open(mace_config_path, 'r') as f:
        # load 
        MACE_CONFIG = json.load(f)
else:
    raise FileNotFoundError(f"Config file {mace_config_path} not found.")


CONFIG_DEFAULTS = {
    "gamma": 100.0,
    "dt_values_fs": [2], # fs
    "PRNGKey_seed": 22,
    "print_every": 0.5,  # Print every n steps
}

if args.ens == "NVT":
    CONFIG_DEFAULTS['ensemble'] = "NVT"
    CONFIG_DEFAULTS['t_eq'] = 10  # 10 ps equilibration
    CONFIG_DEFAULTS['t_total'] = 10000  # 10000 ps total simulation time
elif args.ens == "NVE":
    CONFIG_DEFAULTS['ensemble'] = "NVE"
    CONFIG_DEFAULTS['t_eq'] = 0  # No equilibration for NVE
    CONFIG_DEFAULTS['t_total'] = 100
else:
    raise ValueError(f"Unknown ensemble: {args.ens}. Use NVT or NVE.")
    
if MACE_CONFIG['mol'] == "ala2":
    CONFIG_DEFAULTS['n_chains'] = 100
    MACE_CONFIG['nmol'] = 1    
    CONFIG_DEFAULTS['kT'] = 300. * quantity.kb 
else: 
    raise ValueError(f"Unknown molecule: {MACE_CONFIG['mol']}. Use 'ala2'.")

config = CONFIG_DEFAULTS.copy()
config['type'] = MACE_CONFIG['type']
config['cg_map'] = MACE_CONFIG.get('CG_map', None)

print('-'*50)
for key, value in MACE_CONFIG.items():
    print(f"Found MACE config: {key}: {value}")
print('-'*50)
for key, value in config.items():
    print(f"Using Sim config: {key}: {value}")
print('-'*50)

# -------------------------
# Load dataset
# -------------------------
dataset = onp.load("/ala2/37a/unbiased_05fs.npz", allow_pickle=True)
print(f"Loaded dataset from rerun path: unbiased one")

dataset = dict(dataset)

train_data, val_data, test_data = preprocessing.train_val_test_split(dataset, shuffle=True, shuffle_seed=11)
dataset_ = {
    'training': train_data,
    'validation': val_data,
    'testing': test_data,
}

for split in dataset_.keys():
    dataset_[split]['R'] = dataset_[split]['R']
    dataset_[split]['F'] = dataset_[split]['F']

    dataset_[split]['box'] = dataset_[split]['box']
    dataset_[split]['species'] = dataset_[split]['species']
    dataset_[split]['mask'] = dataset_[split]['mask']

dataset_frac = {}
splits = dataset_.keys()
for split in splits:
    out = pepsol.scale_dataset(
        dataset_[split],
        scale_R=1,
        scale_U=1,
        fractional=True
    )
    dataset_frac[split] = out
    
dataset = dataset_frac
species = dataset['training']['species'][0]
n_species = len(species)
masses = jnp.array([12.011, 12.011, 15.999, 14.007, 12.011, 12.011, 12.011, 15.999, 14.007, 12.011])

dataset_ref = onp.load('/ala2/37a/unbiased_05fs.npz', allow_pickle=True)

dataset_ref = dict(dataset_ref)

train_data, val_data, test_data = preprocessing.train_val_test_split(dataset_ref, shuffle=True, shuffle_seed=11)
dataset_ref_ = {
    'training': train_data,
    'validation': val_data,
    'testing': test_data,
}

for split in dataset_ref_.keys():
    dataset_ref_[split]['R'] = dataset_ref_[split]['R']
    dataset_ref_[split]['F'] = dataset_ref_[split]['F']
    dataset_ref_[split]['box'] = dataset_ref_[split]['box']
    dataset_ref_[split]['species'] = dataset_ref_[split]['species']
    dataset_ref_[split]['mask'] = dataset_ref_[split]['mask']

    
reference_X = onp.concatenate([
    dataset_ref_['training']['R'],
    dataset_ref_['validation']['R'],
    dataset_ref_['testing']['R']
], axis=0)

print("Training set size:", dataset_['training']['R'].shape[0])
print("Validation set size:", dataset_['validation']['R'].shape[0])
print("Testing set size:", dataset_['testing']['R'].shape[0])


# -------------------------
# Neighbor list setup
# -------------------------
r_cutoff = MACE_CONFIG['r_cutoff']
box=dataset['training']['box'][0]


displacement_fn, _ = space.periodic_general(
    box=box, fractional_coordinates=True)

nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = preprocessing.allocate_neighborlist(
    dataset['training'], 
    displacement_fn,
    box,
    r_cutoff=MACE_CONFIG["r_cutoff"], 
    mask_key="mask", 
    box_key="box",
    format=partition.Sparse,
    batch_size=100
)
# -------------------------
# Model initialization
# -------------------------
init_fn, gnn_energy_fn = mace.mace_neighborlist_pp(
    displacement_fn, 
    r_cutoff, 
    n_species,
    max_edges=max_edges, 
    per_particle=False,
    avg_num_neighbors=avg_num_neighbors, 
    mode='energy',
    hidden_irreps=MACE_CONFIG["hidden_irreps"],
    max_ell=MACE_CONFIG["max_ell"],
    num_interactions=MACE_CONFIG["num_interactions"],
    correlation=MACE_CONFIG["correlation"],
    readout_mlp_irreps=MACE_CONFIG["readout_mlp_irreps"],
    output_irreps=MACE_CONFIG["output_irreps"],
)

def energy_fn_template(energy_params):
    def energy_fn(pos, neighbor, **dynamic_kwargs):
        dynamic_kwargs.setdefault('species', species)
        
        if 'box' not in dynamic_kwargs.keys():
            print("Use default box")
            
        gnn_energy = gnn_energy_fn(
            energy_params, pos, neighbor, **dynamic_kwargs)
        return gnn_energy
    return energy_fn

print(f"Max neighbors: {max_neighbors}, max edges: {max_edges}")


# -------------------------
# Load model parameters
# -------------------------
model_path = args.model
base_dir = os.path.dirname(model_path)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found.")
outdir = os.path.join(base_dir, f"simulation_{config['ensemble']}")
os.makedirs(outdir, exist_ok=True)

energy_params = onp.load(model_path, allow_pickle=True)
energy_params = tree_util.tree_map(jnp.asarray, energy_params)
energy_fn = energy_fn_template(energy_params)


# -------------------------
# Simulator initialization
# -------------------------
def init_simulator(
    dataset, energy_fn, masses, nbrs_init,
    kT, dt, n_chains, gamma, t_eq, t_total
):
    key = random.PRNGKey(config['PRNGKey_seed'])
    _, shift_fn = space.periodic_general(
        dataset['validation']['box'][0], fractional_coordinates=True)
 
    key, split = random.split(key)
    selection = random.choice(
        split, jnp.arange(dataset['validation']['R'].shape[0]),
        shape=(n_chains,), replace=False
    )
    r_init = dataset['validation']['R'][selection]


    if config['ensemble'] == "NVT":
        init_simulator_fn = simulate.nvt_langevin
        sim_kwargs = {"kT": kT, "gamma": gamma, "dt": dt}
        init_sim_kwargs = {"mass": masses, "neighbor": nbrs_init}

    elif config['ensemble'] == "NVE":
        init_simulator_fn = simulate.nve
        sim_kwargs = {"dt": dt, "kT": kT}
        init_sim_kwargs = {"mass": masses, "neighbor": nbrs_init, "kT": kT}

    else:
        raise ValueError(f"Unknown ensemble: {config['ensemble']}. Use NVT or NVE.")

    init_ref_state, sim_template = sampling.initialize_simulator_template(
        init_simulator_fn,
        shift_fn=shift_fn, 
        nbrs=nbrs_init,
        init_with_PRNGKey=True,
        extra_simulator_kwargs=sim_kwargs,
    )

    key, split = random.split(key)
    reference_state = init_ref_state(
        split, 
        r_init,
        energy_or_force_fn=energy_fn,
        init_sim_kwargs=init_sim_kwargs
    )

    eval_timings = sampling.process_printouts(
        time_step=dt, total_time=t_total,
        t_equilib=t_eq, print_every=config['print_every'],
    )
    
    quantities = {
        'kT': custom_quantity.temperature,
        'epot': custom_quantity.energy_wrapper(lambda _: energy_fn),
        'force': custom_quantity.force_wrapper(lambda _: energy_fn),
        'etot': custom_quantity.total_energy_wrapper(lambda _: energy_fn),
        # 'ekin': custom_quantity.kinetic_energy_wrapper(lambda _: energy_fn)
    }
    
    traj_gen = sampling.trajectory_generator_init(
        sim_template, lambda _: energy_fn, eval_timings,
        quantities=quantities,
        vmap_sim_batch=config["n_chains"],
        vmap_batch=config["n_chains"],
    )

    return reference_state, jax.jit(traj_gen)


def visualise(traj):
    import visualise_traj
    visualise_traj.vis_ala2(
        traj,
        box,
        reference_X,
        config,
        mapping_change=True if "jan" in base_dir else False,
        name='Simulation'
    )

dt_values_fs = config['dt_values_fs']
dt_values_ps = [dt_fs * 0.001 for dt_fs in dt_values_fs]  # convert to ps

for dt_fs, dt_ps in zip(dt_values_fs, dt_values_ps):
    # Update config for this dt
    config['dt'] = dt_ps

    print(f"\nStarting simulation for dt = {dt_fs} fs ({dt_ps} ps)...")

    # Create output directory
    if "final_params" in model_path:
        folder_name = f"traj_final_dt={dt_fs}_teq={config['t_eq']}_t={config['t_total']}_nmol={MACE_CONFIG['nmol']}_nchain={config['n_chains']}/"
    else:
        folder_name = f"traj_dt={dt_fs}_teq={config['t_eq']}_t={config['t_total']}_nmol={MACE_CONFIG['nmol']}_nchain={config['n_chains']}/"
    save_dir = os.path.join(outdir, folder_name)
    
    # Skip simulation if folder already exists
    if os.path.exists(save_dir) and os.listdir(save_dir):
        print(f"Directory {save_dir} already exists and is not empty. Skipping simulation for dt = {dt_fs} fs.")
        try:
            visualise(save_dir)
        except Exception as e:
            print(f"Error during visualisation: {e}")
        continue
    os.makedirs(save_dir, exist_ok=True)

    reference_state, traj_generator = init_simulator(
        dataset, 
        energy_fn, 
        masses, 
        nbrs_init,
        kT=config['kT'],
        dt=dt_ps,
        n_chains=config['n_chains'],
        gamma=config['gamma'],
        t_eq=config['t_eq'],
        t_total=config['t_total']
    )

    traj_state = traj_generator(None, reference_state)

    # Save trajectory
    with open(os.path.join(save_dir, "trajectory.pkl"), "wb") as f:
        pickle.dump(traj_state.trajectory.position, f)

    with open(os.path.join(save_dir, "traj_state_aux.pkl"), "wb") as f:
        pickle.dump(traj_state.aux, f)

    config_ = config.copy()
    config_['dt'] = dt_ps
    config_.pop('dt_values_fs', None)
    
    # Save config used
    with open(os.path.join(save_dir, "traj_config.json"), "w") as cf:
        json.dump(config_, cf, indent=4)

    print(f"Finished dt = {dt_fs} fs. Results saved to {save_dir}.")

    try:
        visualise(save_dir)
    except Exception as e:
        print(f"Error during visualisation: {e}")
        
        
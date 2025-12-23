import os
import json
import pickle as pkl
from chemtrain import quantity
from matplotlib import pyplot as plt
import numpy as np
from jax import numpy as jnp, jit
from jax_md import space

import utils

def prepare_output_dir(traj_path: str) -> str:
    """
    Create an output directory named 'plots' next to a trajectory file.

    Ensures that a directory called 'plots' exists alongside the given
    trajectory file path. If it does not exist, it is created.

    Parameters
    ----------
    traj_path : str
        Path to a trajectory file.

    Returns
    -------
    str
        Path to the 'plots' directory where outputs will be saved.
    """
    outdir = os.path.join(os.path.dirname(traj_path), 'plots')
    os.makedirs(outdir, exist_ok=True)
    return outdir


def load_trajectory(traj_path: str) -> tuple[jnp.ndarray, dict]:
    """
    Load trajectory coordinates and auxiliary state from pickle files.

    Opens 'trajectory.pkl' and 'traj_state_aux.pkl' in the same directory as
    the provided path, and returns the trajectory as a JAX array along with
    auxiliary simulation data.

    Parameters
    ----------
    traj_path : str
        Path to one of the trajectory pickle files.

    Returns
    -------
    tuple[jnp.ndarray, dict]
        traj : JAX array of shape (n_frames, n_particles, 3)
            Simulation trajectory coordinates.
        aux : dict
            Auxiliary state information (energy, temperature, etc.).
    """
    base = os.path.dirname(traj_path)
    traj = pkl.load(open(os.path.join(base, 'trajectory.pkl'), 'rb'))
    aux = pkl.load(open(os.path.join(base, 'traj_state_aux.pkl'), 'rb'))
    return jnp.array(traj), aux


def periodic_displacement(box: np.ndarray, fractional: bool = False) -> tuple[callable, None]:
    """
    Create a periodic displacement function for simulating boundary conditions.

    Uses JAX MD's periodic_general to produce a function that calculates
    displacement vectors under periodic boundary conditions for a given box.

    Parameters
    ----------
    box : np.ndarray
        Array or matrix defining the simulation box.
    fractional : bool, optional
        Whether input coordinates are in fractional units, by default False.

    Returns
    -------
    tuple[callable, None]
        A function to compute periodic displacements and a placeholder None.
    """
    return space.periodic_general(box=box, fractional_coordinates=fractional)


def add_chain_lines(ax: plt.Axes, line_locations: list[int]) -> None:
    """
    Draw vertical lines to indicate the start of each chain segment.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to annotate.
    line_locations : list[int]
        List of time-step indices where new chains begin.
    """
    for loc in line_locations:
        ax.axvline(x=loc, color='r', linestyle='-', alpha=0.5)


def overlay_chains(
    ax: plt.Axes,
    data: np.ndarray,
    line_locations: list[int],
    y_label: str,
    title: str,
    relative: bool = False
) -> None:
    """
    Overlay multiple chain segments on a single plot.

    Splits a time-series into segments defined by line_locations, and plots
    each segment either in absolute or relative x-axis.

    Parameters
    ----------
    ax : plt.Axes
        Axes on which to draw the overlay.
    data : np.ndarray
        1D array of values to plot.
    line_locations : list[int]
        Indices delimiting chain boundaries.
    y_label : str
        Label for the Y-axis.
    title : str
        Plot title.
    relative : bool, optional
        If True, each segment is plotted from zero, by default False.
    """
    locs = [0] + list(line_locations) + [len(data)]
    for i in range(len(locs) - 1):
        segment = data[locs[i]:locs[i+1]]
        x_vals = range(len(segment)) if relative else range(locs[i], locs[i+1])
        ax.plot(x_vals, segment, alpha=0.7, label=f'Chain {i+1}')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    add_chain_lines(ax, line_locations)
    ax.legend()


def compute_line_locations(config: dict[str, float]) -> np.ndarray:
    """
    Compute chain boundary indices from simulation configuration.

    Based on total simulation time, equilibration time, output interval,
    and number of chains, returns the indices where each chain restarts.

    Parameters
    ----------
    config : dict[str, float]
        Simulation parameters including:
        - 't_total': total time (float)
        - 't_eq': equilibration time (float)
        - 'print_every': output interval (float, default 0.5)
        - 'n_chains': number of chains (int, default 1)

    Returns
    -------
    np.ndarray
        1D integer array of frame indices marking chain starts.
    """
    t_total = config['t_total']
    t_eq = config['t_eq']
    print_every = config.get('print_every', 0.5)
    n_chains = config.get('n_chains', 1)
    steps = int((t_total - t_eq) / print_every)
    arr = np.arange(0, steps * n_chains, steps)
    return arr[1:]


def split_into_chains(data: np.ndarray, line_locations: list[int]) -> np.ndarray:
    """
    Split array data into separate chains using boundary indices.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n, ...) to split along first axis.
    line_locations : list[int]
        Indices at which to split the array.

    Returns
    -------
    np.ndarray
        Array of shape (n_chains, segment_length, ...) after splitting.
    """
    segments: list[np.ndarray] = []
    start = 0
    for loc in line_locations:
        segments.append(data[start:loc])
        start = loc
    segments.append(data[start:])
    return np.array(segments)


def plot_time_series(
    traj_coords: np.ndarray,
    ref_coords: np.ndarray,
    indices: list[int],
    outpath: str,
    name: str,
    line_locations: list[int]
) -> None:
    """
    Plot Cartesian coordinates over time for selected atoms.

    Creates a two-panel figure showing reference and simulation trajectories
    for specified atom indices, with x/y/z as separate line styles.

    Parameters
    ----------
    traj_coords : np.ndarray
        Simulation coordinates, shape (n_frames, n_atoms, 3).
    ref_coords : np.ndarray
        Reference coordinates, same shape.
    indices : list[int]
        Atom indices to visualize.
    outpath : str
        Directory to save output images.
    name : str
        Label used for simulation plots.
    line_locations : list[int]
        Frame indices indicating chain breaks.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, data, title in zip(axes, [ref_coords, traj_coords], ['Reference', name]):
        for idx in indices:
            coord = data[:, idx]
            ax.plot(coord[:, 0], label=f'Atom {idx} x')
            ax.plot(coord[:, 1], linestyle='--', label=f'Atom {idx} y')
            ax.plot(coord[:, 2], linestyle=':', label=f'Atom {idx} z')
        ax.set_title(f'{title} Atom Coordinates (indices {indices})')
        ax.set_ylabel('Coordinate')
        ax.legend(loc='upper right')
        if title == name:
            add_chain_lines(ax, line_locations)
            ax.set_xlabel('Time step')
    plt.tight_layout()
    fname = f"Atom_coords_{'_'.join(map(str, indices))}.png"
    fig.savefig(os.path.join(outpath, fname), dpi=300)
    plt.close(fig)


def plot_dist_series(
    pairs: list[tuple[int, int]],
    ref_dists: list[np.ndarray],
    traj_dists: list[np.ndarray],
    outpath: str,
    name: str,
    line_locations: list[int]
) -> None:
    """
    Plot distance time-series for atom pairs in reference and trajectory.

    Generates two figures: one for reference distances and one for simulation,
    each showing distances for specified atom-pair indices over time.

    Parameters
    ----------
    pairs : list[tuple[int, int]]
        Atom index pairs for distance calculation.
    ref_dists : list[np.ndarray]
        Reference distances arrays per pair.
    traj_dists : list[np.ndarray]
        Simulation distances arrays per pair.
    outpath : str
        Directory for saving plots.
    name : str
        Label for simulation plots.
    line_locations : list[int]
        Frame indices where chains restart.
    """
    # Reference distances
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, dist in enumerate(ref_dists):
        ax.plot(dist, label=f'Dist {i} {pairs[i]}')
    add_chain_lines(ax, line_locations)
    ax.set_title('Reference Atom Pair Distances')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Distance')
    ax.legend(loc='upper right')
    fig.savefig(os.path.join(outpath, 'Reference_atom_pair_distances.png'), dpi=300)
    plt.close(fig)

    # Simulation distances
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for i, dist in enumerate(traj_dists):
        ax2.plot(dist, label=f'Dist {i} {pairs[i]}')
    add_chain_lines(ax2, line_locations)
    ax2.set_title(f'{name} Atom Pair Distances')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Distance')
    ax2.legend(loc='upper right')
    fig2.savefig(os.path.join(outpath, f'{name}_atom_pair_distances.png'), dpi=300)
    plt.close(fig2)


def plot_dihedrals(
    AT_phi: np.ndarray,
    AT_psi: np.ndarray,
    Traj_phi: np.ndarray,
    Traj_psi: np.ndarray,
    outpath: str,
    line_locations: list[int]
) -> None:
    """
    Plot dihedral angle distributions and chain-averaged statistics.

    First panel compares histograms of phi/psi for reference vs simulation.
    Second panel overlays per-chain mean±std for simulation.

    Parameters
    ----------
    AT_phi : np.ndarray
        Reference phi angles per frame.
    AT_psi : np.ndarray
        Reference psi angles per frame.
    Traj_phi : np.ndarray
        Simulation phi angles per frame.
    Traj_psi : np.ndarray
        Simulation psi angles per frame.
    outpath : str
        Directory for saving figures.
    line_locations : list[int]
        Chain boundary indices.
    """
    # Histogram comparison
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    utils.plot_1d_dihedral(axs[0], [AT_phi, Traj_phi], ['Reference', 'Simulation'], bins=60, degrees=True)
    axs[0].set_title('Dihedral angle phi')
    utils.plot_1d_dihedral(axs[1], [AT_psi, Traj_psi], ['Reference', 'Simulation'], bins=60, degrees=True)
    axs[1].set_title('Dihedral angle psi')
    plt.tight_layout()
    fig.savefig(os.path.join(outpath, 'Dihedrals.png'), dpi=300)
    plt.close(fig)

    # Per-chain mean/std overlay
    Traj_phi_chains = split_into_chains(Traj_phi, line_locations)
    Traj_psi_chains = split_into_chains(Traj_psi, line_locations)
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 4))
    utils.plot_1d_dihedral(axs2[0], [AT_phi], ['Reference'], bins=60, degrees=True)
    utils.plot_1d_dihedral_mean_std(axs2[0], [Traj_phi_chains], ['Simulation'], bins=60, degrees=True)
    axs2[0].set_title('Dihedral angle phi')
    utils.plot_1d_dihedral(axs2[1], [AT_psi], ['Reference'], bins=60, degrees=True)
    utils.plot_1d_dihedral_mean_std(axs2[1], [Traj_psi_chains], ['Simulation'], bins=60, degrees=True)
    axs2[1].set_title('Dihedral angle psi')
    plt.tight_layout()
    fig2.savefig(os.path.join(outpath, 'Dihedrals_mean_std.png'), dpi=300)
    plt.close(fig2)


def plot_ramachandran(
    AT_phi: np.ndarray,
    AT_psi: np.ndarray,
    Traj_phi: np.ndarray,
    Traj_psi: np.ndarray,
    kT: float,
    outpath: str
) -> None:
    """
    Plot free-energy surfaces (Ramachandran plots) for phi vs psi.

    Generates side-by-side free-energy contour plots for reference and simulation
    on the same phi-psi grid, colored by kcal/mol.

    Parameters
    ----------
    AT_phi : np.ndarray
        Reference phi angles.
    AT_psi : np.ndarray
        Reference psi angles.
    Traj_phi : np.ndarray
        Simulation phi angles.
    Traj_psi : np.ndarray
        Simulation psi angles.
    kT : float
        Thermal energy in internal units (e.g. 300*kb).
    outpath : str
        Directory to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1, c1 = utils.plot_histogram_free_energy(ax1, AT_phi, AT_psi, kT, degrees=True, ylabel=True)
    ax1.set_title('Reference')
    plt.colorbar(c1, ax=ax1, label='Free energy [kcal/mol]')
    ax2, c2 = utils.plot_histogram_free_energy(ax2, Traj_phi, Traj_psi, kT, degrees=True)
    ax2.set_title('Simulation')
    plt.colorbar(c2, ax=ax2, label='Free energy [kcal/mol]')
    plt.tight_layout()
    fig.savefig(os.path.join(outpath, 'Ramachandran.png'), dpi=300)
    plt.close(fig)


def plot_energy_and_kT(
    aux: dict,
    line_locations: list[int],
    outpath: str
) -> None:
    """
    Plot energy and temperature time-series and overlay chains.

    For each available key in aux ('epot', 'kT', 'etot', 'Temperature'),
    creates two plots: a standard time series and an overlaid chains plot.
    Highlights any chains that "explode" (values >10000).

    Parameters
    ----------
    aux : dict
        Dictionary containing arrays for 'epot', 'kT', 'etot', etc.
    line_locations : list[int]
        Indices delineating chain boundaries.
    outpath : str
        Directory to save plots.
    """
    mapping = [
        ('Epot', aux.get('epot'), utils.plot_energy),
        ('kT', aux.get('kT'), utils.plot_kT),
        ('Etotal', aux.get('etot'), utils.plot_kT),
    ]
    for label, data, plot_fn in mapping:
        if data is None:
            continue
        # Standard time-series
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_fn(ax, data)
        add_chain_lines(ax, line_locations)
        ax.set_title(label)
        plt.tight_layout()
        fig.savefig(os.path.join(outpath, f'{label}.png'), dpi=300)
        plt.close(fig)
        # Overlaid chains
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        boundaries = [0] + list(line_locations) + [len(data)]
        exploded = 0
        for i in range(len(boundaries)-1):
            seg = np.array(data[boundaries[i]:boundaries[i+1]])
            if np.any(seg > 10000):
                exploded += 1
            mask = seg <= 10000
            if mask.any():
                ax2.plot(np.where(mask)[0], seg[mask], alpha=0.7, label=f'Chain {i+1}')
        if exploded:
            ax2.text(0.02, 0.98, f'Chains exploded: {exploded}', transform=ax2.transAxes,
                     color='red', fontweight='bold', verticalalignment='top')
        ax2.set_title(f'{label} - Overlaid chains')
        ax2.set_xlabel('Time step (0.5 ps)')
        ax2.set_ylabel(label)
        plt.tight_layout()
        fig2.savefig(os.path.join(outpath, f'{label}_overlaid.png'), dpi=300)
        plt.close(fig2)

def vis_ala2(
    traj_path: str,
    box: np.ndarray,
    ref_coords: np.ndarray,
    config: dict,
    mapping_change: bool = False,
    name: str = 'Simulation',
    cg_map: str = 'heavyOnly'
) -> None:
    """
    Visualize Ala2 CG trajectory.

    Similar to vis_ala2, but accepts explicit box and reference coords
    and uses specific CG mapping definitions.

    Parameters
    ----------
    traj_path : str
        Path to trajectory directory.
    box : np.ndarray
        Simulation box dimensions.
    ref_coords : np.ndarray
        Reference coordinates array.
    config : dict
        Trajectory configuration parameters.
    name : str, optional
        Label for plots, by default 'Simulation'.
    cg_map : str, optional
        CG mapping key, by default 'heavyOnly'.
    """
    print(f"Visualizing {name} trajectory at {traj_path}")
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)
    disp_fn, _ = periodic_displacement(box, True)
    
    if mapping_change:
        mapping = {'heavyOnly': ([1, 3, 4, 5], [3, 4, 5, 8], [(1,3),(3,4),(4,5)])}
    else:
        mapping = {'heavyOnly': ([1, 3, 4, 6], [3, 4, 6, 8], [(1,3),(3,4),(4,6)])}
    
    # Dihedrals 
    phi_indices, psi_indices, pairs = mapping[cg_map]
    traj_coords, aux = load_trajectory(traj_path)
    ala2_dihedral_fn = utils.init_dihedral_fn(disp_fn, [phi_indices, psi_indices])
    AT_phi, AT_psi = ala2_dihedral_fn(ref_coords)
    Traj_phi, Traj_psi = ala2_dihedral_fn(traj_coords)
    plot_dihedrals(AT_phi, AT_psi, Traj_phi, Traj_psi, outpath, line_locs)
    plot_ramachandran(AT_phi, AT_psi, Traj_phi, Traj_psi, config['kT'], outpath)

    # Atom distances
    AT_dists = [utils.compute_atom_distance(ref_coords, i, j, disp_fn) for i, j in pairs]
    Traj_dists = [utils.compute_atom_distance(traj_coords, i, j, disp_fn) for i, j in pairs]
    plot_dist_series(pairs, AT_dists, Traj_dists, outpath, name, line_locs)
    
    # Energy
    plot_energy_and_kT(aux, line_locs, outpath)


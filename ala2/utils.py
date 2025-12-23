import numpy as onp
from jax import numpy as jnp
import jax
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from jax import vmap
from cycler import cycler
from chemtrain import quantity
from jax_md_mod import custom_quantity
from typing import Callable, Union
import numpy.typing as npt


def compute_angle(coords: jnp.ndarray, idcs: list[int]) -> jnp.ndarray:
    """
    Compute bond angles for every frame of a trajectory.
    
    Parameters
    ----------
    coords : jnp.ndarray
        Trajectory coordinates with shape (n_frames, n_atoms, 3)
    idcs : list[int]
        Three atom indices [i, j, k] where j is the central atom
        
    Returns
    -------
    jnp.ndarray
        Array of bond angles in radians with shape (n_frames,)
    """
    i0, i1, i2 = idcs

    @jax.jit
    def angle_of_frame(frame: jnp.ndarray) -> jnp.ndarray:
        p0 = frame[i0]
        p1 = frame[i1]
        p2 = frame[i2]
        return calculate_angle(p0, p1, p2)

    # Vectorize over frames
    return jax.vmap(angle_of_frame)(coords)


def calculate_angle(p0: jnp.ndarray, p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the bond angle between three points p0-p1-p2.
    
    Parameters
    ----------
    p0 : jnp.ndarray
        Position vector of the first atom with shape (3,)
    p1 : jnp.ndarray
        Position vector of the central atom with shape (3,)
    p2 : jnp.ndarray
        Position vector of the third atom with shape (3,)
        
    Returns
    -------
    jnp.ndarray
        Bond angle in radians (scalar)
    """
    # Vectors from central atom to the other two atoms
    v1 = p0 - p1  # Vector from p1 to p0
    v2 = p2 - p1  # Vector from p1 to p2
    
    # Normalize vectors
    v1_norm = jnp.linalg.norm(v1)
    v2_norm = jnp.linalg.norm(v2)
    
    # Compute cosine of angle using dot product
    cos_angle = jnp.dot(v1, v2) / (v1_norm * v2_norm)
    
    # Clamp to avoid numerical issues with arccos
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    
    # Calculate angle in radians
    angle = jnp.arccos(cos_angle)
    
    return angle


def init_dihedral_fn(displacement_fn: Callable, idcs: list[int]) -> Callable:
    """
    Initialize a function to compute dihedral angles from trajectory positions.
    
    Parameters
    ----------
    displacement_fn : Callable
        Function to compute displacement vectors between atoms
    idcs : list[int]
        Four atom indices defining the dihedral angle
        
    Returns
    -------
    Callable
        Function that takes positions and returns dihedral angles
    """
    idcs = jnp.array(idcs)
    
    def postprocess_fn(positions: jnp.ndarray) -> jnp.ndarray:
        batched_dihedrals = jax.vmap(
            custom_quantity.dihedral_displacement, (0, None, None)
        )
        dihedral_angles = batched_dihedrals(positions, displacement_fn, idcs)
        return dihedral_angles.T
    return postprocess_fn


def init_angle_fn(displacement_fn: Callable, idcs: list[int]) -> Callable:
    """
    Initialize a function to compute bond angles from trajectory positions.
    
    Parameters
    ----------
    displacement_fn : Callable
        Function to compute displacement vectors between atoms
    idcs : list[int]
        Three atom indices defining the bond angle
        
    Returns
    -------
    Callable
        Function that takes positions and returns bond angles
    """
    idcs = jnp.array(idcs)
    
    def postprocess_fn(positions: jnp.ndarray) -> jnp.ndarray:
        batched_angles = jax.vmap(
            custom_quantity.angular_displacement, (0, None, None)
        )
        dihedral_angles = batched_angles(positions, displacement_fn, idcs)
        return dihedral_angles.T
    return postprocess_fn


def compute_atom_distance(
    coords: jnp.ndarray,
    idx1: int,
    idx2: int,
    displacement_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
) -> jnp.ndarray:
    """
    Compute the PBC-aware distance between two atoms over all trajectory frames.
    
    Parameters
    ----------
    coords : jnp.ndarray
        Trajectory coordinates with shape (n_frames, n_atoms, 3)
    idx1 : int
        Index of the first atom
    idx2 : int
        Index of the second atom
    displacement_fn : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        Function that computes PBC-corrected displacement between two position vectors
        
    Returns
    -------
    jnp.ndarray
        Array of scalar distances with shape (n_frames,)
    """
    # Slice out the trajectories of atom idx1 and idx2: each is [n_frames, 3]
    r1 = coords[:, idx1, :]
    r2 = coords[:, idx2, :]

    # Compute the PBC‐corrected displacement for each frame
    # displacement_fn is assumed to take two [3]-vectors → [3]-vector
    disp = vmap(displacement_fn)(r1, r2)   # → [n_frames, 3]
    # Finally, L2‐norm each displacement vector → [n_frames]
    distances = jnp.linalg.norm(disp, axis=-1)

    return distances


def calculate_dihedral(p0: jnp.ndarray, p1: jnp.ndarray, p2: jnp.ndarray, p3: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate dihedral angle between four points in degrees.
    
    Parameters
    ----------
    p0 : jnp.ndarray
        Position vector of the first atom with shape (3,)
    p1 : jnp.ndarray
        Position vector of the second atom with shape (3,)
    p2 : jnp.ndarray
        Position vector of the third atom with shape (3,)
    p3 : jnp.ndarray
        Position vector of the fourth atom with shape (3,)
        
    Returns
    -------
    jnp.ndarray
        Dihedral angle in degrees (scalar)
    """
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1 so it does not influence magnitude of vector rejections
    b1 /= jnp.linalg.norm(b1)

    # Compute cross products
    v = b0 - jnp.dot(b0, b1) * b1
    w = b2 - jnp.dot(b2, b1) * b1

    x = jnp.dot(v, w)
    y = jnp.dot(jnp.cross(b1, v), w)

    return jnp.degrees(jnp.arctan2(y, x))


def plot_1d_dihedral(
    ax: Axes,
    angles: list[jnp.ndarray],
    labels: list[str],
    bins: int = 120,
    degrees: bool = True,
    xlabel: str = '$\phi$ in deg',
    ylabel: bool = True
) -> Axes:
    """
    Plot 1D histogram splines for dihedral angles.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on
    angles : list[jnp.ndarray]
        list of angle arrays, one for each model/dataset
    labels : list[str]
        list of labels for each angle dataset
    bins : int, optional
        Number of histogram bins (default: 120)
    degrees : bool, optional
        Whether angles are in degrees (default: True)
    xlabel : str, optional
        Label for x-axis (default: '$\phi$ in deg')
    ylabel : bool, optional
        Whether to add y-axis label (default: True)
        
    Returns
    -------
    Axes
        The modified matplotlib axes object
    """
    color = ['#368274', '#0C7CBA', '#C92D39', '#FFB347', '#7851A9', '#66CC99', '#FF6B6B', '#4A90E2', '#50514F', '#F4A261']
    line = ['-', '-', '-']
    
    n_models = len(angles)
    for i in range(n_models):
        if degrees:
            angles_conv = angles[i]
            hist_range = [-180, 180]
        else:
            angles_conv = onp.rad2deg(angles[i])
            hist_range = [-onp.pi, onp.pi]

        # Compute the histogram
        hist, x_bins = jnp.histogram(angles_conv, bins=bins, density=True, range=hist_range)
        width = x_bins[1] - x_bins[0]
        bin_center = x_bins + width / 2
        
        ax.plot(
            bin_center[:-1], hist, label=labels[i], color=color[i],
            # linestyle=line[i], 
            linewidth=2.0
        )

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel('Density')
    
    ax.legend()  # Add legend to the plot
    return ax


def plot_1d_dihedral_mean_std(
    ax: Axes,
    angles: list[npt.NDArray],
    labels: list[str],
    bins: int = 120,
    degrees: bool = True,
    xlabel: str = '$\phi$ in deg',
    ylabel: bool = True
) -> Axes:
    """
    Plot 1D histogram with mean and standard deviation as shaded area for each dihedral angle set.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on
    angles : list[npt.NDArray]
        list of angle arrays, one for each model/dataset
    labels : list[str]
        list of labels for each angle dataset
    bins : int, optional
        Number of histogram bins (default: 120)
    degrees : bool, optional
        Whether angles are in degrees (default: True)
    xlabel : str, optional
        Label for x-axis (default: '$\phi$ in deg')
    ylabel : bool, optional
        Whether to add y-axis label (default: True)
        
    Returns
    -------
    Axes
        The modified matplotlib axes object
    """
    color = ['#0C7CBA', '#C92D39', '#FFB347', '#7851A9', '#66CC99', '#FF6B6B', '#4A90E2', '#50514F', '#F4A261']
    n_models = len(angles)
    for i in range(n_models):
        data = onp.array(angles[i])
        if degrees:
            data_conv = data
            hist_range = [-180, 180]
        else:
            data_conv = onp.rad2deg(data)
            hist_range = [-onp.pi, onp.pi]

        # Compute histogram for each sample, then mean/std over samples
        # If data_conv is 2D: [n_samples, n_points]
        if data_conv.ndim == 2:
            hists = []
            for sample in data_conv:
                hist, x_bins = onp.histogram(sample, bins=bins, density=True, range=hist_range)
                hists.append(hist)
            hists = onp.stack(hists)
            hist_mean = hists.mean(axis=0)
            hist_std = hists.std(axis=0)
        else:
            hist_mean, x_bins = onp.histogram(data_conv, bins=bins, density=True, range=hist_range)
            hist_std = onp.zeros_like(hist_mean)

        width = x_bins[1] - x_bins[0]
        bin_center = x_bins[:-1] + width / 2

        ax.plot(bin_center, hist_mean, label=labels[i], color=color[i], linewidth=2.0)
        ax.fill_between(bin_center, hist_mean - hist_std, hist_mean + hist_std, color=color[i], alpha=0.2)

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel('Density')
    ax.legend()
    return ax


def plot_histogram_free_energy(
    ax: Axes,
    phi: jnp.ndarray,
    psi: jnp.ndarray,
    kbt: float,
    degrees: bool = True,
    ylabel: bool = False,
    title: str = ""
) -> tuple[Axes, QuadMesh]:
    """
    Plot 2D free energy histogram for alanine from the dihedral angles.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on
    phi : jnp.ndarray
        Phi dihedral angles
    psi : jnp.ndarray
        Psi dihedral angles
    kbt : float
        Temperature in energy units (kT)
    degrees : bool, optional
        Whether angles are in degrees (default: True)
    ylabel : bool, optional
        Whether to add y-axis label (default: False)
    title : str, optional
        Title for the plot (default: "")
        
    Returns
    -------
    tuple[Axes, QuadMesh]
        The modified matplotlib axes object and the colormap mesh
    """
    cmap = plt.get_cmap('viridis')

    if degrees:
        phi = jnp.deg2rad(phi)
        psi = jnp.deg2rad(psi)

    h, x_edges, y_edges = jnp.histogram2d(phi, psi, bins=60, density=True)

    h = jnp.log(h) * -(kbt / 4.184)
    x, y = onp.meshgrid(x_edges, y_edges)

    cax = ax.pcolormesh(x, y, h.T, cmap=cmap, vmax=5.25)
    ax.set_xlabel('$\phi$ [rad]')
    if ylabel:
        ax.set_ylabel('$\psi$ [rad]')
    ax.set_title(title)
    
    return ax, cax


def plot_atom_distance(
    ax: Axes,
    distances: jnp.ndarray | list[jnp.ndarray],
    labels: list[str] | None = None,
    bins: int = 60,
    xlabel: str = 'Distance',
    ylabel: str = 'Frequency'
) -> Axes:
    """
    Plot histogram of atom distances.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on
    distances : jnp.ndarray | list[jnp.ndarray]
        Distance data - single array or list of arrays for multiple models
    labels : list[str] | None, optional
        list of labels for each set of distances (default: None)
    bins : int, optional
        Number of bins for the histogram (default: 60)
    xlabel : str, optional
        Label for the x-axis (default: 'Distance')
    ylabel : str, optional
        Label for the y-axis (default: 'Frequency')
        
    Returns
    -------
    Axes
        The modified matplotlib axes object
    """
    color = ['#368274', '#0C7CBA', '#C92D39', 'k']
    line = ['-', '-', '-', '--']

    if isinstance(distances, (list, tuple)) and hasattr(distances[0], '__len__'):
        n_models = len(distances)
        for i in range(n_models):
            ax.hist(distances[i], bins=bins, alpha=0.6, label=labels[i] if labels else None,
                    color=color[i % len(color)], histtype='step', linewidth=2.0, linestyle=line[i % len(line)])
    else:
        ax.hist(distances, bins=bins, alpha=0.6, color=color[0], histtype='step', linewidth=2.0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels:
        ax.legend()
    return ax


def compare_atom_distances(
    AT_distances: list[jnp.ndarray],
    Traj_distances: list[jnp.ndarray],
    dist_labels: list[str],
    outpath: str,
    name: str,
    at_label: str = "Reference",
    traj_label: str = "Simulation",
    bins: int = 60,
    at_color: str = "#368274",
    traj_color: str = "#C92D39",
    xlabel: str = "Distance",
    ylabel: str = "Normalized frequency"
) -> str:
    """
    Plot reference vs simulation atom-distance histograms side by side.
    
    Parameters
    ----------
    AT_distances : list[jnp.ndarray]
        list of 1D arrays of reference distances
    Traj_distances : list[jnp.ndarray]
        list of 1D arrays of simulation distances
    dist_labels : list[str]
        list of titles for each subplot (same length as distances)
    outpath : str
        Directory to save the figure in
    name : str
        Basename for the output file
    at_label : str, optional
        Legend label for reference data (default: "Reference")
    traj_label : str, optional
        Legend label for simulation data (default: "Simulation")
    bins : int, optional
        Number of bins (default: 60)
    at_color : str, optional
        Color for reference histograms (default: "#368274")
    traj_color : str, optional
        Color for simulation histograms (default: "#C92D39")
    xlabel : str, optional
        X-axis label (default: "Distance")
    ylabel : str, optional
        Y-axis label (default: "Normalized frequency")
        
    Returns
    -------
    str
        Full path to the saved figure file
    """
    n = len(dist_labels)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 4), sharey=True)

    for i, title in enumerate(dist_labels):
        ax = axes[i] if n > 1 else axes
        # AT
        ax.hist(AT_distances[i],
                bins=bins,
                density=True,
                histtype='step',
                linewidth=2.0,
                linestyle='-',
                color=at_color,
                label=at_label)
        # Simulation
        ax.hist(Traj_distances[i],
                bins=bins,
                density=True,
                histtype='step',
                linewidth=2.0,
                linestyle='-',
                color=traj_color,
                label=traj_label)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)
        ax.legend(frameon=False)

    plt.tight_layout()
    fname = f"{outpath}/Atom_distances_{name}_vs_Reference.png"
    plt.savefig(fname, dpi=300)
    plt.close(fig)
    return fname


def plot_energy(
    ax: Axes,
    energy: Union[jnp.ndarray, list[jnp.ndarray]],
    labels: list[str] = None,
    xlabel: str = 'Time',
    ylabel: str = 'Energy [kJ/mol]'
) -> Axes:
    """
    Plot energy values over time.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on
    energy : Union[jnp.ndarray, list[jnp.ndarray]]
        Energy data - single array or list of arrays for multiple models
    labels : list[str]], optional
        list of labels for each set of energy values (default: None)
    xlabel : str, optional
        Label for the x-axis (default: 'Time')
    ylabel : str, optional
        Label for the y-axis (default: 'Energy [kJ/mol]')
        
    Returns
    -------
    Axes
        The modified matplotlib axes object
    """
    color = ['#368274', '#0C7CBA', '#C92D39', 'k']
    line = ['-', '-', '-', '--']

    if isinstance(energy, (list, tuple)) and hasattr(energy[0], '__len__'):
        n_models = len(energy)
        for i in range(n_models):
            ax.plot(
                range(len(energy[i])),
                energy[i],
                label=labels[i] if labels else None,
                color=color[i % len(color)],
                linestyle=line[i % len(line)],
                linewidth=2.0
            )
    else:
        ax.plot(
            range(len(energy)),
            energy,
            color=color[0],
            linewidth=2.0
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels:
        ax.legend()
    return ax


def plot_kT(
    ax: Axes,
    kT: Union[jnp.ndarray, list[jnp.ndarray]],
    labels: list[str] = None,
    xlabel: str = 'Time',
    ylabel: str = 'kT [kJ/mol]'
) -> Axes:
    """
    Plot kT values over time.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on
    kT : Union[jnp.ndarray, list[jnp.ndarray]]
        kT data - single array or list of arrays for multiple models
    labels : list[str]], optional
        list of labels for each set of kT values (default: None)
    xlabel : str, optional
        Label for the x-axis (default: 'Time')
    ylabel : str, optional
        Label for the y-axis (default: 'kT [kJ/mol]')
        
    Returns
    -------
    Axes
        The modified matplotlib axes object
    """
    color = ['#368274', '#0C7CBA', '#C92D39', 'k']
    line = ['-', '-', '-', '--']

    if isinstance(kT, (list, tuple)) and hasattr(kT[0], '__len__'):
        n_models = len(kT)
        for i in range(n_models):
            ax.plot(
                range(len(kT[i])),
                kT[i],
                label=labels[i] if labels else None,
                color=color[i % len(color)],
                linestyle=line[i % len(line)],
                linewidth=2.0
            )
    else:
        ax.plot(
            range(len(kT)),
            kT,
            color=color[0],
            linewidth=2.0
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels:
        ax.legend()
    return ax


def plot_T(
    ax: Axes,
    kT: Union[jnp.ndarray, list[jnp.ndarray]],
    labels: list[str] = None,
    xlabel: str = 'Time',
    ylabel: str = 'T [K]'
) -> Axes:
    """
    Plot temperature values over time by converting from kT.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on
    kT : Union[jnp.ndarray, list[jnp.ndarray]]
        kT data - single array or list of arrays for multiple models
    labels : list[str], optional
        list of labels for each set of kT values (default: None)
    xlabel : str, optional
        Label for the x-axis (default: 'Time')
    ylabel : str, optional
        Label for the y-axis (default: 'T [K]')
        
    Returns
    -------
    Axes
        The modified matplotlib axes object
    """
    color = ['#368274', '#0C7CBA', '#C92D39', 'k']
    line = ['-', '-', '-', '--']

    if isinstance(kT, (list, tuple)) and hasattr(kT[0], '__len__'):
        n_models = len(kT)
        for i in range(n_models):
            ax.plot(
                range(len(kT[i])),
                kT[i]/quantity.kb,
                label=labels[i] if labels else None,
                color=color[i % len(color)],
                linestyle=line[i % len(line)],
                linewidth=2.0
            )
    else:
        ax.plot(
            range(len(kT)),
            kT/quantity.kb,
            color=color[0],
            linewidth=2.0
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels:
        ax.legend()
    return ax


def plot_predictions(
    predictions: dict,
    reference_data: dict,
    out_dir: str,
    name: str
) -> None:
    """
    Plot force predictions vs reference data with scatter plot and compute MAE.
    
    Parameters
    ----------
    predictions : dict
        Dictionary containing predicted values with 'F' key for forces
    reference_data : dict
        Dictionary containing reference values with 'F' key for forces
    out_dir : str
        Output directory to save the figure
    name : str
        Name for the output file
    """
    # Simplifies comparison: convert units
    scale_energy = 96.485  # [eV] -> [kJ/mol]
    scale_pos = 0.1        # [Å] -> [nm]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5), layout="constrained")
    fig.suptitle("Predictions")

    # Reshape forces and scale units
    pred_F = predictions['F'].reshape(-1, 3) / scale_energy * scale_pos
    ref_F  = reference_data['F'].reshape(-1, 3)       / scale_energy * scale_pos
    
    # Ensure pred_F has same number of entries as ref_F by dropping extra entries
    if len(pred_F) > len(ref_F):
        pred_F = pred_F[:len(ref_F)]
    elif len(ref_F) > len(pred_F):
        ref_F = ref_F[:len(pred_F)]
        
    # Verify shapes match
    assert pred_F.shape == ref_F.shape, f"Shape mismatch: pred_F {pred_F.shape}, ref_F {ref_F.shape}"

    # Compute MAE
    mae = onp.mean(onp.abs(pred_F - ref_F))
    ax.set_title(f"Force (MAE: {mae * 1000:.1f} meV/A)")

    # 45-degree reference line
    ax.axline((0, 0), slope=1, color="black", linestyle=(0, (3, 5, 1, 5)), linewidth=1)

    # Scatter plot
    ax.set_prop_cycle(cycler(color=plt.get_cmap('tab20c').colors))
    ax.scatter(ref_F.ravel(), pred_F.ravel(), s=5, edgecolors='none', alpha=0.2)

    ax.set_xlabel("Ref. F [eV/A]")
    ax.set_ylabel("Pred. F [eV/A]")
    ax.legend().remove()  # no legend needed

    # Save figure
    fig.savefig(f"{out_dir}/{name}.png", bbox_inches="tight", dpi=1200)


def calc_mse_dihedrals(
    phi_ref: jnp.ndarray,
    psi_ref: jnp.ndarray,
    phi_sim: jnp.ndarray,
    psi_sim: jnp.ndarray,
    nbins: int = 60
) -> float:
    """
    Calculate mean squared error between reference and simulation dihedral angle distributions.
    
    Parameters
    ----------
    phi_ref : jnp.ndarray
        Reference phi dihedral angles in degrees
    psi_ref : jnp.ndarray
        Reference psi dihedral angles in degrees
    phi_sim : jnp.ndarray
        Simulation phi dihedral angles in degrees
    psi_sim : jnp.ndarray
        Simulation psi dihedral angles in degrees
    nbins : int, optional
        Number of bins for 2D histogram (default: 60)
        
    Returns
    -------
    float
        Mean squared error between the two 2D density histograms
    """
    # convert to radians
    phi_ref_rad, psi_ref_rad = jnp.deg2rad(phi_ref), jnp.deg2rad(psi_ref)
    phi_sim_rad, psi_sim_rad = jnp.deg2rad(phi_sim), jnp.deg2rad(psi_sim)

    h_ref, _, _ = onp.histogram2d(phi_ref_rad, psi_ref_rad, bins=nbins, density=True)
    h_sim, _, _ = onp.histogram2d(phi_sim_rad, psi_sim_rad, bins=nbins, density=True)

    mse = onp.mean((h_ref - h_sim)**2)
    print('MSE of the phi-psi dihedral density histogram:', mse)
    return mse

def plot_convergence(trainer, out_dir):
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5),
                                        layout="constrained")

    ax1.set_title("Loss")
    ax1.semilogy(trainer.train_losses, label="Training")
    ax1.semilogy(trainer.val_losses, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    fig.savefig(f"{out_dir}/convergence.pdf", bbox_inches="tight")
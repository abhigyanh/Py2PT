import numpy as np
import MDAnalysis as mda
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from numba import njit

@njit
def jit_decompose_velocity_core(positions, velocities, masses, is_linear):
    """
    JIT-accelerated core function to decompose atomic velocities into translational, rotational, and vibrational components.

    Parameters
    ----------
    positions : np.ndarray
        Atomic positions, shape (n_atoms, 3)
    velocities : np.ndarray
        Atomic velocities, shape (n_atoms, 3)
    masses : np.ndarray
        Atomic masses, shape (n_atoms,)
    is_linear : bool
        Whether the molecule is linear (affects principal moments)

    Returns
    -------
    vt : np.ndarray
        Translational velocities, shape (n_atoms, 3)
    rot_vel : np.ndarray
        Rotational velocities, shape (n_atoms, 3)
    vib_vel : np.ndarray
        Vibrational velocities, shape (n_atoms, 3)
    omega : np.ndarray
        Angular velocity vector, shape (3,)
    evl : np.ndarray
        Principal moments of inertia, shape (3,)
    """
    # Calculate the center of mass position and velocity
    n_atoms = positions.shape[0]
    com_pos = np.sum(masses[:, None] * positions, axis=0) / np.sum(masses)
    com_vel = np.sum(masses[:, None] * velocities, axis=0) / np.sum(masses)
    rel_pos = positions - com_pos
    rel_vel = velocities - com_vel
    
    # Assign translational velocity (center of mass velocity) to all atoms
    vt = np.empty((n_atoms, 3), dtype=positions.dtype)
    for i in range(n_atoms):
        vt[i, :] = com_vel
    
    # Calculate angular momentum and inertia tensor
    ang_mom = np.sum(masses[:, None] * np.cross(rel_pos, rel_vel), axis=0)
    inertia = np.zeros((3, 3))
    for i in range(n_atoms):
        m = masses[i]
        x, y, z = rel_pos[i]
        # Diagonal elements
        inertia[0, 0] += m * (y**2 + z**2)  # I_xx
        inertia[1, 1] += m * (x**2 + z**2)  # I_yy
        inertia[2, 2] += m * (x**2 + y**2)  # I_zz
        # Off-diagonal elements (note the minus sign)
        inertia[0, 1] -= m * x * y          # I_xy
        inertia[1, 0] = inertia[0, 1]
        inertia[0, 2] -= m * x * z          # I_xz
        inertia[2, 0] = inertia[0, 2]
        inertia[1, 2] -= m * y * z          # I_yz
        inertia[2, 1] = inertia[1, 2]
    
    # Diagonalize inertia tensor to get principal moments and axes
    evl, evt = np.linalg.eigh(inertia)
    idx = np.argsort(evl)
    evl = evl[idx]
    evt = evt[:, idx]
    if is_linear and n_atoms > 1:
        evl[0] = 0.0  # For linear molecules, set smallest moment to zero
    
    # Calculate angular velocities along principal axes
    pomega = np.zeros(3)
    for i in range(3):
        if evl[i] > 1e-9:
            L_principal_i = np.dot(ang_mom, evt[:, i].copy())
            pomega[i] = L_principal_i / evl[i]
    omega = np.dot(evt, pomega)
    
    # Calculate rotational and vibrational velocities
    rot_vel = np.cross(omega, rel_pos)
    vib_vel = rel_vel - rot_vel
    return vt, rot_vel, vib_vel, omega, evl

def decompose_velocity_for_residue(residue_atoms, is_linear=False):
    """
    Decompose the velocities of a residue's atoms into translational, rotational, and vibrational components.
    Handles special cases for empty or single-atom residues.
    Raises an error if atom velocities are not present.
    """
    n_atoms = len(residue_atoms)
    if n_atoms == 0:
        # No atoms: return empty arrays
        return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(3), np.zeros(3))
    # Check for presence of velocities
    if not hasattr(residue_atoms, 'velocities') or residue_atoms.velocities is None:
        raise ValueError("Atom velocities are not present in the input. Please ensure your trajectory contains velocities.")
    if n_atoms == 1:
        # Single atom: only translational velocity is meaningful
        vt = residue_atoms[0].velocity.copy()[None, :]
        vr = np.zeros((1, 3))
        vv = np.zeros((1, 3))
        omega = np.zeros(3)
        evl = np.zeros(3)
        return vt, vr, vv, omega, evl
    # Extract arrays for JIT-accelerated function
    positions = residue_atoms.positions
    velocities = residue_atoms.velocities
    masses = residue_atoms.masses
    vt, rot_vel, vib_vel, omega, evl = jit_decompose_velocity_core(positions, velocities, masses, is_linear)
    return vt, rot_vel, vib_vel, omega, evl

def _decompose_frame(args):
    """
    Helper function to decompose velocities for a single frame (for multiprocessing).
    Loads the frame, selects atoms, and decomposes velocities for each residue.
    """
    topology, trajectory, selection, is_linear, frame_idx = args
    u = mda.Universe(topology, trajectory)
    ag = u.select_atoms(selection)
    u.trajectory[frame_idx]
    vt_list, vr_list, vv_list = [], [], []
    omega_list = []
    evl_list = []
    for residue in ag.residues:
        vt, vr, vv, omega, evl = decompose_velocity_for_residue(residue.atoms, is_linear=is_linear)
        vt_list.append(vt)
        vr_list.append(vr)
        vv_list.append(vv)
        omega_list.append(omega)
        evl_list.append(evl)
    # Concatenate all atoms' velocities for this frame
    vt_all = np.concatenate(vt_list, axis=0)
    vr_all = np.concatenate(vr_list, axis=0)
    vv_all = np.concatenate(vv_list, axis=0)
    omega_all = np.stack(omega_list, axis=0)  # shape (n_residues, 3)
    evl_all = np.stack(evl_list, axis=0)      # shape (n_residues, 3)
    return vt_all, vr_all, vv_all, omega_all, evl_all

def decompose_trajectory_velocities(atomgroup, is_linear=False, n_workers=4):
    """
    Decompose the velocities of all atoms in a trajectory into translational, rotational, and vibrational components.
    Uses multiprocessing for efficiency on long trajectories.
    Returns arrays of shape (n_frames, n_atoms, 3) for each component, (n_frames, n_residues, 3) for angular velocities, and (n_frames, n_residues, 3) for principal moments.
    """
    universe = atomgroup.universe
    n_frames = len(universe.trajectory)
    n_atoms = len(atomgroup)
    n_residues = len(atomgroup.residues)
    # Preallocate arrays for all frames and atoms
    vt_all = np.zeros((n_frames, n_atoms, 3))
    vr_all = np.zeros((n_frames, n_atoms, 3))
    vv_all = np.zeros((n_frames, n_atoms, 3))
    omega_all = np.zeros((n_frames, n_residues, 3))
    evl_all = np.zeros((n_frames, n_residues, 3))
    topology = universe.filename
    trajectory = universe.trajectory.filename
    # Determine atom selection string
    selection = atomgroup.selections[0] if hasattr(atomgroup, 'selections') else 'all'
    if hasattr(atomgroup, 'string'):
        selection = atomgroup.string
    print(f"Processing trajectory with {n_frames} frames using {n_workers} processes...")
    # Prepare arguments for each frame
    args_list = [(topology, trajectory, selection, is_linear, i) for i in range(n_frames)]
    # Use multiprocessing to process frames in parallel, with safe shutdown on Ctrl+C
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(executor.map(_decompose_frame, args_list), total=n_frames, desc="Processing trajectory"))
    except KeyboardInterrupt:
        print("KeyboardInterrupt: shutting down pool...")
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    # Collect results into output arrays
    for i, (vt, vr, vv, omega, evl) in enumerate(results):
        vt_all[i, :, :] = vt
        vr_all[i, :, :] = vr
        vv_all[i, :, :] = vv
        omega_all[i, :, :] = omega
        evl_all[i, :, :] = evl
    return vt_all, vr_all, vv_all, omega_all, evl_all 
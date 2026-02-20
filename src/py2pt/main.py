# main.py

#==========================================================================
# Imports
#==========================================================================
# Dependency imports
import yaml
import argparse
import sys
import os
import gc
import numpy as np

import MDAnalysis as mda
from MDAnalysis import transformations

# Internal imports
from . import __version__
from .output        import *
from .constants     import *
from .velocity      import *
from .spectrum      import *
from .entropy       import *
from .energy        import *


def main():
    """
    Main function to run the Py2PT workflow from the command line.

    Parses options, loads the trajectory and topology, 
    performs velocity decomposition, computes density of states, and calculates entropy using the 2PT method. 
    Results are printed and saved to output files.
    """
    #==========================================================================
    # Parse command-line arguments
    #==========================================================================
    parser = argparse.ArgumentParser(
        description="Calculate density of states spectra and entropy of a molecular dynamics trajectory using 2PT method."
    )
    parser.add_argument('-c', '--config', type=str, help='Path to the Py2PT config file (default: Py2PT.yml)', default='Py2PT.yml')
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    # Read the config file
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Extract values from the config file
    temperature = config.get('temperature', 300.0)
    volume_config = config.get('volume', None)
    topology_file = config.get('topology', 'topol.tpr')
    trajectory_file = config.get('trajectory', 'traj.trr')
    selection_string = config.get('selection', 'all')
    nproc = config.get('nproc', 1)
    sigma = config.get('sigma', 1)
    FILTERING = config.get('filter', True)
    filter_window = config.get('filter_window', 5)
    CONSTRAINED_BONDS = config.get('constraints', False)
    RENORMALIZE_DOS = config.get('renormalize', True)
    ZERO_CORRECTION = config.get('zero_correction', False)

    # Check if topology and trajectory files exist
    if not os.path.isfile(topology_file):
        print(f"Error: Topology file '{topology_file}' could not be found.")
        sys.exit(1)
    if not os.path.isfile(trajectory_file):
        print(f"Error: Trajectory file '{trajectory_file}' could not be found.")
        sys.exit(1)

    # Value checker
    if not isinstance(nproc, int) or nproc < 1:
        print(f"Error: nproc must must be a positive integer, not {nproc}.")
        sys.exit(1)
    if not isinstance(sigma, int) or sigma <= 0:
        print(f"Error: sigma must be a positive integer, not {sigma}.")
        sys.exit(1)
    if FILTERING not in [True, False]:
        print("Error: FILTERING must be either True or False.")
        sys.exit(1)
    if not isinstance(filter_window, int) or filter_window <= 3:
        print(f"Error: filter_window must be a positive integer > 3 (default = 5).")
        sys.exit(1)
    if CONSTRAINED_BONDS not in [True, False]:
        print("Error: CONSTRAINED_BONDS must be either true or false.")
        sys.exit(1)
    if RENORMALIZE_DOS not in [False, 'false', 'before_f', 'after_f']:
        print("Error: RENORMALIZE_DOS must be either 'false', 'before_f', or 'after_f'.")
        sys.exit(1)
    if ZERO_CORRECTION not in [True, False]:
        print("Error: ZERO_CORRECTION must be either true or false.")
        sys.exit(1)
    if volume_config is not None:
        try:
            volume_config = float(volume_config)
        except (TypeError, ValueError):
            print(f"Error: volume must be a number in Å³, not {volume_config!r}.")
            sys.exit(1)
        if volume_config <= 0:
            print(f"Error: volume must be positive (Å³), not {volume_config}.")
            sys.exit(1)


    #==========================================================================
    # Opening print
    #==========================================================================

    print("\n" + "="*80)
    print(f"Running Py2PT (version {__version__})")
    print(f"Loading Topology: {topology_file}")
    print(f"Loading Trajectory: {trajectory_file}")
    print(f"nproc: {nproc}")

    #==========================================================================
    # Load and prepare trajectory
    #==========================================================================
        # Load the simulation universe
    try:
        print("Loading trajectory...")
        u = mda.Universe(topology_file, trajectory_file)
        
        print("Loading trajectory data into memory (this may take a while)...")
        # Get frame count and atom count for array pre-allocation
        n_frames = len(u.trajectory)
        n_atoms = u.trajectory[0].n_atoms  # Get atom count from first frame
        print(f"Preparing arrays for {n_frames} frames and {n_atoms} atoms...")
        
        # Pre-allocate numpy arrays instead of using lists
        coordinates = np.empty((n_frames, n_atoms, 3), dtype=np.float32)
        velocities = np.empty((n_frames, n_atoms, 3), dtype=np.float32)
        dimensions = np.empty((n_frames, 6), dtype=np.float32)
        
        # Load data frame by frame with verification
        for i, ts in enumerate(tqdm(u.trajectory, desc="Reading frames")):
            # Force data loading with .copy()
            try:
                frame_pos = ts.positions.copy()
                frame_vel = ts.velocities.copy()
                frame_dim = ts.dimensions.copy()
                coordinates[i] = frame_pos
                velocities[i] = frame_vel
                dimensions[i] = frame_dim  
            except Exception as e:
                print(f"\nError loading frame {i}: {str(e)}", flush=True)
                raise
        # Create new universe with pre-allocated arrays
        print("Creating in-memory trajectory...")
        u = mda.Universe(
            topology_file,
            coordinates,
            velocities=velocities,
            dimensions=dimensions,
            in_memory=True,
            dt=u.trajectory.dt
        )
        print("Trajectory loading complete")
        
    except Exception as e:
        print("Could not load Universe.")
        print("Error:", e)
        sys.exit(1)

    # Selection of atoms by string
    ag = u.select_atoms(selection_string)

    #==========================================================================
    # System properties
    #==========================================================================
    # Basic properties
    dt = u.trajectory.dt
    masses = ag.atoms.masses
    total_mass = np.sum(masses)  # in amu
    n_atoms = len(ag)
    n_molecules = len(ag.residues)
    atoms_per_molecule = int(n_atoms/n_molecules)
    molecule_mass = total_mass/n_molecules
    # Volume calculation
    if volume_config is None:
        volume = mda.lib.mdamath.box_volume(u.dimensions)  # Å³
        volume_source = "calculated from trajectory"
    else:
        volume = volume_config  # Å³
        volume_source = "user-defined"
    mass_density = (total_mass * amu) / (volume * angstrom**3)
    
    # Print system information
    print("\n" + "="*80)
    print(" "*30 + "TRAJECTORY INFORMATION")
    print("="*80)
    print(f"{'Time step:':<25} {dt:.3f} ps")
    print(f"{'Selection string:':<25} {selection_string}")
    print(f"{'Number of atoms:':<25} {n_atoms}")
    print(f"{'Number of molecules:':<25} {n_molecules}")
    print(f"{'Atoms per molecule:':<25} {atoms_per_molecule}")
    print(f"{'Mass per molecule:':<25} {molecule_mass:.2f} amu")
    print(f"{'Simulation volume:':<25} {volume:.2f} Å³ ({volume_source})")
    print(f"{'Mass density:':<25} {mass_density/1000:.3e} g/cm³")
    print(f"{'Temperature:':<25} {temperature} K")
    # Print user-defined options
    print("\n" + "="*80)
    print(" "*30 + "USER DEFINED OPTIONS")
    print("="*80)
    print(f"{'FFT filtering:':<25} {FILTERING}")
    print(f"{'FFT filter window:':<25} {filter_window}")
    print(f"{'Constrained bonds:':<25} {CONSTRAINED_BONDS}")
    print(f"{'Renomalize DOS:':<25} {RENORMALIZE_DOS}")

    #==========================================================================
    # Velocity decomposition and power spectra
    #==========================================================================
    # Set is_linear as appropriate for your system (False for most molecules)
    is_linear = False

    # Pre-load trajectory data
    print("\n" + "="*80)
    print(" "*15 + "VELOCITY DECOMPOSITION & DENSITY OF STATES SPECTRA")
    print("="*80)

    # Initialize arrays to store all trajectory data
    n_frames = len(u.trajectory)
    residue_data = []
    
    # Extract data for each residue from in-memory trajectory
    for residue in tqdm(ag.residues, desc="Splicing trajectory by selection group"):
        atom_indices = residue.atoms.indices
        # Use direct array slicing for all frames
        pos = u.trajectory.coordinate_array[:, atom_indices, :].copy()
        vel = u.trajectory.velocity_array[:, atom_indices, :].copy()
        residue_data.append((
            pos,                            # positions array
            vel,                            # velocities array
            residue.atoms.masses,           # masses array
            (residue.resname, residue.resid)  # residue info
        ))
    
    # Clean up the large arrays and universe to free memory
    del coordinates, velocities, dimensions, u
    gc.collect()
    
    # Calculate all DOS components in parallel over molecules
    print("Performing velocity decomposition and DOS calculation...")
    freqs, tDOS, vDOS, rDOS, I_lk = decompose_trajectory_velocities(
        residue_data,
        is_linear=is_linear, 
        n_workers=nproc,
        dt=dt,
        temperature=temperature,
        FILTERING=FILTERING,
        filter_window=filter_window
    )
    print("Finished velocity decomposition and DOS calculation.")
    
    # Apply zero correction if requested
    if ZERO_CORRECTION:
        print("INFO: Performing zero correction on DOS spectra")
        high_freq_mask = freqs > 120
        tDOS -= np.mean(tDOS[high_freq_mask])
        rDOS -= np.mean(rDOS[high_freq_mask])
        vDOS -= np.mean(vDOS[high_freq_mask])

    #==========================================================================
    # Two-Phase Thermodynamics (2PT) Analysis
    #==========================================================================
    print("\n" + "="*80)
    print(" "*25 + "TWO-PHASE THERMODYNAMICS ANALYSIS")
    print("="*80)

    # Principal moments of inertia calculation
    # Use moments averaged over frames and residues from velocity decomposition
    I_1, I_2, I_3 = np.mean(I_lk, axis=0)
    print(f"Averaged principal moments of inertia: I_1 = {I_1:.4f}, I_2 = {I_2:.4f}, I_3 = {I_3:.4f} amu·Å²")
    plot_histogram(I_lk, filename="principal_moments_of_inertia.png")

    # Print DOS calculation parameters
    print(f"\nCalculated DOS spectra at T = {temperature} K")
    if FILTERING:
        print("INFO: Using Blackman windowing before FFT")
        print("INFO: Using Savitsky-Golay filtering after FFT")

    # Integrate the power spectra
    vt_integral = np.trapz(tDOS, freqs)
    vr_integral = np.trapz(rDOS, freqs)
    vv_integral = np.trapz(vDOS, freqs)

    # Print DOS spectra results
    print("\nIntegrated DOS Spectra:")
    print("-"*40)
    print(f"{'Translational:':<20} {vt_integral:>12.4f}")
    print(f"{'Rotational:':<20} {vr_integral:>12.4f}")
    print(f"{'Vibrational:':<20} {vv_integral:>12.4f}")
    print("-"*40)
    print(f"{'Total:':<20} {(vt_integral + vr_integral + vv_integral):>12.4f}")
    
    # Renormalize DOS BEFORE decomposition
    if RENORMALIZE_DOS == 'before_f':
        # DOS re-normalization
        print(f"\nINFO: Re-normalizing DoS by scaling factor to match theoretical DOF")
        print(f"-- Scale the DOS to set the integral equal to 'real' degrees of freedom")
        print(f"-- 3*nMol (for translation, rotation) and 3*nMol*(nAtomPerMol-2) for vibration")
        print(f"-- This can lead to better convergence of entropy for short MD trajectories")
        print(f"-- Reference: M.A. Caro, T. Laurila, and O. Lopez-Acevedo, The Journal of Chemical Physics 145, (2016).")

        tDOS *= 3*n_molecules/np.trapz(tDOS, freqs)
        rDOS *= 3*n_molecules/np.trapz(rDOS, freqs)
        if CONSTRAINED_BONDS == False:
            vDOS *= 3*n_molecules*(atoms_per_molecule-2)/np.trapz(vDOS, freqs)

    # Calculate the total DOS
    DOS_total = tDOS+rDOS+vDOS

    # Decompose density of states using 2PT method
    print("\nDecomposing density of states...")
    print("-"*50)
    Delta_tr, f_tr, s0_tr, DOS_tr_g, DOS_tr_s = decompose_translational_dos(
        freqs, tDOS, temperature, volume, n_molecules, molecule_mass
    )
    Delta_rot, f_rot, s0_rot, DOS_rot_g, DOS_rot_s = decompose_rotational_dos(
        freqs, rDOS, temperature, volume, n_molecules, molecule_mass
    )
    Delta_vib, f_vib, s0_vib, _, _ = decompose_vibrational_dos(
        freqs, vDOS
    )
    print("DOS decomposition finished.")

    # Renormalize DOS AFTER decomposition
    if RENORMALIZE_DOS == 'after_f':
        # DOS re-normalization
        print(f"\nINFO: Re-normalizing DoS by scaling factor to match theoretical DOF")
        print(f"-- Scale the DOS to set the integral equal to 'real' degrees of freedom")
        print(f"-- 3*nMol (for translation, rotation) and 3*nMol*(nAtomPerMol-2) for vibration")
        print(f"-- This can lead to better convergence of entropy for short MD trajectories")
        print(f"-- Reference: M.A. Caro, T. Laurila, and O. Lopez-Acevedo, The Journal of Chemical Physics 145, (2016).")

        tDOS *= 3*n_molecules/np.trapz(tDOS, freqs)
        rDOS *= 3*n_molecules/np.trapz(rDOS, freqs)
        if CONSTRAINED_BONDS == False:
            vDOS *= 3*n_molecules*(atoms_per_molecule-2)/np.trapz(vDOS, freqs)

    # Calculate entropies
    S_tr = calculate_translational_entropy(
        freqs, DOS_tr_s, DOS_tr_g,
        f_tr, Delta_tr, molecule_mass, n_molecules, volume, temperature
    )
    S_rot = calculate_rotational_entropy(
        freqs, DOS_rot_s, DOS_rot_g,
        I_1, I_2, I_3, sigma, temperature
    )
    S_vib = calculate_vibrational_entropy(
        freqs, vDOS, temperature
    )
    
    # Calculate energies
    E_tr = calculate_translational_energy(
        freqs, DOS_tr_s, DOS_tr_g, temperature
    )
    E_rot = calculate_rotational_energy(
        freqs, DOS_rot_s, DOS_rot_g, temperature
    )
    E_vib = calculate_vibrational_energy(
        freqs, vDOS, temperature
    )
    E_zpe = calculate_zero_point_energy(
        freqs, vDOS, temperature
    )
    kinetic = (3*n_molecules)/(kB*temperature*eVtoJ) * (1 - f_tr/2 - f_rot/2)
    E_minus_EMD = (-kinetic + E_tr + E_rot + E_vib)

    # Save and plot spectra
    spectra_list = [
        {'freqs': freqs, 'DOS': tDOS, 'label': 'Translational'},
        {'freqs': freqs, 'DOS': DOS_tr_g, 'label': 'Translational (g)'},
        {'freqs': freqs, 'DOS': DOS_tr_s, 'label': 'Translational (s)'},
        {'freqs': freqs, 'DOS': rDOS, 'label': 'Rotational'},
        {'freqs': freqs, 'DOS': DOS_rot_g, 'label': 'Rotational (g)'},
        {'freqs': freqs, 'DOS': DOS_rot_s, 'label': 'Rotational (s)'},
        {'freqs': freqs, 'DOS': vDOS, 'label': 'Vibrational'},
        {'freqs': freqs, 'DOS': DOS_total, 'label': 'Total'},
    ]
    save_spectra_txt(spectra_list, filename='density_of_states.txt')
    plot_spectra(spectra_list, filename='density_of_states.png', xlim=[0,1500])

    # Calculate totals for each entropy-related row
    s0_total = s0_tr + s0_rot + s0_vib
    Delta_total = Delta_tr + Delta_rot + Delta_vib
    f_total = f_tr + f_rot + f_vib
    S_tr_mol = S_tr / n_molecules
    S_rot_mol = S_rot / n_molecules
    S_vib_mol = S_vib / n_molecules
    S_total_mol = (S_tr + S_rot + S_vib) / n_molecules

    # Calculate totals for energy row (convert J to kJ)
    kinetic_mol = (kinetic / n_molecules) / 1000
    E_tr_mol  = (E_tr / n_molecules)  / 1000
    E_rot_mol = (E_rot / n_molecules) / 1000
    E_vib_mol = (E_vib / n_molecules) / 1000
    E_zpe_mol = (E_zpe / n_molecules) / 1000
    E_minus_EMD_mol = (E_minus_EMD / n_molecules) / 1000

    # Print 2PT results
    print("\nTWO-PHASE THERMODYNAMICS RESULTS:")
    print("-"*100)
    print(f"{'Parameter':<20}{'Translational':>16}{'Rotational':>16}{'Vibrational':>16}{'Total':>16}")
    print("-"*100)
    print(f"{'S_0 (ps)':<20}{s0_tr:>16.2f}{s0_rot:>16.2f}{s0_vib:>16.2f}{s0_total:>16.2f}")
    print(f"{'Delta':<20}{Delta_tr:>16.5f}{Delta_rot:>16.5f}{Delta_vib:>16.5f}{Delta_total:>16.5f}")
    print(f"{'Fluidicity':<20}{f_tr:>16.5f}{f_rot:>16.5f}{f_vib:>16.5f}{f_total:>16.5f}")
    print(f"{'S (J/molK)':<20}{S_tr_mol:>16.2f}{S_rot_mol:>16.2f}{S_vib_mol:>16.2f}{S_total_mol:>16.2f}")
    print(f"{'E - E(MD) (kJ/mol)':<20}{E_tr_mol:>16.2f}{E_rot_mol:>16.2f}{E_vib_mol:>16.2f}{E_minus_EMD_mol:>16.2f}")
    print(f"{'ZPE (kJ/mol)':<20}{'':>16}{'':>16}{E_zpe_mol:>16.2f}{'':>16}")
    print("-"*100)

    # Save entropy results with total column as sum of three components per row
    np.savetxt(
        'entropy.txt',
        np.array(
            [S_tr_mol, S_rot_mol, S_vib_mol, S_total_mol],
        ),
        header='S:(J/mol K)   Translational   Rotational   Vibrational   Total',
        fmt='%.3f'
    )
    np.savetxt(
        'energy.txt',
        np.array(
            [E_tr_mol, E_rot_mol, E_vib_mol, E_zpe_mol, kinetic_mol, E_minus_EMD_mol]
        ),
        header='E-E(MD):(kJ/mol)   Translational   Rotational   Vibrational   ZPE   Kinetic-E   Total',
        fmt='%.3f',
    )

if __name__ == "__main__":
    main()

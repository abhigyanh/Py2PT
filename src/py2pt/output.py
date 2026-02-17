import matplotlib.pyplot as plt
import numpy as np

from .constants import *


def plot_spectra(spectra_list, filename='output.png', xlim=[0,1500]):
    """
    Plot multiple spectra on the same figure.
    
    Parameters
    ----------
    spectra_list : list of dict
        List of dictionaries containing spectrum data. Each dictionary should have:
        - 'freqs': array of frequencies
        - 'DOS': array of density of states values
        - 'label': label for the spectrum
    filename : str, optional
        Output filename, by default 'output.png'
    xlim : tuple, optional
        x-axis limits, by default None
    """
    n_spectra = len(spectra_list)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_spectra))
    for idx,spectrum in enumerate(spectra_list):
        freqs = spectrum['freqs']/c
        dos = spectrum['DOS']*c
        label = spectrum['label']
        plt.plot(freqs, dos, label=label, linestyle='-', color=colors[idx])
    
    plt.gca().set(
        xlabel = r'Wavenumber, $\nu$ (cm$^{-1}$)',
        ylabel = r'DoS, $S_{\nu}$ (cm)',
    )
    plt.legend()
    if xlim is not None:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def save_spectra_txt(spectra_list, filename='dos.txt'):
    """
    Save multiple spectra to a text file using numpy.savetxt.
    
    Parameters
    ----------
    spectra_list : list of dict
        List of dictionaries containing spectrum data. Each dictionary should have:
        - 'freqs': array of frequencies
        - 'DOS': array of density of states values
        - 'label': label for the spectrum
    filename : str, optional
        Output filename, by default 'dos.txt'
    """
    # Get frequencies (they should be the same for all spectra)
    freqs = spectra_list[0]['freqs']/c
    
    # Stack all DOS arrays together
    dos_arrays = []
    for spectrum in spectra_list:
        dos_arrays.append(c*spectrum['DOS'])
    
    # Combine all data into one array
    combined_data = np.column_stack([freqs] + dos_arrays)
    
    # Create detailed header
    header = "# Column 0: Wavenumber (cm^-1)\n"
    for i, spectrum in enumerate(spectra_list, 1):
        header += f"# Column {i}: {spectrum['label']}\n"
    
    # Save to file
    np.savetxt(filename, combined_data, fmt='%.6f', header=header, comments='')


def plot_histogram(I_lk, filename, bins=100, dpi=300):
    """
    Plot per-residue principal moments of inertia as histograms.

    Parameters
    ----------
    I_lk : np.ndarray
        Array of principal moments per molecule/residue, shape (n_molecules, 3).
        Columns correspond to (I_1, I_2, I_3).
    filename : str, optional
        Output image filename.
    bins : int, optional
        Number of histogram bins, by default 100.
    dpi : int, optional
        Figure DPI, by default 300.
    """
    I_lk = np.asarray(I_lk)
    if I_lk.ndim != 2 or I_lk.shape[1] != 3:
        raise ValueError(f"I_lk must have shape (n_molecules, 3); got {I_lk.shape}")

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    labels = [r"$I_1$", r"$I_2$", r"$I_3$"]
    palette = plt.cm.Set2(np.linspace(0.0, 1.0, 3))
    data = [I_lk[:, 0], I_lk[:, 1], I_lk[:, 2]]
    # Draw the histogram
    ax.hist(
        data,
        bins=bins,
        density=False,
        histtype="stepfilled",
        alpha=0.55,
        color=palette,
        edgecolor="black",
        linewidth=0.6,
        label=labels,
    )
    ax.set(
        title = "Distribution of principal moments of inertia (per molecule)",
        xlabel = r"Principal moment of inertia (amu·Å$^2$)",
        ylabel = "Count (Num. molecules/resids)",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)

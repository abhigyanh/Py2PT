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
    
    plt.xlabel(r'Wavenumber, $\nu$ (cm$^{-1}$)')
    plt.ylabel(r'DoS, $S_{\nu}$ (cm)')
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

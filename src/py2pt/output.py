import matplotlib.pyplot as plt

colors = ['red', 'blue', 'green']

def plot_power_spectra(wavenumbers, vt_power, vr_power, vv_power, filename='output.png', xlim=None):
    plt.figure()
    plt.plot(wavenumbers, vt_power, label='Translational', color=colors[0])
    plt.plot(wavenumbers, vr_power, label='Rotational', color=colors[1])
    plt.plot(wavenumbers, vv_power, label='Vibrational', color=colors[2])
    plt.xlabel(r'Wavenumber, $\nu$ (cm$^{-1}$)')
    plt.ylabel(r'DoS, $S(\nu)$')
    plt.legend()
    if xlim is not None:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)

def save_power_spectra_txt(wavenumbers, vt_power, vr_power, vv_power, filename='dos.txt'):
    with open(filename, 'w') as f:
        f.write("# Wavenumber (cm^-1), Translational Power, Rotational Power, Vibrational Power\n")
        for i in range(len(wavenumbers)):
            f.write(f"{wavenumbers[i]:.6f}, {vt_power[i]:.6f}, {vr_power[i]:.6f}, {vv_power[i]:.6f}\n")

def plot_dos_curves(nu, dos_dict, filename, xlim=None, xlabel=None, ylabel=None, title=None):
    """
    Plot one or more DoS curves from a dictionary and save the figure.
    dos_dict: dict of label -> y-array
    """
    plt.figure()
    for label, y in dos_dict.items():
        plt.plot(nu, y, label=label, color=colors[list(dos_dict.keys()).index(label) % len(colors)])
    if xlim is not None:
        plt.xlim(*xlim)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close() 
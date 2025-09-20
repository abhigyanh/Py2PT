import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from py2pt.entropy import calculate_delta, _equation_of_fluidicity, calculate_fluidicity

def fluidicity_inverse_problem(s0_guess=1.0):
    def func(s0):
        delta = calculate_delta(T, V, N, m, s0)
        return _equation_of_fluidicity(f_target, delta)
    # Use root_scalar to solve for s0
    result = root_scalar(func, bracket=[1e-8, 1e8], method='brentq')
    if result.converged:
        print(f"For fluidicity f = {f_target}, solved s0 = {result.root} ps")
    else:
        print("Root finding did not converge.")

def plot_delta_vs_fluidicity():
    # Create array of delta values
    delta_values = np.logspace(-4, 4, 100)  # 100 points from 1e-4 to 1e4
    
    # Calculate fluidicity for each delta
    fluidicity_values = []
    for delta in delta_values:
        try:
            f = calculate_fluidicity(delta)
            fluidicity_values.append(f)
        except:
            fluidicity_values.append(np.nan)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.loglog(delta_values, fluidicity_values, 'b-', label='Fluidicity')
    plt.xlabel('Delta')
    plt.ylabel('Fluidicity')
    plt.title('Delta vs Fluidicity')
    plt.grid(True)
    plt.legend()
    plt.savefig('delta_vs_fluidicity.png')
    plt.close()

def plot_fluidicity_equation(delta=None):
    # Calculate delta for some s0
    if delta is None:
        s0 = 1.0  # example s0 value
        delta = calculate_delta(T, V, N, m, s0)
        print(f"Using delta = {delta:.2e}")
    
    # Create array of f values
    f_values = np.linspace(1e-6, 1, 1000)
    
    # Calculate equation values
    eq_values = [_equation_of_fluidicity(f, delta) for f in f_values]
    
    # Check signs at bracket ends
    f_low = 1e-6
    f_high = 1.0
    val_low = _equation_of_fluidicity(f_low, delta)
    val_high = _equation_of_fluidicity(f_high, delta)
    print(f"\nBracket check:")
    print(f"At f={f_low}: {val_low:.2e}")
    print(f"At f={f_high}: {val_high:.2e}")
    print(f"Opposite signs: {np.sign(val_low) != np.sign(val_high)}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(f_values, eq_values, 'b-', label='Fluidicity equation')
    plt.axhline(y=0, color='r', linestyle='--', label='Zero line')
    plt.xlabel('f')
    plt.ylabel('Equation value')
    plt.title(f'Fluidicity equation vs f (delta={delta:.2e})')
    plt.grid(True)
    plt.legend()
    plt.savefig('fluidicity_equation.png')
    plt.close()

if __name__ == "__main__":
    # Example fixed parameters
    T = 298.15  # Kelvin
    rho = 0.9968 # g/cm3
    N = 2000.0   # Number of molecules
    m = 18.0   # amu (e.g., water)

    # Calculate total mass in grams
    mass_amu = N * m
    mass_g = mass_amu * 1.66053906660e-24  # 1 amu = 1.66054e-24 g
    # Calculate volume in cm^3
    V_cm3 = mass_g / rho
    # Convert cm^3 to Angstrom^3 (1 cm^3 = 1e24 Angstrom^3)
    V = V_cm3 * 1e24

    # Ask user for delta value
    delta_input = input("Enter delta value (press Enter to use example): ").strip()
    delta_input = float(delta_input)
    
    if delta_input:
        try:
            delta = float(delta_input)

        except ValueError:
            print("Invalid input. Using example delta value.")
            s0 = 1.0  # example s0 value
            delta = calculate_delta(T, V, N, m, s0)
    else:
        s0 = 1.0  # example s0 value
        delta = calculate_delta(T, V, N, m, s0)
    
    print(f"Using delta = {delta:.2e}")

    # Specify the target fluidicity value
    f_target = 0.29  # Example value

    # fluidicity_inverse_problem()
    # plot_delta_vs_fluidicity()
    # Calculate fluidicity using the solver if delta_input is provided
    if delta_input:
        f = calculate_fluidicity(delta_input)
        print(f"Calculated fluidicity: {f:.4f}")
    plot_fluidicity_equation(delta=delta_input)
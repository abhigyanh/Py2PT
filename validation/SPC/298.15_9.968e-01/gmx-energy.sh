# Create energy output directories if they don't exist
mkdir -p ener-eq ener

# Extract energy data for eq.edr
echo "Extracting energy data from eq.edr..."
echo "Potential" | gmx energy -f eq.edr -o ./ener-eq/potential.xvg
echo "Kinetic-En" | gmx energy -f eq.edr -o ./ener-eq/kinetic.xvg
echo "Total-Energy" | gmx energy -f eq.edr -o ./ener-eq/total.xvg
echo "Temperature" | gmx energy -f eq.edr -o ./ener-eq/temperature.xvg
echo "Pressure" | gmx energy -f eq.edr -o ./ener-eq/pressure.xvg
echo "Density" | gmx energy -f eq.edr -o ./ener-eq/density.xvg

# Extract energy data for ener.edr
echo "Extracting energy data from ener.edr..."
echo "Potential" | gmx energy -f ener.edr -o ./ener/potential.xvg
echo "Kinetic-En" | gmx energy -f ener.edr -o ./ener/kinetic.xvg
echo "Total-Energy" | gmx energy -f ener.edr -o ./ener/total.xvg
echo "Temperature" | gmx energy -f ener.edr -o ./ener/temperature.xvg
echo "Pressure" | gmx energy -f ener.edr -o ./ener/pressure.xvg
echo "Density" | gmx energy -f ener.edr -o ./ener/density.xvg

echo "Energy extraction complete."
# Create energy output directories if they don't exist
mkdir -p ener-npt ener

# Extract energy data for npt.edr
echo "Extracting energy data from npt.edr..."
echo "Potential" | gmx energy -f npt.edr -o ./ener-npt/potential.xvg
echo "Kinetic-En" | gmx energy -f npt.edr -o ./ener-npt/kinetic.xvg
echo "Total-Energy" | gmx energy -f npt.edr -o ./ener-npt/total.xvg
echo "Temperature" | gmx energy -f npt.edr -o ./ener-npt/temperature.xvg
echo "Pressure" | gmx energy -f npt.edr -o ./ener-npt/pressure.xvg
echo "Density" | gmx energy -f npt.edr -o ./ener-npt/density.xvg

# Extract energy data for ener.edr
echo "Extracting energy data from ener.edr..."
echo "Potential" | gmx energy -f ener.edr -o ./ener/potential.xvg
echo "Kinetic-En" | gmx energy -f ener.edr -o ./ener/kinetic.xvg
echo "Total-Energy" | gmx energy -f ener.edr -o ./ener/total.xvg
echo "Temperature" | gmx energy -f ener.edr -o ./ener/temperature.xvg
echo "Pressure" | gmx energy -f ener.edr -o ./ener/pressure.xvg
echo "Density" | gmx energy -f ener.edr -o ./ener/density.xvg

echo "Energy extraction complete."
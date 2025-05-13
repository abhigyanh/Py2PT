#!/bin/bash

# Exit on error
set -e

# Define input/output files
TRAJ="traj.trr"
OUTPUT="entropy/trajectory.gro"
TPR="prod.tpr"  # TPR file is needed for proper PBC handling
NDX="no_mw.ndx"  # Index file excluding MW atoms

# Check if input files exist
if [ ! -f $TRAJ ]; then
    echo "Error: $TRAJ not found!"
    exit 1
fi

if [ ! -f $TPR ]; then
    echo "Error: $TPR not found!"
    exit 1
fi

echo "Creating index file excluding MW atoms..."
# Create index file excluding MW atoms
echo 'not name MW' | gmx select -s $TPR -on $NDX

echo "Processing trajectory with PBC corrections..."
# Run trjconv with PBC whole option and the new index file
# Echo "0" to select the group from our custom index file
echo "System" | gmx trjconv -f $TRAJ -s $TPR -n $NDX -o $OUTPUT -pbc whole

echo "Trajectory processing complete. Output saved to $OUTPUT" 
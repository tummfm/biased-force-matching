#!/bin/bash

# Exit on any error
set -e

# Number of threads
THREADS=10
MD_THREADS=10
GPU_ID=0

# Step 1: Generate topology
echo "### Step 1: pdb2gmx ###"
gmx pdb2gmx -f C5.pdb -water tip3p -o initial.gro -ff amber99sb-ildn

# Step 2: Create cubic box
echo "### Step 2: editconf ###"
gmx editconf -f initial.gro -bt cubic -box 3.7 3.7 3.7 -o cubic.gro

# Step 3: Solvate
echo "### Step 3: solvate ###"
gmx solvate -cp cubic.gro -cs spc216.gro -p topol.top -o water.gro

# Step 4: Energy Minimization
echo "### Step 4: energy minimization ###"
gmx grompp -f minim.mdp -c water.gro -p topol.top -o em.tpr
gmx mdrun -v -deffnm em -nt "$THREADS"

# Step 5: NVT Equilibration
echo "### Step 5: NVT equilibration ###"
gmx grompp -f mdSolventNVT.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
gmx mdrun -v -deffnm nvt -nt "$THREADS"

# Step 6: NPT Equilibration
echo "### Step 6: NPT equilibration ###"
gmx grompp -f mdSolventNPT.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 1
gmx mdrun -v -deffnm npt -nt "$MD_THREADS" -gpu_id "$GPU_ID"

# Step 7: Prepare MD
echo "### Step 7: Prepare MD production ###"
gmx grompp -f mdSolventVrescale.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr

# Step 8: Run MD Simulation
echo "### Step 8: Run MD simulation ###"
gmx mdrun -v -deffnm md -nt "$MD_THREADS" -gpu_id "$GPU_ID" -cpt 50 -plumed plumed.dat

echo "### Step 9: Rerun MD simulation ###"
gmx mdrun -s md.tpr -rerun md.trr -deffnm rerun_forces -nt "$MD_THREADS" -gpu_id "$GPU_ID"

# Step 9: Create heavy atom index
echo "### Step 9: Create heavy atom index ###"
echo -e "keep 2\nq" | gmx make_ndx -f md.gro -o heavy.ndx

# Step 10: Convert trajectory to heavy atom only, centered, PBC-corrected
echo "### Step 10: Extract heavy atoms trajectory ###"
gmx trjconv -s md.tpr -f rerun_forces.trr -n heavy.ndx -o rerun_heavyMD.trr -pbc mol -center -ur compact -force
gmx trjconv -s md.gro -f md.trr -n heavy.ndx -o heavy_first.gro -dump 0 
gmx rama -f md.trr -s md.tpr -o output.xvg


echo "Simulation workflow completed."
#!/usr/bin/env python

"""
Markov Chains Traffic Assignment (MCTA) Command Line Interface (CLI)

Syntax:
	ipython mcta_cli.py -- [options] {BaseName}

options:
	--verbose	|	-v		display verbose messages and detailed results
	--showfigs	|	-f		generate figures
	--figs2png	|	-s		save figures into PNG files
	--memdump	|	-m		save mem dump variables into JSON file
							({BaseName}-Results.json)

Input data files to load (must be in the script folder):
	mcta_vis.json					visualization settings JSON file
	{BaseName}-Map.png				Base map PNG image, used in road map rendering
	{BaseName}-Map.geojson			Road network GeoJSON file (for road attributes)

Optional output files:
	{BaseName}-Results-Fig#.png		Figures png files
	{BaseName}-Results.json			mem dump variables into JSON file

Complete Code run-through test:
	ipython ./mcta.py -- --verbose --showfigs --figs2png --memdump {BaseName}
	ipython ./mcta.py -- -v -f -s -m {BaseName}

Code by Sinan Salman, 2017-2019
"""

HELP_MSG = "Options:\n\
\t--verbose  | -v  display verbose messages and detailed results\n\
\t--showfigs | -f  generate figures\n\
\t--figs2png | -s  save figures into PNG files\n\
\t--memdump  | -m  save mem dump variables into JSON file\n\
\t                   ({BaseName}-Results.json)\n"

__author__ = "Sinan Salman (sinan.salman@zu.ac.ae)"
__version__ = "Revision: 0.18"
__date__ = "Date: 2019/07/17"
__copyright__ = "Copyright (c)2017-2019 Sinan Salman"
__license__ = "GPLv3"

### Initialization #######################################################################

import sys
import mcta
import mcta_vis
import mcta_rw
import scipy as sp

# added for updating
import numpy as np
#from simulated_annealing import solve_simulated_annealing_problem  # Step 1: Import simulated_annealing function
from simulated_annealing import solve_annealing_problem as solve_simulated_annealing_problem


# options
verbose = False
showfigs = False
figs2png = False
savememdump = False

# global variables
base = None

# OLD: sp.set_printoptions(suppress=True, precision=3, linewidth=140)
# NEW:
np.set_printoptions(suppress=True, precision=3, linewidth=140)


### Data processing ######################################################################


def ProcessCLI():
    """Process CLI parameters"""
    global base
    global verbose
    global showfigs
    global figs2png
    global savememdump

    if len(sys.argv) == 1:
        print(
            "Missing argument\n\nSyntax:\n\tipython mcta_cli.py -- [options] {BaseName}"
        )
        print(HELP_MSG)
        sys.exit(0)
    if "--verbose" in sys.argv or "-v" in sys.argv:
        print("*** option: verbose mode")
        verbose = True
    if "--showfigs" in sys.argv or "-f" in sys.argv:
        print("*** option: generate figures")
        showfigs = True
    if "--figs2png" in sys.argv or "-s" in sys.argv:
        print("*** option: figures will be saved to PNG files")
        figs2png = True
    if "--memdump" in sys.argv or "-m" in sys.argv:
        print("*** option: save MemDump to JSON file")
        savememdump = True
    base = sys.argv[len(sys.argv) - 1]


### Main #################################################################################

if __name__ == "__main__":
    ProcessCLI()
    (Settings, GeoJSON) = mcta_rw.LoadDataFiles(base)
    lengths, lanes, FFS, P = mcta.SetupMCTA(GeoJSON, Verbose=verbose)

    # simulated_annealing Solution
    Results_simulated_annealing = solve_simulated_annealing_problem(
        lengths=lengths,
        lanes=lanes,
        P=P,
        VehiclesCount=GeoJSON["mcta_json"]["VehiclesCountEst"],
        Objectives=["D","K","C","PI","E"],
        FreeFlowSpeeds=FFS,
        SkipChecksForSpeed=False,
    )
        #max_iterations=200  # Added parameter
    # GA solution
    Results_GA = mcta.SolveMCTA(
        lengths=lengths,
        lanes=lanes,
        P=P,
        VehiclesCount=GeoJSON["mcta_json"]["VehiclesCountEst"],
        Objectives=["D", "K", "C", "PI", "E"],
        FreeFlowSpeeds=FFS,
        SkipChecksForSpeed=False,
    )

    # Comparison of Results
    print("\n=== Comparison of GA and simulated_annealing Solutions ===")
    print(f"GA Total Cost: {Results_GA['TotalNetworkEmissionCost']:.2f} $/hr")
    # uncomment when solved
    print(f"simulated_annealing Total Cost: {Results_simulated_annealing['TotalNetworkEmissionCost']:.2f} $/hr")

    if savememdump:
        mcta_rw.SaveResults(Results_GA)
    if showfigs or figs2png:
        mcta_vis.Initialize(
            GeoJSON, Settings, base, ShowFigs=showfigs, SaveFigs2PNG=figs2png
        )
    for x in Results_GA.keys():
        if x not in ["linkIDs", "P_org", "P_updated", "eigenvalues"]:
            print(f"\n{x}: {Results_GA[x]}")
            if showfigs or figs2png:
                mcta_vis.Generate_Figure(Results_GA, x)
    if showfigs:
        import matplotlib.pyplot as plt

        plt.show(block=True)

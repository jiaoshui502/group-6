# Accreted Milky Way Globular Clusters

PHYS3116 computational assignment,  Option 1.

Group 6: Jianing(Boris) Mao, Wenze Du, Manyan(Rachael) Li.

Some of the Milky Way's globular clusters formed in the Galaxy itself, and others were brought in by dwarf galaxies that merged with it. This project uses ages, metallicities ([Fe/H]), positions and line-of-sight velocities to find clusters that may have been accreted. It then tests the cluster system as a whole to estimate how many clusters were accreted and what that says about how the Milky Way formed to answer questions ;mentioned in the assignment brief.

## Quick Start

You need Python 3 with numpy, pandas, matplotlib and scipy.

Each script runs on its own:

- It reads the CSV files from its own folder, so you can run it from any directory
- It prints its results to the terminal.
- It opens its figures one at a time. Close each window to continue to the next plot.


## Scripts used for the final results


| Script                                                         | What it does                                                                                                                                                                                                                    | Figures                                                              |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| [stellar_population_method.py](stellar_population_method.py)   |  Fits a straight line of age against [Fe/H] (van den Berg ages), clips clusters beyond 3σ and fits again. Clusters more than 2σ younger than the trend are candidates.                                               | `age-metallicity.png`                                                |
| [Krause21_population_method.py](Krause21_population_method.py) |  The same test with the Krause21 ages.                                                                                                                                                                               | `age-metallicity-krause21.png`                                       |
| [dynamics_method.py](dynamics_method.py)                       |  Rotation of the whole cluster system from line-of-sight velocities (Frenk & White 1980). Clusters more than 2σ from the bulk rotation are candidates.                                                               | `rotation.png`                                                       |
| [HB_metallicity_analysis.py](HB_metallicity_analysis.py)       | Models horizontal-branch (HB) type against [Fe/H] and finds HB outliers (beyond 1.5σ).                                                                                                                                          | `hb-metallicity.png`                                                 |
| [HBage.py](HBage.py)                                           | Correlates each cluster's offset from the age trend with its offset from the HB trend. This tests "age as the second parameter".                                                                                                | `hb-age-residuals.png`                                               |
| [combined_candidates.py](combined_candidates.py)               | Repeats tests A to C and adds test D, a friends-of-friends search for groups of outer-halo clusters that are close in both position and velocity, with a shuffle test for chance. Joins all four tests into one candidate list. | `candidates-by-test.png`, `outer-halo-sky.png`                       |
| [formation_analysis.py](formation_analysis.py)                 | Tests the cluster system as a whole: rotation by metallicity, age offset against R_gc, two age-metallicity branches, and the estimate of how many clusters are accreted.                                                        | `two-branches.png`, `age-offset-vs-rgc.png`, `rotation-by-group.png` |
| [option1.py](option1.py)                   | Our first combined script, written before the work was split into the separate method scripts above. It has been replaced by those scripts. |
| [age.py](age.py)                           | An alternative weighted age fit.                                                                                                            |
| [sptial analysis.py](sptial%20analysis.py) | Face-on and edge-on maps of the cluster positions.                                                                                          |

                               |




## Other files

- [meeting.md](meeting.md): weekly meeting minutes
- [figures/](figures/): saved figures from code output





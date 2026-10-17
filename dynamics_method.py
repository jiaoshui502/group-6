import math 
from pathlib import Path 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


# some flexible settings:

# we define a cluster does not follow the rotation if it is more than this many sigma from what the rotation predicts. 
dynamics_sigma_cut = 2.0

# motion of the sun
V0 = 229.0 # km/s, speed of the Sun's circular motion around the Galaxy
R0 = 8.178 # kpc, distance from the Sun to the galactic centre


# loading data
folder = Path(__file__).resolve().parent 
harris1 = pd.read_csv(folder / "HarrisPartI.csv") # positions
harris3 = pd.read_csv(folder / "HarrisPartIII.csv") # velocities


# Rotational dynamics
# We only know the radial (line-of-sight) velocity, the sideways motion(proper motion) is not available in the given data.
# So we ask: if the whole GC system rotated around the galactic centre at one speed v_rot, what actual velocity would each cluster show to us? 
# This is the method of Frenk & White (1980).

# join positions and velocities by cluster name, keep clusters in both tables
dyn = pd.merge(harris1, harris3, on="ID", how="inner")
# drop clusters missing any number the method needs
dyn = dyn.dropna(subset=["L", "B", "R_Sun", "v_LSR"]).copy()

l = np.radians(dyn["L"]) # galactic longitude in radians
b = np.radians(dyn["B"]) # galactic latitude in radians
d = dyn["R_Sun"] # distance from the Sun, kpc

# Step 1
# how much of the Sun's circular motion (towards l = 90 deg) lies along our line of sight to the cluster
cos_A = np.sin(l) * np.cos(b)
# line-of-sight velocity seen by an observer at rest at the Sun's position
dyn["V_S"] = dyn["v_LSR"] + V0 * cos_A

# Step 2
x = d * np.cos(b) * np.cos(l) # position in the disc plane, towards the galactic centre
y = d * np.cos(b) * np.sin(l) # position in the disc plane, along the Sun's motion
# part of a circular orbit at the cluster's position that points along our line of sight; the square root is the cluster's distance from the
# galactic centre measured in the disc plane
dyn["cos_Psi"] = R0 * cos_A / np.sqrt((x - R0) ** 2 + y ** 2)

# Step 3
# best single rotation speed (least-squares line through zero)
v_rot = (dyn["cos_Psi"] * dyn["V_S"]).sum() / (dyn["cos_Psi"] ** 2).sum()

# Step 4
# peculiar velocity: what is left after the shared rotation is taken away
dyn["v_pec"] = dyn["V_S"] - v_rot * dyn["cos_Psi"]
# sigma: the typical size of the peculiar velocities, how much random motion the clusters have on top of the rotation
dyn_sigma = dyn["v_pec"].std()
# score: how many sigma each cluster is from the rotation prediction
dyn["dyn_z"] = dyn["v_pec"] / dyn_sigma
# True if the cluster is more than the cut away, in either direction
dyn["dyn_standout"] = dyn["dyn_z"].abs() > dynamics_sigma_cut

# Uncertainty of v_rot (standard result for this estimator): the error of the slope of a straight line through zero, given scatter dyn_sigma.
v_rot_err = dyn_sigma / np.sqrt((dyn["cos_Psi"] ** 2).sum())

print("Rotation of the GC System")
print(f"  clusters used: {len(dyn)}") # how many clusters had all the numbers needed
print(f"  rotation speed v_rot = {v_rot:.0f} +/- {v_rot_err:.0f} km/s") # the shared rotation speed and its error
print(f"  typical scatter (sigma) = {dyn_sigma:.0f} km/s") # the random motion on top of the rotation
# How many clusters do not follow the rotation, and how many would by pure chance
print(f"  clusters that do not follow the rotation: {dyn['dyn_standout'].sum()} of {len(dyn)}"
      f"   (about {len(dyn) * math.erfc(dynamics_sigma_cut / math.sqrt(2)):.1f} expected by chance)") 

# Assignment task 1: Identify potentially accreted clusters using dynamics method
# candidate that does not follow the bulk rotation. Both signs count: a cluster moving faster than expected in either direction does not follow the bulk.

# keep only the flagged clusters, sorted by their score
candidates = dyn[dyn["dyn_standout"]].sort_values("dyn_z")

print()
print("Potentially accreted cluester")
# how many candidates, out of how many clusters that could be tested
print(f"  {len(candidates)} candidates out of {len(dyn)} clusters with a velocity")
# Table of the candidates: name, distance from the galactic centre,
print(candidates[["ID", "R_gc", "v_LSR", "v_pec", "dyn_z"]]
      .round(1).to_string(index=False)) # measured velocity, peculiar velocity and score



# Making sure close by labels alternate between four positions so that they do not sit on top of each other.

def label_points(x_values, y_values, names):
    # the four label positions, in points measured from the marker
    offsets = [(10, 10), (10, -16), (-10, 10), (-10, -16)]
    for i, (xv, yv, name) in enumerate(zip(x_values, y_values, names)):
        # i 4 cycles through 0, 1, 2, 3, 0, ... so that neighbouring labels get different positions
        dx, dy = offsets[i % 4]
        plt.annotate(
            name, #
            (xv, yv), 
            xytext=(dx, dy),
            textcoords="offset points", 
            ha="left" if dx > 0 else "right", 
            fontsize=8, 
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.6), 
        ) 


# plot
out = dyn[dyn["dyn_standout"]].sort_values("cos_Psi")   # clusters that do not follow

plt.figure(figsize=(10, 7)) 

# every cluster: its rotation geometry (x) against its measured velocity (y)
plt.scatter(dyn["cos_Psi"], dyn["V_S"], color="tab:blue", alpha=0.6, label="Clusters")

# the rotation line V_S = v_rot * cos(Psi) and the band of +-sigma cut
x_line = np.linspace(-1, 1, 100) # cos(Psi) can only be between -1 and +1
y_line = v_rot * x_line # the velocity the bulk rotation predicts at each cos(Psi)
plt.plot(x_line, y_line, color="black", label=f"Bulk rotation: v_rot = {v_rot:.0f} +/- {v_rot_err:.0f} km/s") 
plt.fill_between(x_line, y_line - dynamics_sigma_cut * dyn_sigma, y_line + dynamics_sigma_cut * dyn_sigma,  color="gray", alpha=0.2, 
label=f"Within {dynamics_sigma_cut:g} sigma of rotation") 


plt.scatter(out["cos_Psi"], out["V_S"], s=90, color="red", zorder=3, label=f"Do not follow the rotation (more than {dynamics_sigma_cut:g} sigma)") 
label_points(out["cos_Psi"], out["V_S"], out["ID"]) 

plt.axhline(0, color="black", linewidth=0.6, alpha=0.4) 
plt.axvline(0, color="black", linewidth=0.6, alpha=0.4) 
plt.xlabel("cos(Psi)   (+1 or -1: moving along our line of sight, 0: sideways)")
plt.ylabel("V_S   (km/s)") 
plt.title("Rotation of the Milky Way Globular Cluster System") 
plt.legend(loc="lower right") 
plt.tight_layout() 
plt.show()

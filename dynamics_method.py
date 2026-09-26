import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# some flexible settings:

# we define acluster does not follow the rotation if it is more than this many
# sigma from what the rotation predicts
dynamics_sigma_cut = 2.0

# motion of the sun, modern values
V0 = 229.0 # speed of the sun's circular motion around the galaxy, km/s
R0 = 8.178 # distance from the sun to the galactic centre, kpc


# Part 1: loading data
folder = Path(__file__).resolve().parent #locate file path
harris1 = pd.read_csv(folder / "HarrisPartI.csv") # positions
harris3 = pd.read_csv(folder / "HarrisPartIII.csv") # velocities


# Part 2. rotational dynamics
# we only know the radial velocity,and sideways motion is not available 
# in the given data
# So we ask: if the whole GC system rotated around the galactic centre at 
# one speed v_rot, what actural velocity would each cluster show to us? 
# Then we compare with what was measured.

# Step 1:
# We use V_S, the radial velocity relative to a stationary observer at
# the Sun's position. Harris gives v_LSR, measured relative to the
# Local Standard of Rest, which itself circles the Galaxy at V0. We add
# back the part of that circular motion that points along the line of
# sight:  V_S = v_LSR + V0 * cos(A),  where cos(A) = sin(l) * cos(b)
# and l and b are the cluster's galactic longitude and latitude.)
#
# Step 2:
# we use cos(Psi), the angle Psi is between the line of sight and the
# direction the cluster would move if it circled the galactic centre
# with the system. A cluster moving straight along our line of sight
# has cos(Psi) = +-1; one moving sideways has cos(Psi) = 0.
# If the system rotates at v_rot, we expect  V_S = v_rot * cos(Psi).
# To get cos(Psi) we need the cluster's distance from the galactic
# centre measured in the plane of the disc (R_disc), using the distance
# from the Sun (R_Sun) and the direction (l, b) we have:
#       cos(Psi) = R0 * cos(A) / R_disc
#
# Step 3:
# Best rotation speed: the single number v_rot that makes the
# line V_S = v_rot * cos(Psi) fit best (least squares through zero):
#       v_rot = sum(cos(Psi) * V_S) / sum(cos(Psi)^2)
#
# Step 4:
# Peculiar velocity v_pec = V_S - v_rot * cos(Psi): how much a
# cluster's motion differs from the group rotation. sigma = typical
# size of v_pec for all clusters. Score = v_pec / sigma.
# |score| > dynamics_sigma_cut means the cluster does not follow the bulk.

# Join positions and velocities on the cluster name, and keep only
# clusters that have every number we need.
dyn = pd.merge(harris1, harris3, on="ID", how="inner")
dyn = dyn.dropna(subset=["L", "B", "R_Sun", "v_LSR"]).copy()

# Angles in radians because numpy trigonometry needs radians.
l = np.radians(dyn["L"])
b = np.radians(dyn["B"])
d = dyn["R_Sun"] # distance from the Sun, kpc

# Step 1
cos_A = np.sin(l) * np.cos(b)
dyn["V_S"] = dyn["v_LSR"] + V0 * cos_A

# Step 2: position in the disc plane, with the Sun at (0, 0) and the
# galactic centre at (R0, 0). R_disc is the cluster's distance from the
# galactic centre measured in that plane.
x = d * np.cos(b) * np.cos(l)
y = d * np.cos(b) * np.sin(l)
R_disc = np.sqrt((x - R0) ** 2 + y ** 2)
dyn["cos_Psi"] = R0 * cos_A / R_disc

# Step 3
v_rot = (dyn["cos_Psi"] * dyn["V_S"]).sum() / (dyn["cos_Psi"] ** 2).sum()

# Step 4
dyn["v_pec"] = dyn["V_S"] - v_rot * dyn["cos_Psi"]
dyn_sigma = dyn["v_pec"].std()
dyn["dyn_z"] = dyn["v_pec"] / dyn_sigma
dyn["dyn_standout"] = dyn["dyn_z"].abs() > dynamics_sigma_cut

# Uncertainty of v_rot (standard result for this estimator).
v_rot_err = dyn_sigma / np.sqrt((dyn["cos_Psi"] ** 2).sum())

print("Rotation of the GC System")
print(f"  clusters used: {len(dyn)}")
print(f"  rotation speed v_rot = {v_rot:.0f} +/- {v_rot_err:.0f} km/s")
print(f"  typical scatter (sigma) = {dyn_sigma:.0f} km/s")
print(f"  clusters that do not follow the rotation: {dyn['dyn_standout'].sum()} of {len(dyn)}"
      f"   (about {len(dyn) * math.erfc(dynamics_sigma_cut / math.sqrt(2)):.1f} expected by chance)")


# Assignment task 1: Identify potentially accreted cluester using dynamics
# method
# Rule: candidate that does not follow the bulk rotation. Both signs count:
# a cluster moving faster than expected in either direction does not
# follow the bulk.

candidates = dyn[dyn["dyn_standout"]].sort_values("dyn_z")

print()
print("Potentially accreted cluester")
print(f"  {len(candidates)} candidates out of {len(dyn)} clusters with a velocity")
print(candidates[["ID", "R_gc", "v_LSR", "v_pec", "dyn_z"]]
      .round(1).to_string(index=False))


# Labels for the plot:
# Making sure close by labels alternate between four positions so that 
# they do not sit on top of each other.
# 4 marker locations: (right, up), (right, down), (left, up), (left, down).

def label_points(x_values, y_values, names):
    offsets = [(10, 10), (10, -16), (-10, 10), (-10, -16)]
    for i, (xv, yv, name) in enumerate(zip(x_values, y_values, names)):
        dx, dy = offsets[i % 4]
        plt.annotate(
            name,
            (xv, yv), # the point being labelled
            xytext=(dx, dy), # where the text goes
            textcoords="offset points",
            ha="left" if dx > 0 else "right", # text grows away from the point
            fontsize=8,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
        )


# plot
out = dyn[dyn["dyn_standout"]].sort_values("cos_Psi")   # clusters that do not follow

plt.figure(figsize=(10, 7))

# every cluster
plt.scatter(dyn["cos_Psi"], dyn["V_S"], color="tab:blue", alpha=0.6, label="Clusters")

# the rotation line V_S = v_rot * cos(Psi) and the band of +-sigma cut
x_line = np.linspace(-1, 1, 100)
y_line = v_rot * x_line
plt.plot(x_line, y_line, color="black",
         label=f"Bulk rotation: v_rot = {v_rot:.0f} +/- {v_rot_err:.0f} km/s")
plt.fill_between(x_line, y_line - dynamics_sigma_cut * dyn_sigma, y_line + dynamics_sigma_cut * dyn_sigma,
                 color="gray", alpha=0.2, label=f"Within {dynamics_sigma_cut:g} sigma of rotation")

# clusters outside the band, in red, with names
plt.scatter(out["cos_Psi"], out["V_S"], s=90, color="red", zorder=3,
            label=f"Do not follow the rotation (more than {dynamics_sigma_cut:g} sigma)")
label_points(out["cos_Psi"], out["V_S"], out["ID"])

plt.axhline(0, color="black", linewidth=0.6, alpha=0.4)
plt.axvline(0, color="black", linewidth=0.6, alpha=0.4)
plt.xlabel("cos(Psi)   (+1 or -1: moving along our line of sight, 0: sideways)")
plt.ylabel("V_S   (km/s)")
plt.title("Rotation of the Milky Way Globular Cluster System")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DATA

folder = Path(__file__).resolve().parent

harris = pd.read_csv(
    folder / "HarrisPartI.csv"
)

# Keep clusters with complete spatial information
spatial = harris.dropna(
    subset=["X", "Y", "Z", "R_gc"]
).copy()

print("Number of clusters used:", len(spatial))

# GALACTOCENTRIC COORDINATES
# Harris coordinate convention:
# Sun = (0, 0, 0)
# X points towards the Galactic Centre
# Y points in the direction of Galactic rotation
# Z points towards the North Galactic Pole
# Galactic Centre is approximately at (8, 0, 0) kpc
# in the original Harris coordinate system.

R0 = 8.0  # Sun-Galactic Centre distance used by Harris (kpc)

spatial["X_gc"] = spatial["X"] - R0
spatial["Y_gc"] = spatial["Y"]
spatial["Z_gc"] = spatial["Z"]

# DERIVED SPATIAL QUANTITIES

# Cylindrical radius:
# distance from the Galactic rotation axis
spatial["R_cyl"] = np.sqrt(
    spatial["X_gc"]**2
    + spatial["Y_gc"]**2
)

# Height from Galactic plane
spatial["abs_Z"] = np.abs(
    spatial["Z_gc"]
)

# Recalculate Galactocentric radius from XYZ
spatial["R_gc_calc"] = np.sqrt(
    spatial["X_gc"]**2
    + spatial["Y_gc"]**2
    + spatial["Z_gc"]**2
)

# Compare with Harris catalogue value
spatial["R_gc_diff"] = (
    spatial["R_gc_calc"]
    - spatial["R_gc"]
)

max_difference = np.abs(
    spatial["R_gc_diff"]
).max()

print(
    f"Maximum |calculated R_gc - catalogue R_gc| "
    f"= {max_difference:.3f} kpc"
)

# SPATIAL PERCENTILES
# These give the relative spatial position of each cluster
# without imposing an arbitrary physical cutoff.
#
# Example:
# R_gc_percentile = 95 means the cluster has a larger
# Galactocentric radius than approximately 95% of the sample.

spatial["R_gc_percentile"] = (
    spatial["R_gc"]
    .rank(pct=True)
    * 100
)

spatial["R_cyl_percentile"] = (
    spatial["R_cyl"]
    .rank(pct=True)
    * 100
)

spatial["Z_percentile"] = (
    spatial["abs_Z"]
    .rank(pct=True)
    * 100
)

# BASIC SPATIAL STATISTICS

print("\nSpatial distribution summary:")

summary = spatial[
    ["R_gc", "R_cyl", "abs_Z"]
].describe(
    percentiles=[
        0.25,
        0.50,
        0.75,
        0.90
    ]
)

print(
    summary.round(2)
)

# FACE-ON VIEW
# Galactic plane: X_gc vs Y_gc

plt.figure(figsize=(8, 8))

# All clusters
plt.scatter(
    spatial["X_gc"],
    spatial["Y_gc"],
    alpha=0.65,
    label="Globular clusters"
)

# Galactic Centre
plt.scatter(
    0,
    0,
    marker="*",
    s=180,
    label="Galactic Centre",
    zorder=4
)

# Sun
plt.scatter(
    -R0,
    0,
    marker="o",
    s=90,
    label="Sun",
    zorder=4
)

# Solar circle
theta = np.linspace(
    0,
    2 * np.pi,
    400
)

plt.plot(
    R0 * np.cos(theta),
    R0 * np.sin(theta),
    linestyle="--",
    linewidth=1,
    alpha=0.4,
    label="Solar circle"
)

# Reference axes
plt.axhline(
    0,
    linewidth=0.8,
    alpha=0.3
)

plt.axvline(
    0,
    linewidth=0.8,
    alpha=0.3
)

plt.xlabel(
    r"$X_{\rm GC}$ (kpc)"
)

plt.ylabel(
    r"$Y_{\rm GC}$ (kpc)"
)

plt.title(
    "Face-on Spatial Distribution of "
    "Milky Way Globular Clusters"
)

plt.axis("equal")
plt.grid(alpha=0.25)
plt.legend()

plt.tight_layout()
plt.show()


# EDGE-ON VIEW
# Cylindrical radius vs signed Z

plt.figure(figsize=(9, 7))

plt.scatter(
    spatial["R_cyl"],
    spatial["Z_gc"],
    alpha=0.65,
    label="Globular clusters"
)

# Galactic plane
plt.axhline(
    0,
    linewidth=1,
    linestyle="--",
    alpha=0.5,
    label="Galactic plane"
)

# Solar Galactocentric radius
plt.axvline(
    R0,
    linewidth=1,
    linestyle=":",
    alpha=0.4,
    label="Solar radius"
)

plt.xlabel(
    r"$R_{\rm cyl}$ (kpc)"
)

plt.ylabel(
    r"$Z_{\rm GC}$ (kpc)"
)

plt.title(
    "Edge-on Spatial Distribution of "
    "Milky Way Globular Clusters"
)

plt.grid(alpha=0.25)
plt.legend()

plt.tight_layout()
plt.show()

# GALACTOCENTRIC RADIUS VS HEIGHT

plt.figure(figsize=(9, 7))

plt.scatter(
    spatial["R_gc"],
    spatial["abs_Z"],
    alpha=0.65
)

plt.xlabel(
    r"$R_{\rm gc}$ (kpc)"
)

plt.ylabel(
    r"$|Z|$ (kpc)"
)

plt.title(
    "Galactocentric Radius vs Distance from Galactic Plane"
)

plt.grid(alpha=0.25)

plt.tight_layout()
plt.show()

# MOST DISTANT CLUSTERS FROM GALACTIC CENTRE

columns = [
    "ID",
    "Name",
    "R_gc",
    "R_cyl",
    "Z_gc",
    "abs_Z",
    "R_gc_percentile",
    "R_cyl_percentile",
    "Z_percentile"
]

print(
    "\nClusters with largest Galactocentric radii:"
)

print(
    spatial[
        columns
    ]
    .sort_values(
        "R_gc",
        ascending=False
    )
    .head(10)
    .round(2)
    .to_string(index=False)
)

# CLUSTERS FURTHEST FROM GALACTIC PLANE

print(
    "\nClusters with largest |Z|:"
)

print(
    spatial[
        columns
    ]
    .sort_values(
        "abs_Z",
        ascending=False
    )
    .head(10)
    .round(2)
    .to_string(index=False)
)

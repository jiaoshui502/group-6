from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Data
CURRENT_FOLDER = Path(__file__).resolve().parent

harris1 = pd.read_csv(
    CURRENT_FOLDER / "HarrisPartI.csv"
)


# Keep only clusters with complete spatial information
spatial_data = harris1.dropna(
    subset=["X", "Y", "Z", "R_gc"]
).copy()


print(
    "Number of clusters used:",
    len(spatial_data)
)

# The X, Y, Z coordinates follow the coordinate convention
# described in the Harris Galactic Globular Cluster Catalogue.
# In the Harris system:
#   - the Sun is at (0, 0, 0)
#   - X points toward the Galactic Centre
#   - Y points in the direction of Galactic rotation
#   - Z points toward the North Galactic Pole
#   - the Galactic Centre is at approximately (8.0, 0, 0) kpc
# Therefore, to express the positions relative to the
# Galactic Centre, the coordinate origin is translated by
# 8.0 kpc along the X direction.
# Reference:
# Harris Galactic Globular Cluster Catalogue

R0_HARRIS = 8.0


spatial_data["X_gc"] = (
    spatial_data["X"]
    - R0_HARRIS
)

spatial_data["Y_gc"] = (
    spatial_data["Y"]
)

spatial_data["Z_gc"] = (
    spatial_data["Z"]
)

# Spatial quantities

# Cylindrical distance from the Galactic rotation axis
spatial_data["R_cyl"] = np.sqrt(
    spatial_data["X_gc"]**2
    + spatial_data["Y_gc"]**2
)


# Height above/below the Galactic plane
spatial_data["abs_Z"] = np.abs(
    spatial_data["Z_gc"]
)


# Galactocentric radius calculated from XYZ
# This should approximately reproduce the Harris R_gc values
spatial_data["R_gc_from_xyz"] = np.sqrt(
    spatial_data["X_gc"]**2
    + spatial_data["Y_gc"]**2
    + spatial_data["Z_gc"]**2
)


# Check consistency with catalogue R_gc
spatial_data["R_gc_difference"] = (
    spatial_data["R_gc_from_xyz"]
    - spatial_data["R_gc"]
)

print(
    "Maximum |R_gc calculated - R_gc catalogue|:",
    np.abs(
        spatial_data["R_gc_difference"]
    ).max(),
    "kpc"
)


# Galactic X-Y distribution
# View looking down onto the Galactic plane
# Galactic Centre is now at (0, 0)

plt.figure(figsize=(8, 8))

plt.scatter(
    spatial_data["X_gc"],
    spatial_data["Y_gc"],
    alpha=0.7,
    label="Globular clusters"
)


# Galactic Centre
plt.scatter(
    0,
    0,
    marker="*",
    s=180,
    label="Galactic Centre"
)


plt.axhline(
    0,
    linewidth=0.8,
    alpha=0.4
)

plt.axvline(
    0,
    linewidth=0.8,
    alpha=0.4
)


plt.xlabel(
    r"$X_{\rm GC}$ (kpc)"
)

plt.ylabel(
    r"$Y_{\rm GC}$ (kpc)"
)

plt.title(
    "Spatial Distribution of Galactic Globular Clusters"
)

plt.axis("equal")

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()

# Galactocentric radius vs vertical height
# R_gc already measures total distance from Galactic Centre.
# |Z| measures vertical distance from Galactic plane.

plt.figure(figsize=(9, 7))

plt.scatter(
    spatial_data["R_gc"],
    spatial_data["abs_Z"],
    alpha=0.7
)


plt.xlabel(
    r"$R_{\rm gc}$ (kpc)"
)

plt.ylabel(
    r"$|Z|$ (kpc)"
)

plt.title(
    "Galactocentric Radius vs Height Above the Galactic Plane"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()


# Cylindrical radius vs vertical height
# R_cyl measures distance within the Galactic plane.
# |Z| measures distance perpendicular to the Galactic plane.
# These two quantities give a clearer picture of the
# three-dimensional spatial distribution.

plt.figure(figsize=(9, 7))

plt.scatter(
    spatial_data["R_cyl"],
    spatial_data["abs_Z"],
    alpha=0.7
)


plt.xlabel(
    r"$R_{\rm cyl}=\sqrt{X_{\rm GC}^2+Y_{\rm GC}^2}$ (kpc)"
)

plt.ylabel(
    r"$|Z|$ (kpc)"
)

plt.title(
    "Cylindrical Galactic Position of Globular Clusters"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.show()


# Clusters with largest Galactocentric radii

print()

print(
    "Clusters with largest Galactocentric radii:"
)

print(
    spatial_data[
        [
            "ID",
            "Name",
            "X_gc",
            "Y_gc",
            "Z_gc",
            "R_gc",
            "R_cyl",
            "abs_Z"
        ]
    ]
    .sort_values(
        "R_gc",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)

#  Clusters furthest from the Galactic plane

print()

print(
    "Clusters with largest |Z|:"
)

print(
    spatial_data[
        [
            "ID",
            "Name",
            "X_gc",
            "Y_gc",
            "Z_gc",
            "R_gc",
            "R_cyl",
            "abs_Z"
        ]
    ]
    .sort_values(
        "abs_Z",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)

# Clusters with largest cylindrical radii

print()

print(
    "Clusters with largest cylindrical radii:"
)

print(
    spatial_data[
        [
            "ID",
            "Name",
            "X_gc",
            "Y_gc",
            "Z_gc",
            "R_gc",
            "R_cyl",
            "abs_Z"
        ]
    ]
    .sort_values(
        "R_cyl",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)
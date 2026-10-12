# HB morphology vs Age analysis
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import spearmanr

# Idea:
# At a fixed [Fe/H], a cluster with a redder horizontal branch (HB) is
# usually younger ("age as the second parameter"). If this is true for our
# clusters, the clusters that are younger than the age-metallicity trend
# should also be redder than the HB-metallicity trend, so the two
# residuals should be correlated.
#
# This file runs on its own. The age trend is found in the same way as in
# stellar_population_method.py, and the HB model is the same as in
# HB_metallicity_analysis.py.


# data

folder = Path(__file__).resolve().parent

vandenberg = pd.read_csv(
    folder / "vandenBerg_table2.csv"
)


# Age-metallicity trend
# Straight-line fit, remove clusters more than 3 sigma from the line
# (only Pal 12), then fit again without them.

slope, intercept = np.polyfit(
    vandenberg["FeH"],
    vandenberg["Age"],
    1
)

age_resid = vandenberg["Age"] - (slope * vandenberg["FeH"] + intercept)
kept = age_resid.abs() < 3 * age_resid.std()

slope, intercept = np.polyfit(
    vandenberg.loc[kept, "FeH"],
    vandenberg.loc[kept, "Age"],
    1
)

vandenberg["Age_fit"] = (
    slope * vandenberg["FeH"]
    + intercept
)

print(f"Age trend: Age = {slope:.2f} [Fe/H] + {intercept:.2f} Gyr")
print("Removed from the age fit:", vandenberg.loc[~kept, "Name"].tolist())


# HB morphology model
# A straight line between [Fe/H] = -1.6 and -1.0, joined smoothly to
# exponential tails that level off at HBtype = +1 (metal-poor, blue HB)
# and HBtype = -1 (metal-rich, red HB).

hb_data = vandenberg.dropna(
    subset=["FeH", "HBtype"]
).copy()

left_break = -1.6
right_break = -1.0

# Boundaries for the change of scatter between regions
left_outer = -2.2
right_outer = -0.5

# Robust straight-line fit in the middle region,
# so strong outliers have less influence
hb_mid = hb_data[
    (hb_data["FeH"] >= left_break)
    & (hb_data["FeH"] <= right_break)
]

x_mid = hb_mid["FeH"].to_numpy()
y_mid = hb_mid["HBtype"].to_numpy()

m0, b0 = np.polyfit(x_mid, y_mid, 1)


def line_residual(params):
    m, b = params
    return y_mid - (m * x_mid + b)


fit = least_squares(
    line_residual,
    [m0, b0],
    loss="soft_l1",
    f_scale=0.20
)

m, b = fit.x

# Tails chosen so the curve and its slope are continuous at the breaks
HB_left = m * left_break + b
HB_right = m * right_break + b

tau_left = -(1 - HB_left) / m
tau_right = -(HB_right + 1) / m


def hb_model(feh):

    feh = np.asarray(feh, dtype=float)
    result = np.empty_like(feh)

    left = feh < left_break
    middle = (feh >= left_break) & (feh <= right_break)
    right = feh > right_break

    # Metal-poor side -> approaches +1
    result[left] = 1 - (1 - HB_left) * np.exp(
        (feh[left] - left_break) / tau_left
    )

    # Middle linear relation
    result[middle] = m * feh[middle] + b

    # Metal-rich side -> approaches -1
    result[right] = -1 + (HB_right + 1) * np.exp(
        -(feh[right] - right_break) / tau_right
    )

    return result


hb_data["HB_fit"] = hb_model(hb_data["FeH"])

hb_data["HB_residual"] = (
    hb_data["HBtype"]
    - hb_data["HB_fit"]
)


# Scatter of the HB residuals in each metallicity region

def robust_sigma(values):

    values = np.asarray(values, dtype=float)
    centre = np.median(values)
    sigma = 1.4826 * np.median(np.abs(values - centre))

    # Fallback if MAD = 0
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = np.std(values, ddof=1)

    return sigma


def bulk_sigma(values, clip_limit=2.5):

    values = np.asarray(values, dtype=float)
    centre = np.median(values)
    initial_sigma = robust_sigma(values)

    # Remove only obvious outliers once
    bulk = values[np.abs(values - centre) <= clip_limit * initial_sigma]

    if len(bulk) >= 2:
        sigma = np.std(bulk, ddof=1)
    else:
        sigma = initial_sigma

    # Avoid zero sigma
    return max(sigma, 1e-6)


left_sigma = bulk_sigma(
    hb_data.loc[hb_data["FeH"] < left_break, "HB_residual"]
)

middle_sigma = bulk_sigma(
    hb_data.loc[
        (hb_data["FeH"] >= left_break) & (hb_data["FeH"] <= right_break),
        "HB_residual"
    ]
)

right_sigma = bulk_sigma(
    hb_data.loc[hb_data["FeH"] > right_break, "HB_residual"]
)


# Scatter that changes smoothly with metallicity between the regions

def smoothstep(t):
    t = np.clip(t, 0, 1)
    return 3 * t**2 - 2 * t**3


def dynamic_sigma(feh):

    feh = np.asarray(feh, dtype=float)
    sigma = np.empty_like(feh)

    # Far metal-poor
    sigma[feh <= left_outer] = left_sigma

    # left_sigma -> middle_sigma
    left_transition = (feh > left_outer) & (feh < left_break)
    t_left = (feh[left_transition] - left_outer) / (left_break - left_outer)
    sigma[left_transition] = (
        left_sigma + (middle_sigma - left_sigma) * smoothstep(t_left)
    )

    # Middle region
    sigma[(feh >= left_break) & (feh <= right_break)] = middle_sigma

    # middle_sigma -> right_sigma
    right_transition = (feh > right_break) & (feh < right_outer)
    t_right = (feh[right_transition] - right_break) / (right_outer - right_break)
    sigma[right_transition] = (
        middle_sigma + (right_sigma - middle_sigma) * smoothstep(t_right)
    )

    # Far metal-rich
    sigma[feh >= right_outer] = right_sigma

    return sigma


# HB outliers: more than 1.5 sigma from the HB model

hb_data["HB_sigma"] = dynamic_sigma(hb_data["FeH"])

hb_data["HB_norm_residual"] = (
    hb_data["HB_residual"]
    / hb_data["HB_sigma"]
)

hb_data["HB_outlier"] = (
    np.abs(hb_data["HB_norm_residual"])
    > 1.5
)


# Keep only clusters with all measurements required
# for the Age-HB residual analysis

required_columns = [
    "Age",
    "Age_err",
    "FeH",
    "HBtype",
    "Age_fit",
    "HB_fit",
    "HB_residual",
    "HB_norm_residual",
    "HB_outlier"
]

hb_age_data = hb_data.dropna(
    subset=required_columns
).copy()
# Age residual:
# positive -> older than expected at its metallicity
# negative -> younger than expected at its metallicity

hb_age_data["Age_residual"] = (
    hb_age_data["Age"]
    - hb_age_data["Age_fit"]
)
# HB residual already comes from the previous
# HB morphology-metallicity analysis:
#
# positive -> bluer than expected at its metallicity
# negative -> redder than expected at its metallicity

hb_age_data["HB_age_residual"] = (
    hb_age_data["HB_residual"]
)

print("Number of clusters used:", len(hb_age_data))


# Raw HB morphology vs Age

plt.figure(figsize=(8, 6))

plt.errorbar(
    hb_age_data["Age"],
    hb_age_data["HBtype"],
    xerr=hb_age_data["Age_err"],
    fmt="o",
    alpha=0.7,
    capsize=3
)

plt.xlabel("Age (Gyr)")
plt.ylabel("HB type")
plt.title("HB Morphology vs Cluster Age")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# Spearman correlation:
#    Age residual vs HB residual

rho_all, p_all = spearmanr(
    hb_age_data["Age_residual"],
    hb_age_data["HB_age_residual"]
)

print()
print("Spearman correlation - all clusters")
print("rho =", rho_all)
print("p-value =", p_all)
# Test whether the relation is driven by HB outliers

normal_hb = hb_age_data[
    hb_age_data["HB_outlier"] == False
].copy()

hb_age_outliers = hb_age_data[
    hb_age_data["HB_outlier"] == True
].copy()


rho_normal, p_normal = spearmanr(
    normal_hb["Age_residual"],
    normal_hb["HB_age_residual"]
)

print()
print("Spearman correlation - excluding HB outliers")
print("rho =", rho_normal)
print("p-value =", p_normal)

# Age residual vs HB residual
plt.figure(figsize=(9, 7))
# Normal clusters
plt.errorbar(
    normal_hb["Age_residual"],
    normal_hb["HB_age_residual"],
    xerr=normal_hb["Age_err"],
    fmt="o",
    alpha=0.65,
    capsize=2,
    label="Normal clusters"
)


# HB outliers
plt.errorbar(
    hb_age_outliers["Age_residual"],
    hb_age_outliers["HB_age_residual"],
    xerr=hb_age_outliers["Age_err"],
    fmt="x",
    markersize=10,
    markeredgewidth=2,
    capsize=3,
    label="HB outliers"
)


# Reference lines
plt.axvline(
    0,
    linestyle="--",
    linewidth=1
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

# Label HB outliers

for _, row in hb_age_outliers.iterrows():

    ngc = str(row["#NGC"])
    name = str(row["Name"])

    if ngc != "XXXX":

        label = "NGC " + ngc

        if name != "XXXX":
            label += " / " + name

    else:
        label = name


    plt.annotate(
        label,
        (
            row["Age_residual"],
            row["HB_age_residual"]
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )


plt.xlabel(
    r"$\Delta$Age = Age - Age$_{\rm fit}$ (Gyr)"
)

plt.ylabel(
    r"$\Delta$HB = HBtype - HB$_{\rm fit}$"
)

plt.title(
    "Age Residual vs HB Morphology Residual"
)


# Show both correlation results
plt.text(
    0.03,
    0.03,
    (
        f"All clusters: "
        f"$\\rho$ = {rho_all:.2f}, "
        f"p = {p_all:.2e}\n"
        f"Without HB outliers: "
        f"$\\rho$ = {rho_normal:.2f}, "
        f"p = {p_normal:.2e}"
    ),
    transform=plt.gca().transAxes,
    verticalalignment="bottom"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Print HB outliers with Age information

columns_to_show = [
    "#NGC",
    "Name",
    "FeH",
    "Age",
    "Age_err",
    "Age_fit",
    "Age_residual",
    "HBtype",
    "HB_fit",
    "HB_age_residual",
    "HB_norm_residual"
]

print()
print("HB outliers in the Age-HB analysis")

print(
    hb_age_outliers[
        columns_to_show
    ]
    .sort_values(
        "HB_age_residual",
        key=abs,
        ascending=False
    )
    .to_string(index=False)
)

# Residual distance
# It measures how far each cluster lies from the origin
# in the Age-residual / HB-residual plane.
# Large residual_distance can result from:
#   - large Age residual
#   - large HB residual
#   - or both


age_std = hb_age_data["Age_residual"].std()
hb_std = hb_age_data["HB_age_residual"].std()
hb_age_data["Age_residual_norm"] = (
    hb_age_data["Age_residual"]
    / age_std
)
hb_age_data["HB_residual_norm"] = (
    hb_age_data["HB_age_residual"]
    / hb_std
)
hb_age_data["residual_distance"] = np.sqrt(
    hb_age_data["Age_residual_norm"]**2
    +
    hb_age_data["HB_residual_norm"]**2
)
print()
print(
    "Clusters farthest from the origin "
    "in Age-HB residual space"
)
print(
    hb_age_data[
        [
            "#NGC",
            "Name",
            "FeH",
            "Age_residual",
            "HB_age_residual",
            "residual_distance"
        ]
    ]
    .sort_values(
        "residual_distance",
        ascending=False
    )
    .head(10)
    .to_string(index=False)
)

# Quadrant analysis
# Q1:
# older + bluer than expected
q1 = hb_age_data[
    (hb_age_data["Age_residual"] > 0)
    &
    (hb_age_data["HB_age_residual"] > 0)
]
# Q2:
# younger + bluer than expected
q2 = hb_age_data[
    (hb_age_data["Age_residual"] < 0)
    &
    (hb_age_data["HB_age_residual"] > 0)
]
# Q3:
# younger + redder than expected
q3 = hb_age_data[
    (hb_age_data["Age_residual"] < 0)
    &
    (hb_age_data["HB_age_residual"] < 0)
]
# Q4:
# older + redder than expected
q4 = hb_age_data[
    (hb_age_data["Age_residual"] > 0)
    &
    (hb_age_data["HB_age_residual"] < 0)
]


print()
print("Quadrant counts")

print(
    "Older + bluer than expected:",
    len(q1)
)

print(
    "Younger + bluer than expected:",
    len(q2)
)

print(
    "Younger + redder than expected:",
    len(q3)
)

print(
    "Older + redder than expected:",
    len(q4)
)

# Age residual vs normalised HB residual

plt.figure(figsize=(9, 7))
# Normal clusters
plt.errorbar(
    normal_hb["Age_residual"],
    normal_hb["HB_norm_residual"],
    xerr=normal_hb["Age_err"],
    fmt="o",
    alpha=0.65,
    capsize=2,
    label="Normal clusters"
)

# HB outliers
plt.errorbar(
    hb_age_outliers["Age_residual"],
    hb_age_outliers["HB_norm_residual"],
    xerr=hb_age_outliers["Age_err"],
    fmt="x",
    markersize=10,
    markeredgewidth=2,
    capsize=3,
    label="HB outliers"
)


# Age residual reference line
plt.axvline(
    0,
    linestyle="--",
    linewidth=1
)


# HB outlier thresholds
plt.axhline(
    1.5,
    linestyle="--",
    linewidth=1,
    label=r"HB outlier threshold ($\pm1.5\sigma$)"
)

plt.axhline(
    -1.5,
    linestyle="--",
    linewidth=1
)


# Label HB outliers
for _, row in hb_age_outliers.iterrows():

    ngc = str(row["#NGC"])
    name = str(row["Name"])

    if ngc != "XXXX":

        label = "NGC " + ngc

        if name != "XXXX":
            label += " / " + name

    else:
        label = name


    plt.annotate(
        label,
        (
            row["Age_residual"],
            row["HB_norm_residual"]
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )


plt.xlabel(
    r"$\Delta$Age = Age - Age$_{\rm fit}$ (Gyr)"
)

plt.ylabel(
    r"Normalised HB residual "
    r"$=(HBtype-HB_{\rm fit})/\sigma_{\rm HB}$"
)

plt.title(
    "Age Residual vs Normalised HB Morphology Residual"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
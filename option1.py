from pathlib import Path
from scipy.stats import spearmanr
from scipy.optimize import least_squares
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data input
# testing GitHub
CURRENT_FLODER = Path(__file__).resolve().parent
harris1 = pd.read_csv(CURRENT_FLODER / "HarrisPartI.csv")
harris3 = pd.read_csv(CURRENT_FLODER / "HarrisPartIII.csv")
krause = pd.read_csv(CURRENT_FLODER / "Krause21.csv")
vandenberg = pd.read_csv(CURRENT_FLODER / "vandenBerg_table2.csv")

# checking

print("Harris Part I:", harris1.shape)
print("Harris Part III:", harris3.shape)
print("Krause21:", krause.shape)
print("van den Berg:", vandenberg.shape)

# ploting the graph for Fe/H VS Age

plt.figure(figsize=(10, 6))

plt.errorbar(
    vandenberg["FeH"],
    vandenberg["Age"],
    yerr=vandenberg["Age_err"],
    fmt="o",
    capsize=3
)

plt.xlabel("[Fe/H]")
plt.ylabel("Age")
plt.title("Age-Metallicity Relation of Milky Way Globular Clusters")

plt.tight_layout()
plt.show()

# Linear fit exclude Pal 12 (Because it's an outlier obviously)

# Remove Pal 12 only from the fitting sample
fit_data = vandenberg[
    vandenberg["Name"] != "Pal12"
].copy()

# Linear fit: Age = slope * FeH + intercept
slope, intercept = np.polyfit(
    fit_data["FeH"],
    fit_data["Age"],
    1
)

print("Slope:", slope)
print("Intercept:", intercept)

# Generate points for the fitted line
x_fit = np.linspace(
    vandenberg["FeH"].min(),
    vandenberg["FeH"].max(),
    100
)

y_fit = slope * x_fit + intercept

# Plot all clusters
plt.figure(figsize=(10, 6))

plt.errorbar(
    vandenberg["FeH"],
    vandenberg["Age"],
    yerr=vandenberg["Age_err"],
    fmt="o",
    capsize=3
)

# Plot fit based on all clusters except Pal 12
plt.plot(
    x_fit,
    y_fit,
    label="Linear fit excluding Pal 12"
)

plt.xlabel("[Fe/H]")
plt.ylabel("Age")
plt.title("Age–Metallicity Relation")
plt.legend()

plt.tight_layout()
plt.show()

# Calculate fitted age for all clusters
vandenberg["Age_fit"] = (
    slope * vandenberg["FeH"] + intercept
)

# Difference between observed age and fitted age
vandenberg["Age_residual"] = (
    vandenberg["Age"] - vandenberg["Age_fit"]
)

# Select clusters whose error bars doesn't cross the fitted line
outside_fit = vandenberg[
    np.abs(vandenberg["Age_residual"]) > vandenberg["Age_err"]
].copy()

print("Clusters whose age error bars do not include the fitted line:")
print(
    outside_fit[
        ["#NGC", "Name", "FeH", "Age",
         "Age_err", "Age_fit", "Age_residual"]
    ].sort_values("Age_residual")
)

# HBtype vs [Fe/H]

hb_data = vandenberg.dropna(subset=["FeH", "HBtype"]).copy()

plt.figure(figsize=(10, 6))

plt.scatter(
    hb_data["FeH"],
    hb_data["HBtype"],
    alpha=0.7,
    label="All clusters"
)

plt.xlabel("[Fe/H]")
plt.ylabel("HBtype")
plt.title("Horizontal Branch Morphology vs Metallicity")

plt.legend()
plt.tight_layout()
plt.show()

# HBtype vs [Fe/H] with young age-metallicity outliers highlighted

plt.figure(figsize=(10, 6))

# All clusters
plt.scatter(
    hb_data["FeH"],
    hb_data["HBtype"],
    alpha=0.6,
    label="All clusters"
)

# Highlight younger-than-fit outliers

age_outliers = outside_fit.copy()

plt.scatter(
    age_outliers["FeH"],
    age_outliers["HBtype"],
    marker="x",
    s=100,
    label="age_outliers"
)

plt.xlabel("[Fe/H]")
plt.ylabel("HBtype")
plt.title("HBtype vs [Fe/H]")

plt.legend()
plt.tight_layout()
plt.show()

# HB morphology analysis

# Model boundaries
left_break = -1.6
right_break = -1.0

# Boundaries for dynamic sigma transition
left_outer = -2.2
right_outer = -0.5

# Robust linear fit in the transition region

hb_mid = hb_data[(hb_data["FeH"] >= left_break) &(hb_data["FeH"] <= right_break)].copy()

x_mid = hb_mid["FeH"].to_numpy()
y_mid = hb_mid["HBtype"].to_numpy()

# Initial fit
m0, b0 = np.polyfit(x_mid, y_mid, 1)


def line_residual(params):
    m, b = params
    return y_mid - (m * x_mid + b)


# Robust fit so strong outliers have less influence
fit = least_squares(
    line_residual,
    [m0, b0],
    loss="soft_l1",
    f_scale=0.20
)

m, b = fit.x

print("HB middle fit:")
print("Slope:", m)
print("Intercept:", b)

# Exponential-Linear-Exponential model

HB_left = m * left_break + b
HB_right = m * right_break + b

# Choose tau so both the function and slope are continuous
tau_left = -(1 - HB_left) / m
tau_right = -(HB_right + 1) / m


def hb_model(feh):

    feh = np.asarray(feh, dtype=float)
    result = np.empty_like(feh)

    left = feh < left_break

    middle = ((feh >= left_break) &(feh <= right_break))

    right = feh > right_break

    # Metal-poor side -> approaches +1
    result[left] = (1- (1 - HB_left)* np.exp((feh[left] - left_break) / tau_left))

    # Middle linear relation
    result[middle] = (m * feh[middle] + b)

    # Metal-rich side -> approaches -1
    result[right] = (-1+ (HB_right + 1)* np.exp(-(feh[right] - right_break) / tau_right))

    return result

# Calculate expected HBtype and residual
hb_data["HB_fit"] = hb_model(hb_data["FeH"])

hb_data["HB_residual"] = (hb_data["HBtype"] - hb_data["HB_fit"])

# Estimate bulk scatter

def robust_sigma(values):

    values = np.asarray(values, dtype=float)

    centre = np.median(values)

    mad = np.median(
        np.abs(values - centre)
    )

    sigma = 1.4826 * mad

    # Fallback if MAD = 0
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = np.std(values, ddof=1)

    return sigma


def bulk_sigma(values, clip_limit=2.5):

    values = np.asarray(values, dtype=float)

    centre = np.median(values)
    initial_sigma = robust_sigma(values)

    # Remove only obvious outliers once
    keep = (
        np.abs(values - centre)
        <= clip_limit * initial_sigma
    )

    bulk = values[keep]

    if len(bulk) >= 2:
        sigma = np.std(bulk, ddof=1)
    else:
        sigma = initial_sigma

    # Avoid zero sigma
    return max(sigma, 1e-6)


# Residuals in each broad metallicity region
left_residuals = hb_data.loc[
    hb_data["FeH"] < left_break,
    "HB_residual"
]

middle_residuals = hb_data.loc[
    (hb_data["FeH"] >= left_break) &
    (hb_data["FeH"] <= right_break),
    "HB_residual"
]

right_residuals = hb_data.loc[
    hb_data["FeH"] > right_break,
    "HB_residual"
]


left_sigma = bulk_sigma(left_residuals)
middle_sigma = bulk_sigma(middle_residuals)
right_sigma = bulk_sigma(right_residuals)

print("\nHB regional scatter:")
print("Left sigma:", left_sigma)
print("Middle sigma:", middle_sigma)
print("Right sigma:", right_sigma)

# Changing sigma

def smoothstep(t):
    t = np.clip(t, 0, 1)
    return (3 * t**2 - 2 * t**3)

def dynamic_sigma(feh):

    feh = np.asarray(feh, dtype=float)
    sigma = np.empty_like(feh)

    # Far metal-poor
    far_left = feh <= left_outer
    sigma[far_left] = left_sigma


    # left_sigma -> middle_sigma
    left_transition = ((feh > left_outer) &(feh < left_break))

    t_left = ((feh[left_transition] - left_outer)/(left_break - left_outer))

    sigma[left_transition] = (left_sigma+ (middle_sigma - left_sigma)* smoothstep(t_left))


    # Middle region
    middle = ((feh >= left_break) &(feh <= right_break))

    sigma[middle] = middle_sigma

    # middle_sigma -> right_sigma
    right_transition = ((feh > right_break) &(feh < right_outer))

    t_right = ((feh[right_transition] - right_break)/(right_outer - right_break))

    sigma[right_transition] = (middle_sigma + (right_sigma - middle_sigma) * smoothstep(t_right))

    # Far metal-rich
    far_right = feh >= right_outer
    sigma[far_right] = right_sigma

    return sigma

# Sigma corresponding to each cluster
hb_data["HB_sigma"] = dynamic_sigma(hb_data["FeH"])

# Identify HB morphology outliers

hb_data["HB_norm_residual"] = (hb_data["HB_residual"]/hb_data["HB_sigma"])

sigma_limit = 1.5

hb_data["HB_outlier"] = (
    np.abs(hb_data["HB_norm_residual"])
    > sigma_limit
)

hb_outliers = hb_data[
    hb_data["HB_outlier"]
].copy()


print("\nHB morphology outliers:")

print(
    hb_outliers[
        [
            "#NGC",
            "Name",
            "FeH",
            "HBtype",
            "HB_fit",
            "HB_sigma",
            "HB_norm_residual"
        ]
    ]
    .sort_values("FeH")
    .to_string(index=False)
)

# Smooth curves for plotting

x_plot = np.linspace(
    hb_data["FeH"].min(),
    hb_data["FeH"].max(),
    800
)

y_plot = hb_model(x_plot)
sigma_plot = dynamic_sigma(x_plot)

upper = np.clip( y_plot + sigma_limit * sigma_plot , -1 , 1 )

lower = np.clip( y_plot - sigma_limit * sigma_plot , -1 , 1 )

# HB morphology plot

plt.figure(figsize=(11, 7))

# All clusters
plt.scatter(
    hb_data["FeH"],
    hb_data["HBtype"],
    alpha=0.55,
    label="All clusters"
)

# Expected HB relation
plt.plot(
    x_plot,
    y_plot,
    linewidth=2.5,
    label="Exponential-Linear-Exponential model"
)

# Dynamic 1.5 sigma region
plt.fill_between(
    x_plot,
    lower,
    upper,
    alpha=0.20,
    label="Dynamic ±1.5σ region"
)

# HB outliers
plt.scatter(
    hb_outliers["FeH"],
    hb_outliers["HBtype"],
    marker="x",
    s=120,
    linewidths=2,
    label="HB outliers (> 1.5σ)"
)


# Label outliers
offsets = [
    (7, 8),
    (7, -15),
    (-58, 8),
    (-58, -15)
]

for i, (_, row) in enumerate(
    hb_outliers.iterrows()
):

    ngc = str(row["#NGC"]).strip()

    if ngc in ["", "nan", "XXXX"]:
        label = str(row["Name"])

    else:
        try:
            label = "NGC " + str(
                int(float(ngc))
            )
        except ValueError:
            label = "NGC " + ngc

    plt.annotate(
        label,
        (row["FeH"], row["HBtype"]),
        xytext=offsets[i % 4],
        textcoords="offset points",
        fontsize=8
    )


# Show model/scatter boundaries
plt.axvline(
    left_outer,
    linestyle=":",
    alpha=0.20
)

plt.axvline(
    left_break,
    linestyle="--",
    alpha=0.25
)

plt.axvline(
    right_break,
    linestyle="--",
    alpha=0.25
)

plt.axvline(
    right_outer,
    linestyle=":",
    alpha=0.20
)


plt.xlabel("[Fe/H]")
plt.ylabel("HBtype")

plt.title(
    "Horizontal Branch Morphology vs Metallicity"
)

plt.ylim(-1.1, 1.1)

plt.legend()
plt.tight_layout()
plt.show()

# Changing sigma plot

plt.figure(figsize=(10, 4))

plt.plot(
    x_plot,
    sigma_plot,
    linewidth=2
)

plt.axvline(
    left_outer,
    linestyle=":",
    alpha=0.25
)

plt.axvline(
    left_break,
    linestyle="--",
    alpha=0.25
)

plt.axvline(
    right_break,
    linestyle="--",
    alpha=0.25
)

plt.axvline(
    right_outer,
    linestyle=":",
    alpha=0.25
)

plt.xlabel("[Fe/H]")
plt.ylabel("Local sigma")

plt.title(
    "Metallicity-dependent HB Scatter"
)

plt.tight_layout()
plt.show()

# HB morphology vs Age analysis
# Data

# Keep only clusters with all required measurements
hb_age_data = hb_data.dropna(
    subset=[
        "Age",
        "Age_err",
        "FeH",
        "HBtype",
        "Age_fit",
        "HB_fit",
        "HB_residual",
        "HB_outlier"
    ]
).copy()


# Age residual:
# positive -> older than expected at its metallicity
# negative -> younger than expected at its metallicity
hb_age_data["Age_residual"] = ( hb_age_data["Age"] - hb_age_data["Age_fit"] )

# HB residual already comes from the HB-metallicity analysis:
# positive -> bluer than expected at its metallicity
# negative -> redder than expected at its metallicity
hb_age_data["HB_age_residual"] = ( hb_age_data["HB_residual"])

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
# Age residual vs HB residual

rho_all, p_all = spearmanr(
    hb_age_data["Age_residual"],
    hb_age_data["HB_age_residual"]
)

print()
print("Spearman correlation - all clusters")
print("rho =", rho_all)
print("p-value =", p_all)

# Test whether the correlation is driven mainly by HB outliers

normal_hb = hb_age_data[
    hb_age_data["HB_outlier"] == False
].copy()

rho_normal, p_normal = spearmanr(
    normal_hb["Age_residual"],
    normal_hb["HB_age_residual"]
)
print()
print("Spearman correlation - excluding HB outliers")
print("---------------------------------------------")
print("rho =", rho_normal)
print("p-value =", p_normal)

# Residual plot highlighting HB outliers

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
# HB age outliers

hb_age_outliers = hb_age_data[
    hb_age_data["HB_outlier"] == True
].copy()

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
# Refrence line
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

# 6. Label outliers

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

# Show both correlation results on the figure
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
    "HB_age_residual"
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
# This is NOT a new outlier criterion.
# It only measures how far each cluster lies from the origin
# of the Age-residual / HB-residual plane.
# A large value can be caused by:
#     large Age residual,
#     large HB residual,
#     or both.

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
print("Clusters farthest from the origin in Age-HB residual space")

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

# Optional quadrant summary
# Q1: older + bluer than expected
# Q2: younger + bluer than expected
# Q3: younger + redder than expected
# Q4: older + redder than expected

q1 = hb_age_data[
    (hb_age_data["Age_residual"] > 0)
    &
    (hb_age_data["HB_age_residual"] > 0)
]

q2 = hb_age_data[
    (hb_age_data["Age_residual"] < 0)
    &
    (hb_age_data["HB_age_residual"] > 0)
]

q3 = hb_age_data[
    (hb_age_data["Age_residual"] < 0)
    &
    (hb_age_data["HB_age_residual"] < 0)
]

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

# Normalised HB residual vs Age residual

plt.figure(figsize=(9, 7))

# Normal clusters
normal_hb = hb_age_data[
    hb_age_data["HB_outlier"] == False
]
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

# Age residual reference
plt.axvline(
    0,
    linestyle="--",
    linewidth=1
)

# Exact HB outlier thresholds
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
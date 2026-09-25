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


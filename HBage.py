# HB morphology vs Age analysis
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# data
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
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setting
AGE_CUT = 2.0
CLIP_CUT = 3.0

folder = Path(__file__).resolve().parent

vdb = pd.read_csv(folder / "vandenBerg_table2.csv")
vdb = vdb.dropna(subset=["FeH", "Age", "Age_err"]).copy()
vdb = vdb[vdb["Age_err"] > 0].reset_index(drop=True)
# Cluster

vdb["ID"] = "NGC " + vdb["#NGC"].astype(str).str.strip()

name_map = {
    "Arp21": "Arp 2",
    "Pal12": "Pal 12",
    "Ter8": "Terzan 8"
}

no_ngc = vdb["#NGC"].astype(str).str.strip() == "XXXX"

vdb.loc[no_ngc, "ID"] = (
    vdb.loc[no_ngc, "Name"]
    .map(name_map)
    .fillna(vdb.loc[no_ngc, "Name"])
)
# functions

def robust_sigma(x):
    """Robust scatter estimate using MAD."""
    x = np.asarray(x)
    med = np.median(x)
    sigma = 1.4826 * np.median(np.abs(x - med))

    if sigma <= 0:
        sigma = np.std(x, ddof=1)

    return sigma


def weighted_fit(x, y, sigma):
    """Weighted linear fit: Age = m*[Fe/H] + b."""
    return np.polyfit(x, y, 1, w=1 / sigma)


# Data arrays

x = vdb["FeH"].to_numpy()
y = vdb["Age"].to_numpy()
age_err = vdb["Age_err"].to_numpy()

keep = np.ones(len(vdb), dtype=bool)
sigma_int = 0.0

#Iterative fit
for _ in range(20):

    sigma_total = np.sqrt(age_err**2 + sigma_int**2)

    slope, intercept = weighted_fit(
        x[keep],
        y[keep],
        sigma_total[keep]
    )

    age_fit = slope * x + intercept
    residual = y - age_fit

    sigma_obs = robust_sigma(residual[keep])
    sigma_meas = np.sqrt(np.mean(age_err[keep]**2))

    new_sigma_int = np.sqrt(
        max(sigma_obs**2 - sigma_meas**2, 0)
    )

    sigma_total = np.sqrt(
        age_err**2 + new_sigma_int**2
    )

    z = residual / sigma_total

    new_keep = np.abs(z) < CLIP_CUT

    converged = (
        np.array_equal(new_keep, keep)
        and abs(new_sigma_int - sigma_int) < 1e-4
    )

    keep = new_keep
    sigma_int = new_sigma_int

    if converged:
        break
#Final fit

sigma_total = np.sqrt(age_err**2 + sigma_int**2)

slope, intercept = weighted_fit(
    x[keep],
    y[keep],
    sigma_total[keep]
)

vdb["Age_fit"] = slope * x + intercept
vdb["Age_residual"] = y - vdb["Age_fit"]

vdb["Age_sigma_total"] = sigma_total

vdb["Age_z"] = (
    vdb["Age_residual"]
    / vdb["Age_sigma_total"]
)


# Classification
vdb["Age_younger"] = vdb["Age_z"] < -AGE_CUT
vdb["Age_older"] = vdb["Age_z"] > AGE_CUT
vdb["Age_standout"] = np.abs(vdb["Age_z"]) > AGE_CUT

younger = vdb[vdb["Age_younger"]].sort_values("Age_z")
older = vdb[vdb["Age_older"]].sort_values("Age_z", ascending=False)
clipped = vdb[~keep].sort_values("Age_z")
# Results

print("\nAge-Metallicity Relation")
print(
    f"Age = {slope:.3f} [Fe/H] + {intercept:.3f} Gyr"
)
print(
    f"Intrinsic scatter = {sigma_int:.3f} Gyr"
)

print("\nExcluded from baseline fit (>3 sigma):")
print(
    clipped[
        ["ID", "FeH", "Age", "Age_z"]
    ].round(2).to_string(index=False)
)

print("\nPotentially accreted candidates (z < -2):")
print(
    younger[
        ["ID", "FeH", "Age", "Age_err",
         "Age_residual", "Age_z"]
    ].round(2).to_string(index=False)
)

print("\nOlder-than-trend outliers (z > 2):")
print(
    older[
        ["ID", "FeH", "Age", "Age_err",
         "Age_residual", "Age_z"]
    ].round(2).to_string(index=False)
)

plt.figure(figsize=(11, 7))

# All clusters with age uncertainties
plt.errorbar(
    vdb["FeH"],
    vdb["Age"],
    yerr=vdb["Age_err"],
    fmt="o",
    capsize=3,
    alpha=0.55,
    label="All clusters",
    zorder=1
)

# Best-fit age-metallicity relation
x_plot = np.linspace(
    vdb["FeH"].min(),
    vdb["FeH"].max(),
    300
)

plt.plot(
    x_plot,
    slope * x_plot + intercept,
    linewidth=2.2,
    label="Weighted age-metallicity trend",
    zorder=2
)

# Younger candidates
plt.scatter(
    younger["FeH"],
    younger["Age"],
    marker="v",
    s=110,
    label="Younger candidates (z < -2)",
    zorder=4
)
# Older outliers
plt.scatter(
    older["FeH"],
    older["Age"],
    marker="^",
    s=110,
    label="Older outliers (z > 2)",
    zorder=4
)
label_offsets = {
    "Pal 12":   (8, 5),

    "NGC 4590": (5, 8),
    "NGC 1851": (7, 12),
    "NGC 362":  (7, -17),
    "NGC 1261": (-8, 10),

    "NGC 104":  (7, 8),
    "NGC 6218": (7, 7),

    "NGC 6397": (-12, 12),
    "NGC 6809": (8, 12),

    "NGC 6723": (8, -15),
    "NGC 6362": (-10, 10)
}

standouts = vdb[vdb["Age_standout"]]

for _, row in standouts.iterrows():

    offset = label_offsets.get(
        row["ID"],
        (6, 6)
    )

    plt.annotate(
        row["ID"],
        (row["FeH"], row["Age"]),
        xytext=offset,
        textcoords="offset points",
        fontsize=8,
        ha="left" if offset[0] >= 0 else "right",
        va="bottom" if offset[1] >= 0 else "top",
        arrowprops=dict(
            arrowstyle="-",
            linewidth=0.5,
            alpha=0.5
        )
    )

plt.xlabel("[Fe/H]")
plt.ylabel("Age (Gyr)")
plt.title(
    "Age-Metallicity Relation of Milky Way Globular Clusters"
)
plt.legend(
    loc="upper right",
    frameon=True
)
plt.tight_layout()
plt.show()
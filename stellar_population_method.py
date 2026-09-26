import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# some flexible settings:

# A cluster stands out if it is more than this many sigma from the trend.
age_cut = 2.0

# When fitting the age trend, first throw away extreme clusters that
# would drag the line towards themselves (more than this many sigma).
clip = 3.0


# Part 1: loading data
folder = Path(__file__).resolve().parent
vdb = pd.read_csv(folder / "vandenBerg_table2.csv") # ages and [Fe/H]

# We give every cluster a readable name (ID) for the plot labels, in the
# style "NGC 104". The table stores only the number "104".
vdb["ID"] = "NGC " + vdb["#NGC"].astype(str).str.strip()

# Three clusters have no NGC number ("XXXX") and their names are spelled
# differently ("Arp21" means Arp 2). We write them by hand.
name_to_id = {"Arp21": "Arp 2", "Pal12": "Pal 12", "Ter8": "Terzan 8"}
no_ngc = vdb["#NGC"].astype(str).str.strip() == "XXXX"
vdb.loc[no_ngc, "ID"] = vdb.loc[no_ngc, "Name"].map(name_to_id)


# Part 2: stellar population: age-metalicity relation
# Method:
#   (a) Fit a straight line  Age = slope * [Fe/H] + intercept
#       through all clusters.
#   (b) Residual = observed age - age predicted by the line.
#       A negative residual means the cluster is younger than the trend.
#   (c) If one cluster is extremely far from the line it pulls the line
#       towards itself and also inflates the scatter. So we remove
#       clusters more than clip sigma away and fit again.
#   (d) sigma = typical size of the residuals of the remaining clusters.
#       Score (z) = residual / sigma = how many sigma from the trend.
#   (e) |z| > AGE_CUT means the cluster stands out.

# (a) first straight-line fit using every cluster
slope, intercept = np.polyfit(vdb["FeH"], vdb["Age"], 1)
vdb["age_resid"] = vdb["Age"] - (slope * vdb["FeH"] + intercept)

# (c) remove extreme clusters and fit again
sigma_first = vdb["age_resid"].std()
kept = vdb["age_resid"].abs() < clip * sigma_first
print("Removed before the final age fit:", vdb.loc[~kept, "ID"].tolist())

slope, intercept = np.polyfit(vdb.loc[kept, "FeH"], vdb.loc[kept, "Age"], 1)
vdb["age_resid"] = vdb["Age"] - (slope * vdb["FeH"] + intercept)

# (d) sigma from the clusters that were kept, then a score for every cluster
age_sigma = vdb.loc[kept, "age_resid"].std()
vdb["age_z"] = vdb["age_resid"] / age_sigma

# (e) some flags
vdb["age_younger"] = vdb["age_z"] < -age_cut
vdb["age_older"] = vdb["age_z"] > age_cut
vdb["age_standout"] = vdb["age_younger"] | vdb["age_older"]

print()
print("Age-metallicity Relation")
print(f"  trend: Age = {slope:.2f} * [Fe/H] + {intercept:.2f}  (Gyr)")
print(f"  typical scatter (sigma) = {age_sigma:.2f} Gyr")
print(f"  clusters that stand out: {vdb['age_standout'].sum()} of {len(vdb)}"
      f"   (about {len(vdb) * math.erfc(age_cut / math.sqrt(2)):.1f} expected by chance)")
print(vdb.loc[vdb["age_standout"], ["ID", "FeH", "Age", "age_z"]]
      .sort_values("age_z").round(2).to_string(index=False))

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
            (xv, yv),                          # the point being labelled
            xytext=(dx, dy),                   # where the text goes
            textcoords="offset points",
            ha="left" if dx > 0 else "right",  # text grows away from the point
            fontsize=8,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
        )


# Assignment task 1: Identify potentially accreted cluester using stellar
# population method
# Rule: candidate that are much younger than the age-metallicity trend.
# We count only younger because accreted clusters tend to be younger
# at a given [Fe/H]. Clusters that are older than the trend still stand
# out in the plot, but they are more likely just old Milky Way clusters,
# so they are not called candidates.

candidates = vdb[vdb["age_younger"]].sort_values("age_z")

print()
print("TASK 1: POTENTIALLY ACCRETED CLUSTERS (younger than the trend)")
print(f"  {len(candidates)} candidates out of {len(vdb)} clusters with an age")
print(candidates[["ID", "FeH", "Age", "age_z"]].round(2).to_string(index=False))


# plot
out = vdb[vdb["age_standout"]].sort_values("FeH") # clusters that stand out

plt.figure(figsize=(10, 6))

# every cluster with its age error bar
plt.errorbar(vdb["FeH"], vdb["Age"], yerr=vdb["Age_err"], fmt="o",
             color="tab:blue", alpha=0.6, capsize=3, label="Clusters")

# the trend line and the band of +-age cut sigma around it
x_line = np.linspace(vdb["FeH"].min(), vdb["FeH"].max(), 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, color="black", label="Trend (straight-line fit)")
plt.fill_between(x_line, y_line - age_cut * age_sigma, y_line + age_cut * age_sigma,
                 color="gray", alpha=0.2, label=f"Within {age_cut:g} sigma of trend")

# clusters outside the band, in red, with names
plt.scatter(out["FeH"], out["Age"], s=90, color="red", zorder=3,
            label=f"Stand out (more than {age_cut:g} sigma)")
label_points(out["FeH"], out["Age"], out["ID"])

plt.xlabel("[Fe/H]")
plt.ylabel("Age (Gyr)")
plt.title("Age-Metallicity Relation of Milky Way Globular Clusters")
plt.legend()
plt.tight_layout()
plt.show()

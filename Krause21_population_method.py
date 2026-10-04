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
krause = pd.read_csv(folder / "Krause21.csv") # ages and [Fe/H], no error bars

# We give every cluster a readable name (ID) for the plot labels, in the
# style "NGC 104", the same style as the Harris catalogue.
# Krause21 writes "NGC104", so we add the space.
krause["ID"] = krause["Object"].str.replace("NGC", "NGC ", regex=False)

# Three clusters have no NGC number and are spelled differently from
# Harris ("Ruprecht106" means Rup 106). We write them by hand.
name_to_id = {"Ruprecht106": "Rup 106", "Terzan7": "Terzan 7", "Palomar12": "Pal 12"}
krause["ID"] = krause["ID"].replace(name_to_id)


# Part 2: stellar population: age-metallicity relation, using Krause21 ages
# This is the same method as stellar_population_method.py, applied to a
# second, independent set of ages (61 clusters instead of 55).
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
#   (e) |z| > age_cut means the cluster stands out.
# Krause21 gives no age error bars, so every cluster counts equally in the
# fit, and sigma mixes measurement error with the real spread of ages.

# (a) first straight-line fit using every cluster
slope, intercept = np.polyfit(krause["FeH"], krause["Age"], 1)
krause["age_resid"] = krause["Age"] - (slope * krause["FeH"] + intercept)

# (c) remove extreme clusters and fit again
sigma_first = krause["age_resid"].std()
kept = krause["age_resid"].abs() < clip * sigma_first
print("Removed before the final age fit:", krause.loc[~kept, "ID"].tolist())

slope, intercept = np.polyfit(krause.loc[kept, "FeH"], krause.loc[kept, "Age"], 1)
krause["age_resid"] = krause["Age"] - (slope * krause["FeH"] + intercept)

# (d) sigma from the clusters that were kept, then a score for every cluster
age_sigma = krause.loc[kept, "age_resid"].std()
krause["age_z"] = krause["age_resid"] / age_sigma

# (e) some flags
krause["age_younger"] = krause["age_z"] < -age_cut
krause["age_older"] = krause["age_z"] > age_cut
krause["age_standout"] = krause["age_younger"] | krause["age_older"]

print()
print("Age-metallicity Relation (Krause21 ages)")
print(f"  trend: Age = {slope:.2f} * [Fe/H] + {intercept:.2f}  (Gyr)")
print(f"  typical scatter (sigma) = {age_sigma:.2f} Gyr")
print(f"  clusters that stand out: {krause['age_standout'].sum()} of {len(krause)}"
      f"   (about {len(krause) * math.erfc(age_cut / math.sqrt(2)):.1f} expected by chance)")
print(krause.loc[krause["age_standout"], ["ID", "FeH", "Age", "age_z"]]
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


# Assignment task 1: Identify potentially accreted clusters using the
# stellar population method with Krause21 ages
# Rule: candidates are clusters much younger than the age-metallicity trend.
# We count only younger because accreted clusters tend to be younger
# at a given [Fe/H]. Clusters that are older than the trend still stand
# out in the plot, but they are more likely just old Milky Way clusters,
# so they are not called candidates.

candidates = krause[krause["age_younger"]].sort_values("age_z")

print()
print("TASK 1: POTENTIALLY ACCRETED CLUSTERS (younger than the trend, Krause21 ages)")
print(f"  {len(candidates)} candidates out of {len(krause)} clusters with an age")
print(candidates[["ID", "FeH", "Age", "age_z"]].round(2).to_string(index=False))


# plot
out = krause[krause["age_standout"]].sort_values("FeH") # clusters that stand out

plt.figure(figsize=(10, 6))

# every cluster (no error bars, because Krause21 does not give any)
plt.scatter(krause["FeH"], krause["Age"], color="tab:blue", alpha=0.6,
            label="Clusters (no error bars)")

# the trend line and the band of +-age cut sigma around it
x_line = np.linspace(krause["FeH"].min(), krause["FeH"].max(), 100)
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
plt.title("Age-Metallicity Relation of Milky Way Globular Clusters (Krause21 Ages)")
plt.legend()
plt.tight_layout()
plt.show()

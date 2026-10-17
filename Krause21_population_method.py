import math 
from pathlib import Path
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 


# some flexible settings:

# A cluster stands out if it is more than this many sigma from the trend.
# For a normal (bell curve) distribution about 95% of ordinary clusters lie within 2 sigma, so only about 1 in 20 ordinary clusters is flagged by chance.
age_cut = 2.0

# When fitting the age trend, first throw away extreme clusters that are more than this many sigma away 
clip = 3.0


# loading data
folder = Path(__file__).resolve().parent # the folder this script is in, where the data files are
krause = pd.read_csv(folder / "Krause21.csv") # ages and [Fe/H], no error bars

# Krause21 writes "NGC104", so we add the space
krause["ID"] = krause["Object"].str.replace("NGC", "NGC ", regex=False)
# and three clusters are spelled differently from Harris
krause["ID"] = krause["ID"].replace(
    {"Ruprecht106": "Rup 106", "Terzan7": "Terzan 7", "Palomar12": "Pal 12"}) 


# This is the same method as stellar_population_method.py, applied to a second, independent set of ages. A cluster that stands out 
# with both sets of ages is a stronger candidate than one that stands out with only one.
# Krause21 gives no age error bars, so every cluster counts equally in the fit, and sigma mixes measurement error with the real spread of ages.

# first straight-line fit using every cluster.
slope, intercept = np.polyfit(krause["FeH"], krause["Age"], 1)
# residual of every cluster: its measured age minus the age the line predicts at its [Fe/H]. Negative = younger than the trend.
krause["age_resid"] = krause["Age"] - (slope * krause["FeH"] + intercept)

# remove extreme clusters and fit again
# Standard deviation of the residuals of all clusters
sigma_first = krause["age_resid"].std()
# True for clusters within clip (3) sigma of the line, False for the extreme ones that would pull the line towards themselves
kept = krause["age_resid"].abs() < clip * sigma_first
# Show which clusters are left out of the final fit
print("Removed before the final age fit:", krause.loc[~kept, "ID"].tolist())

# Fit the straight line again with the kept clusters only, so that the trend describes the normal clusters.
slope, intercept = np.polyfit(krause.loc[kept, "FeH"], krause.loc[kept, "Age"], 1)
# Residuals from the new line for every cluster
krause["age_resid"] = krause["Age"] - (slope * krause["FeH"] + intercept)

# sigma from the clusters that were kept, then a score for every cluster.
age_sigma = krause.loc[kept, "age_resid"].std()
# Score z: how many sigma each cluster is from the line (negative = younger)
krause["age_z"] = krause["age_resid"] / age_sigma

# some flags
# much younger than the trend: the sign we expect for an accreted cluster
krause["age_younger"] = krause["age_z"] < -age_cut
# much older than the trend
krause["age_older"] = krause["age_z"] > age_cut
# stands out in either direction
krause["age_standout"] = krause["age_younger"] | krause["age_older"]

print()
print("Age-metallicity Relation (Krause21 ages)")
# the fitted trend
print(f"  trend: Age = {slope:.2f} * [Fe/H] + {intercept:.2f}  (Gyr)")
# the scatter
print(f"  typical scatter (sigma) = {age_sigma:.2f} Gyr")
# How many clusters stand out, and how many would by pure chance.
print(f"  clusters that stand out: {krause['age_standout'].sum()} of {len(krause)}" f"   (about {len(krause) * math.erfc(age_cut / math.sqrt(2)):.1f} expected by chance)") 
print(krause.loc[krause["age_standout"], ["ID", "FeH", "Age", "age_z"]] .sort_values("age_z").round(2).to_string(index=False)) 


# Labels for the plot: making sure close by labels alternate between four positions so that they do not sit on top of each other(annoying).

def label_points(x_values, y_values, names): 
    # the four label positions, in points measured from the marker
    offsets = [(10, 10), (10, -16), (-10, 10), (-10, -16)]
    for i, (xv, yv, name) in enumerate(zip(x_values, y_values, names)):
        # i cycles through 0, 1, 2, 3, 0, ... so that neighbouring labels get different positions
        dx, dy = offsets[i % 4]
        plt.annotate(
            name, 
            (xv, yv),                        
            xytext=(dx, dy),                
            textcoords="offset points", 
            ha="left" if dx > 0 else "right",  
            fontsize=8,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.6), 
        ) 


# Assignment task 1: Identify potentially accreted clusters using the stellar population method with Krause21 ages
# candidates are clusters much younger than the age-metallicity trend. We count only younger because accreted clusters tend to be younger at a given [Fe/H]
# Clusters that are older than the trend still stand out in the plot, but they are more likely just old Milky Way clusters, so they are not called candidates.

candidates = krause[krause["age_younger"]].sort_values("age_z")

print()
print("TASK 1: POTENTIALLY ACCRETED CLUSTERS (younger than the trend, Krause21 ages)")
# how many candidates, out of how many clusters that could be tested
print(f"  {len(candidates)} candidates out of {len(krause)} clusters with an age")
# the candidate table
print(candidates[["ID", "FeH", "Age", "age_z"]].round(2).to_string(index=False))


# plot
out = krause[krause["age_standout"]].sort_values("FeH") # clusters that stand out

plt.figure(figsize=(10, 6)) 

# every cluster
plt.scatter(krause["FeH"], krause["Age"], color="tab:blue", alpha=0.6, 
            label="Clusters (no error bars)") 


x_line = np.linspace(krause["FeH"].min(), krause["FeH"].max(), 100)
# the ages the trend predicts at those [Fe/H] values
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, color="black", label="Trend (straight-line fit)")
plt.fill_between(x_line, y_line - age_cut * age_sigma, y_line + age_cut * age_sigma, color="gray", alpha=0.2, label=f"Within {age_cut:g} sigma of trend")

# clusters outside the band
plt.scatter(out["FeH"], out["Age"], s=90, color="red", zorder=3,
            label=f"Stand out (more than {age_cut:g} sigma)") 
label_points(out["FeH"], out["Age"], out["ID"]) 

plt.xlabel("[Fe/H]")
plt.ylabel("Age (Gyr)") 
plt.title("Age-Metallicity Relation of Milky Way Globular Clusters (Krause21 Ages)") 
plt.legend() 
plt.tight_layout()
plt.show()

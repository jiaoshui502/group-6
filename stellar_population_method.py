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
vdb = pd.read_csv(folder / "vandenBerg_table2.csv") # ages and [Fe/H]

# All tables need the Harris format names (NGC 104, Pal 12, ...) so that the results of the different tests can be joined cluster by cluster.
# van den Berg stores only the number (104...), so we put "NGC " in front
vdb["ID"] = "NGC " + vdb["#NGC"].astype(str).str.strip()
# for the three clusters without an NGC number (XXXX)
no_ngc = vdb["#NGC"].astype(str).str.strip() == "XXXX"
# give those three their Harris names
vdb.loc[no_ngc, "ID"] = vdb.loc[no_ngc, "Name"].map(
    {"Arp21": "Arp 2", "Pal12": "Pal 12", "Ter8": "Terzan 8"})  # Arp21 is likely to be a typo, since it has identical properties to Arp 2


# first straight-line fit using every cluster.
slope, intercept = np.polyfit(vdb["FeH"], vdb["Age"], 1)
# residual of every cluster: its measured age minus the age the line predicts at its [Fe/H]. Negative = younger than the trend.
vdb["age_resid"] = vdb["Age"] - (slope * vdb["FeH"] + intercept)

# remove extreme clusters and fit again
# Standard deviation of the residuals of all clusters
sigma_first = vdb["age_resid"].std()
# True for clusters within clip (3) sigma of the line, False for the extreme ones that would pull the line towards themselves
kept = vdb["age_resid"].abs() < clip * sigma_first
# Show which clusters are left out of the final fit
print("Removed before the final age fit:", vdb.loc[~kept, "ID"].tolist())

# Fit the straight line again with the kept clusters only, so that the trend describes the normal clusters.
slope, intercept = np.polyfit(vdb.loc[kept, "FeH"], vdb.loc[kept, "Age"], 1)
# Residuals from the new line for every cluster
vdb["age_resid"] = vdb["Age"] - (slope * vdb["FeH"] + intercept)

# sigma from the clusters that were kept, then a score for every cluster.
age_sigma = vdb.loc[kept, "age_resid"].std()
# Score z: how many sigma each cluster is from the line (negative = younger)
vdb["age_z"] = vdb["age_resid"] / age_sigma

# some flags
# much younger than the trend: the sign we expect for an accreted cluster
vdb["age_younger"] = vdb["age_z"] < -age_cut
# much older than the trend
vdb["age_older"] = vdb["age_z"] > age_cut
# stands out in either direction
vdb["age_standout"] = vdb["age_younger"] | vdb["age_older"]

print()
print("Age-metallicity Relation")
# the fitted trend
print(f"  trend: Age = {slope:.2f} * [Fe/H] + {intercept:.2f}  (Gyr)")
# the scatter
print(f"  typical scatter (sigma) = {age_sigma:.2f} Gyr")
# How many clusters stand out, and how many would by pure chance.
print(f"  clusters that stand out: {vdb['age_standout'].sum()} of {len(vdb)}" f" (about {len(vdb) * math.erfc(age_cut / math.sqrt(2)):.1f} expected by chance)") 
print(vdb.loc[vdb["age_standout"], ["ID", "FeH", "Age", "age_z"]] .sort_values("age_z").round(2).to_string(index=False)) 


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


# Assignment task 1: Identify potentially accreted clusters using the stellar population method
# candidates are clusters much younger than the age-metallicity trend. We count only younger because accreted clusters tend to be younger at a given [Fe/H]
# Clusters that are older than the trend still stand out in the plot, but they are more likely just old Milky Way clusters, so they are not called candidates.

# keep only the rows flagged as much younger, sorted from the most extreme
candidates = vdb[vdb["age_younger"]].sort_values("age_z")

print()
print("TASK 1: POTENTIALLY ACCRETED CLUSTERS (younger than the trend)")
# how many candidates, out of how many clusters that could be tested
print(f"  {len(candidates)} candidates out of {len(vdb)} clusters with an age")
# the candidate table
print(candidates[["ID", "FeH", "Age", "age_z"]].round(2).to_string(index=False))


# plot
out = vdb[vdb["age_standout"]].sort_values("FeH") # clusters that stand out, sorted by [Fe/H] for the labels

plt.figure(figsize=(10, 6)) 

# every cluster with its age error bar
plt.errorbar(vdb["FeH"], vdb["Age"], yerr=vdb["Age_err"], fmt="o", 
             color="tab:blue", alpha=0.6, capsize=3, label="Clusters")


x_line = np.linspace(vdb["FeH"].min(), vdb["FeH"].max(), 100)
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
plt.title("Age-Metallicity Relation of Milky Way Globular Clusters") 
plt.legend() 
plt.tight_layout()
plt.show()

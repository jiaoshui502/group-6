import math
from pathlib import Path
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D 


# some flexible settings:

# A cluster stands out if it is more than this many sigma from a trend
sigma_cut = 2.0

# When fitting the age trend, first throw away extreme clusters that are more than this many sigma away 
clip = 3.0

# motion of the sun
V0 = 229.0 # km/s, speed of the Sun's circular motion around the Galaxy
R0 = 8.178 # kpc, distance from the Sun to the galactic centre

# Test D: two clusters are "friends" if they are closer than link_r in space, and their velocity differ by less than link_v.
link_r = 5.0 # kpc, roughly the size of a dwarf galaxy
link_v = 40.0 # km/s, roughly twice the velocity spread inside a dwarf galaxy
outer_r = 10.0 # kpc, only clusters further out than this are consider grouped
min_members = 3 # smallest group that we count, because a pair can be by chance
n_shuffle = 1000 # number of shuffles for the chance test, to see how often a group as big as the real one appears


# loading data
folder = Path(__file__).resolve().parent # the folder this script is in, where the data files are
harris1 = pd.read_csv(folder / "HarrisPartI.csv") # positions
harris3 = pd.read_csv(folder / "HarrisPartIII.csv") # velocities
vdb = pd.read_csv(folder / "vandenBerg_table2.csv") # ages and [Fe/H]
krause = pd.read_csv(folder / "Krause21.csv") # second set of ages

# All tables need the Harris format names (NGC 104, Pal 12, ...) so that the results of the different tests can be joined cluster by cluster.
# van den Berg stores only the number (104...), so we put "NGC " in front
vdb["ID"] = "NGC " + vdb["#NGC"].astype(str).str.strip()
# for the three clusters without an NGC number (XXXX)
no_ngc = vdb["#NGC"].astype(str).str.strip() == "XXXX"
# give those three their Harris names
vdb.loc[no_ngc, "ID"] = vdb.loc[no_ngc, "Name"].map(
    {"Arp21": "Arp 2", "Pal12": "Pal 12", "Ter8": "Terzan 8"})  # Arp21 is likely to be a typo, since it has identical properties to Arp 2

# Krause21 writes "NGC104", so we add the space
krause["ID"] = krause["Object"].str.replace("NGC", "NGC ", regex=False)
# and three clusters are spelled differently from Harris
krause["ID"] = krause["ID"].replace(
    {"Ruprecht106": "Rup 106", "Terzan7": "Terzan 7", "Palomar12": "Pal 12"}) 


# tests A and B, age-metallicity relation
# Same method as stellar_population_method.py (A, van den Berg ages) and Krause21_population_method.py (B, Krause21 ages): 
# straight-line fit of Age against [Fe/H], remove clusters more than clip sigma from it, fit again, and give every cluster a score
# z = (distance from the line) / sigma. A cluster is a candidate if it is younger than the trend by more than sigma_cut (z < -sigma_cut)


def age_scores(table): # table: van den Berg or Krause21, with columns FeH and Age
    # first straight-line fit through every cluster
    slope, intercept = np.polyfit(table["FeH"], table["Age"], 1)
    # residual: measured age minus the age the line predicts (negative = younger)
    resid = table["Age"] - (slope * table["FeH"] + intercept)
    # True for clusters within clip sigma of the line; the others are too extreme and would pull the line towards themselves
    kept = resid.abs() < clip * resid.std()

    # fit again with the kept clusters only
    slope, intercept = np.polyfit(table.loc[kept, "FeH"], table.loc[kept, "Age"], 1)
    # residuals from the new line for every cluster
    resid = table["Age"] - (slope * table["FeH"] + intercept)
    # score = residual / scatter of the kept clusters
    return resid / resid[kept].std()


vdb["age_z"] = age_scores(vdb) # test A scores
krause["age_z"] = age_scores(krause) # test B scores


# test C, rotation
# Same method as dynamics_method.py: one rotation speed v_rot is fitted to all clusters (V_S = v_rot * cos(Psi)), and every cluster gets a score
# z = (V_S - v_rot * cos(Psi)) / sigma. A cluster is a candidate if |z| > sigma_cut (it is moving away from the rotation trend by more than sigma_cut).

# join positions and velocities by cluster name, keep clusters in both tables
dyn = pd.merge(harris1, harris3, on="ID", how="inner")
# drop clusters missing any number the method needs; copy() makes it an independent table so that new columns can be added safely
dyn = dyn.dropna(subset=["L", "B", "R_Sun", "v_LSR"]).copy()

l = np.radians(dyn["L"]) # galactic longitude in radians (numpy trigonometry need radians)
b = np.radians(dyn["B"]) # galactic latitude in radians
d = dyn["R_Sun"] # distance from the Sun, kpc

# how much of the Sun's circular motion (towards l = 90 deg) lies along our line of sight to the cluster
cos_A = np.sin(l) * np.cos(b)
# line-of-sight velocity seen by an observer at rest at the Sun's position
dyn["V_S"] = dyn["v_LSR"] + V0 * cos_A

x = d * np.cos(b) * np.cos(l) # position in the disc plane, towards the galactic centre
y = d * np.cos(b) * np.sin(l) # position in the disc plane, along the Sun's motion
# part of a circular orbit at the cluster's position that points along our line of sight; the square root is the cluster's distance from the
# galactic centre measured in the disc plane
dyn["cos_Psi"] = R0 * cos_A / np.sqrt((x - R0) ** 2 + y ** 2)

# best single rotation speed (least-squares line through zero)
v_rot = (dyn["cos_Psi"] * dyn["V_S"]).sum() / (dyn["cos_Psi"] ** 2).sum()
# peculiar velocity: what is left after the shared rotation is taken away
dyn["v_pec"] = dyn["V_S"] - v_rot * dyn["cos_Psi"]
# score: peculiar velocity in unit of the typical random motion, to see how much the cluster is moving away from the rotation trend
dyn["dyn_z"] = dyn["v_pec"] / dyn["v_pec"].std()


# test D: groups in position and velocity
# A dwarf galaxy that falls into the Milky Way brings its own clusters, and they may stay close together and keep the velocity of the dwarf galaxy
# they will form a tight group in position and velocity. Clusters that formed in the Milky Way have much less chance to form such groups, because
# the inner Galaxy is so crowded. So we only search clusters further than outer_r from the Galactic centre.
#
# Method : we define two clusters are 'friends' if they are closer than link_r and their V_S differ by less than link_v. 
# A friend of a friend is also in the same group.

# Chance test: shuffle the velocities between the clusters, so that positions and velocities are no longer together, and find the groups again.
# Repeat n_shuffle times and count how often a group as big as the real biggest one appears.

# the outer clusters only; reset_index numbers them 0, 1, 2, ... so that the row number matches the position in the numpy arrays below
outer = dyn[dyn["R_gc"] > outer_r].reset_index(drop=True)

# distance between every pair of clusters, using the Harris X, Y, Z (kpc)
pos = outer[["X", "Y", "Z"]].to_numpy()
# pos[:, None, :] - pos[None, :, :] subtracts every cluster's position from every other's at once (an n x n x 3 array). Square, add X, Y and Z,
# and taking the square root gives an n x n table where separation[i, k] is the distance between cluster i and cluster k.
separation = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))


def find_groups(velocity): # velocity: V_S of each outer cluster, in the same order as pos
    # n x n table of True/False: True where two clusters are close in space and have similar velocities(friends!)
    friends = (separation < link_r) & (np.abs(velocity[:, None] - velocity[None, :]) < link_v)
    group = np.arange(len(velocity)) # every cluster starts in its own group
    for i in range(len(velocity)):
        for k in range(i + 1, len(velocity)): 
            if friends[i, k]: 
                group[group == group[k]] = group[i] #if the two are friends, join the two groups
    return group


# group number of every outer cluster, using the real velocities
outer["group"] = find_groups(outer["V_S"].to_numpy())
# size of the group each cluster belongs to: value_counts() counts how many clusters share each group number, and map() looks up each cluster's group 
# number in that count
outer["group_size"] = outer["group"].map(outer["group"].value_counts())
biggest = outer["group_size"].max() # size of the biggest real group

rng = np.random.default_rng(1) # fixed seed, so the result can be repeated
# For each of the n_shuffle trials: shuffle the velocities between the clusters (rng.permutation), find the groups again, and record the size of the biggest group
shuffled_biggest = np.array([
    np.bincount(find_groups(rng.permutation(outer["V_S"].to_numpy()))).max() # shuffle, regroup, size of the biggest group
    for _ in range(n_shuffle) 
])
# fraction of shuffled trials with a group at least as big as the real one: the chance of finding such a group if position and velocity were unrelated
p_chance = np.mean(shuffled_biggest >= biggest)

print("Test D: groups in position and velocity") 
print(f"  clusters searched (R_gc > {outer_r:g} kpc): {len(outer)}") # how many clusters were searched
# print every group with at least min_members clusters
for _, members in outer[outer["group_size"] >= min_members].groupby("group"):
    # age_z is the score of test A (NaN if there is no van den Berg age)
    members = members.merge(vdb[["ID", "age_z"]], on="ID", how="left")
    # name, sky position, distances, velocities and age score of each member
    print(members[["ID", "L", "B", "R_Sun", "R_gc", "v_r", "V_S", "age_z"]]
          .round(2).to_string(index=False)) # 2 decimals
print(f"  A group of {biggest} or more clusters appeared in " f"{100 * p_chance:.1f}% of {n_shuffle} shuffles") # the chance probability, as a percentage 

# one table with every test for every cluster in the Harris catalogue:
# True = flagged by the test
# False = tested, not flagged
# NaN = the test was not possible (no age, no velocity)


tests = pd.DataFrame({
    "A: age, vdB": vdb.set_index("ID")["age_z"] < -sigma_cut, # much younger, van den Berg ages
    "B: age, Krause21": krause.set_index("ID")["age_z"] < -sigma_cut, # much younger, Krause21 ages
    "C: rotation": dyn.set_index("ID")["dyn_z"].abs() > sigma_cut, # does not follow the rotation
    "D: position+velocity group": outer.set_index("ID")["group_size"] >= min_members, # in a group(friends)
}).reindex(harris1["ID"]) 

n_tests = tests.notna().sum(axis=1) # for each cluster: how many tests were possible
n_flags = (tests == True).sum(axis=1) # for each cluster: how many tests flag it

# How many flags we would expect from scatter alone (normal distribution).
# A and B only count younger clusters (one side), C counts both sides.
one_sided = 0.5 * math.erfc(sigma_cut / math.sqrt(2)) # chance of being more than 2 sigma below
two_sided = math.erfc(sigma_cut / math.sqrt(2)) # chance of being more than 2 sigma either side

print() 
print("Clusters flagged by each test") 
# for tests A, B and C: number flagged, number tested, and the chance expectation
for name, chance in [("A: age, vdB", one_sided), # test A only counts younger clusters: one side
                     ("B: age, Krause21", one_sided), # test B too
                     ("C: rotation", two_sided)]: # test C counts both directions: two sides
    n_tested = tests[name].notna().sum() # clusters this test could be applied to
    print(f"  {name:28s} {int((tests[name] == True).sum()):2d} of {n_tested:3d}" # flagged out of tested
          f"   (about {n_tested * chance:.1f} expected by chance)") # expected number = tested x chance
name = "D: position+velocity group" # test D has the shuffle probability instead
print(f"  {name:28s} {int((tests[name] == True).sum()):2d} of {tests[name].notna().sum():3d}" # flagged out of tested
      f"   (group this big in {100 * p_chance:.1f}% of shuffles)") # chance from the shuffle test

# Finally to direclty answer assignment task 1: the combined list of potentially accreted clusters
candidates = tests[n_flags >= 1].copy() # clusters flagged by at least one test
candidates["flags"] = n_flags[n_flags >= 1] # how many tests flag each candidate
candidates["tests possible"] = n_tests[n_flags >= 1] # how many tests could be applied to it
# most flags first; clusters with the same number of flags in alphabetical order
candidates = candidates.sort_values(["flags", "ID"], ascending=[False, True])

print() 
print("Potentially Accreted Clusters (flagged by at least one test)") 
print(candidates.replace({True: "yes", False: "no"}).fillna("-").to_string())

print() 
print(f"Clusters in the Harris catalogue: {len(tests)}") # all clusters
print(f"  At least one test possible: {(n_tests >= 1).sum()}") # clusters we could test at all
print(f"  Both an age (A or B) and a velocity (C): " # clusters with both kinds of information
      f"{(tests[['A: age, vdB', 'B: age, Krause21']].notna().any(axis=1) & tests['C: rotation'].notna()).sum()}") # (A or B possible) and C possible
print(f"  Flagged by at least one test: {len(candidates)}" # number of candidates,
      f" ({100 * len(candidates) / len(tests):.0f}% of the catalogue)") # as a percentage of all clusters
# This is a lower limit on the number of accreted clusters: old accreted clusters can sit on the age trend, and most clusters have no age at all.


# Figure 1: which test flags which candidate
fig, ax = plt.subplots(figsize=(8, 0.42 * len(candidates) + 2))

for col, name in enumerate(tests.columns): 
    for row, value in enumerate(candidates[name]):
        if value == True: # flagged
            ax.scatter(col, row, s=170, color="red", zorder=3)
        elif value == False: # tested but not flagged
            ax.scatter(col, row, s=170, facecolors="none", edgecolors="gray")
        else: # test not possible
            ax.scatter(col, row, s=60, marker="x", color="lightgray") 

ax.set_xticks(range(len(tests.columns))) 
ax.set_xticklabels([n.replace(": ", ":\n") for n in tests.columns]) 
ax.set_yticks(range(len(candidates)))
ax.set_yticklabels([f"{c}  ({f})" for c, f in zip(candidates.index, candidates["flags"])]) 
ax.set_xlim(-0.5, len(tests.columns) - 0.5) 
ax.set_ylim(len(candidates) - 0.5, -0.5) 
ax.set_ylabel("Cluster  (number of tests that flag it)")
ax.set_title("Potentially Accreted Globular Clusters: Which Test Flags Them") 
ax.legend(handles=[
    Line2D([], [], marker="o", color="red", linestyle="", markersize=10, label="flagged"),
    Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="gray", 
           linestyle="", markersize=10, label="tested, not flagged"), 
    Line2D([], [], marker="x", color="lightgray", linestyle="", markersize=8, 
           label="test not possible"), 
], loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False) 
plt.tight_layout() 
plt.show() 


# Figure 2: sky positions of the outer clusters, coloured by velocity
# Galactic longitude is shifted to -180 to 180 so that the Galactic centre direction l = 0 is in the middle.
outer["l_plot"] = np.where(outer["L"] > 180, outer["L"] - 360, outer["L"]) 
in_group = outer["group_size"] >= min_members # True for the clusters in the group found by test D
v_max = outer["V_S"].abs().max() # largest speed, to make the colour scale symmetric around 0

plt.figure(figsize=(10, 6)) 
# every outer cluster at its sky position, coloured by V_S: blue = coming towards us, red = moving away, white = about 0. same colour = same velocity
points = plt.scatter(outer["l_plot"], outer["B"], c=outer["V_S"], cmap="coolwarm",
                     vmin=-v_max, vmax=v_max, s=60, edgecolors="black", linewidths=0.4) # symmetric colour scale
# a big black open circle around each member of the group
plt.scatter(outer.loc[in_group, "l_plot"], outer.loc[in_group, "B"], s=260,
            facecolors="none", edgecolors="black", linewidths=1.5, 
            label=f"Group found by test D ({in_group.sum()} clusters)") 
# The group members are close together on the sky, so their labels are stacked to the right of the group
group_rows = outer[in_group].sort_values("B", ascending=False) # members from highest to lowest latitude
for i, (_, row) in enumerate(group_rows.iterrows()): # i counts down the stack
    plt.annotate(row["ID"], (row["l_plot"], row["B"]), xytext=(45, 28 - 19 * i), # each label 19 points below the last
                 textcoords="offset points", fontsize=8, va="center", # offsets in points
                 arrowprops=dict(arrowstyle="-", color="gray", lw=0.6)) # thin grey line to the cluster

plt.colorbar(points, label="V_S   (km/s)") # colour bar
plt.xlim(-180, 180) # the whole sky in longitude
plt.ylim(-90, 90) # the whole sky in latitude
plt.axhline(0, color="black", linewidth=0.6, alpha=0.4) 
plt.axvline(0, color="black", linewidth=0.6, alpha=0.4) 
plt.ylabel("Galactic latitude b (deg)") 
plt.title(f"Clusters More Than {outer_r:g} kpc from the Galactic Centre") 
plt.legend(loc="lower left")
plt.tight_layout() 
plt.show() 

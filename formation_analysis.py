from pathlib import Path 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.stats import spearmanr, mannwhitneyu # two statistical tests


# some flexible settings:

# boundary between metal-rich and metal-poor clusters
feh_split = -1.0

# Below this [Fe/H] the two age branches meet (Leaman et al. 2013), so the
# branch of a cluster cannot be decided from its age (parts 4 and 5).
feh_branch_min = -1.8

# distance ranges from the Galactic centre in kpc (parts 3 and 5):
# inner Galaxy (0 to 5), middle (5 to 10) and outer halo (10 and beyond)
r_edges = [0, 5, 10, np.inf]

# motion of the sun, same values as dynamics_method.py
V0 = 229.0 # km/s, speed of the Sun's circular motion around the Galaxy
R0 = 8.178 # kpc, distance from the Sun to the galactic centre


#loading data
folder = Path(__file__).resolve().parent 
harris1 = pd.read_csv(folder / "HarrisPartI.csv") # positions
harris3 = pd.read_csv(folder / "HarrisPartIII.csv") # velocities
vdb = pd.read_csv(folder / "vandenBerg_table2.csv") # ages and [Fe/H]
krause = pd.read_csv(folder / "Krause21.csv") # second set of ages and [Fe/H]

# All tables need the Harris format names (NGC 104, Pal 12, ...) so that the results of the different tests can be joined cluster by cluster.
# van den Berg stores only the number (104...), so we put "NGC " in front
vdb["ID"] = "NGC " + vdb["#NGC"].astype(str).str.strip()
# for the three clusters without an NGC number (XXXX)
no_ngc = vdb["#NGC"].astype(str).str.strip() == "XXXX"
# give those three their Harris names
vdb.loc[no_ngc, "ID"] = vdb.loc[no_ngc, "Name"].map(
    {"Arp21": "Arp 2", "Pal12": "Pal 12", "Ter8": "Terzan 8"}) # table spelling -> Harris name

# Krause21 writes "NGC104", so we add the space
krause["ID"] = krause["Object"].str.replace("NGC", "NGC ", regex=False)
# and three clusters are spelled differently from Harris
krause["ID"] = krause["ID"].replace(
    {"Ruprecht106": "Rup 106", "Terzan7": "Terzan 7", "Palomar12": "Pal 12"}) 

# distance from the Galactic centre for every cluster with an age. Adds the Harris column R_gc to the van den Berg table, matched by name
vdb = vdb.merge(harris1[["ID", "R_gc"]], on="ID")

# [Fe/H] for as many clusters as possible: van den Berg first, then the Krause21 clusters that van den Berg does not have.
feh = pd.concat([
    vdb[["ID", "FeH"]], 
    krause.loc[~krause["ID"].isin(vdb["ID"]), ["ID", "FeH"]], # plus the 9 that only Krause21 has
]) 

# join positions and velocities by cluster name, keep clusters in both tables
dyn = pd.merge(harris1, harris3, on="ID", how="inner")
# drop clusters missing any number the method needs; copy() makes it an independent table so that new columns can be added safely
dyn = dyn.dropna(subset=["L", "B", "R_Sun", "v_LSR"]).copy()

l = np.radians(dyn["L"]) # galactic longitude in radians (numpy trigonometry needs radians)
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

def rotation_fit(sample): # sample can be any group of clusters with V_S and cos_Psi
    # Frenk & White (1980) estimator, same as in dynamics_method.py
    cos_psi = sample["cos_Psi"] # the rotation geometry of each cluster in the group
    # best single rotation speed: least-squares line V_S = v_rot * cos(Psi) through zero
    v_rot = (cos_psi * sample["V_S"]).sum() / (cos_psi ** 2).sum()
    # random motion: scatter of the velocities around the rotation line
    sigma = (sample["V_S"] - v_rot * cos_psi).std()
    # error of v_rot: the error of the slope of a line through zero
    v_rot_err = sigma / np.sqrt((cos_psi ** 2).sum())
    return v_rot, v_rot_err, sigma 

# results of every rotation fit
rotation_results = {}


def report_rotation(name, sample): # fits one group, prints the result and saves it for figure 3
    v_rot, v_rot_err, sigma = rotation_fit(sample) # rotation fit of this group
    rotation_results[name] = (len(sample), v_rot, v_rot_err, sigma) # number of clusters and the fit results
    print(f"  {name:26s} n = {len(sample):3d}   v_rot = {v_rot:5.0f} +/- {v_rot_err:3.0f} km/s" # name, size, rotation
          f"   sigma = {sigma:4.0f} km/s") # and random motion


# Rotation of metal-rich and metal-poor clusters
# Clusters that formed in the Milky Way's own disc and bulge should share their fast rotation and small random motions.
# Clusters brought in by dwarf galaxies arrive on orbits in all directions, so a group that contains many of them rotates slowly and has large random motions.
# Only the clusters with an [Fe/H] in our tables can be used here.

with_feh = dyn.merge(feh, on="ID") # velocity clusters that also have an [Fe/H]

print("Rotation of different groups of clusters")
report_rotation("all clusters", dyn) # the whole system, for comparison (same as dynamics_method.py)
report_rotation(f"metal-rich, [Fe/H] > {feh_split:g}", with_feh[with_feh["FeH"] > feh_split]) # metal-rich only
report_rotation(f"metal-poor, [Fe/H] <= {feh_split:g}", with_feh[with_feh["FeH"] <= feh_split]) # metal-poor only


# Age offset from the trend against distance from the Galactic centre. The offset (residual) is measured from the straight-line trend of 
# stellar_population_method.py. A negative residual means younger than other clusters of the same [Fe/H]. If the accreted clusters live mostly in the 
# outer halo, the residual should get more negative further out. 

# the same trend as stellar_population_method.py:
slope, intercept = np.polyfit(vdb["FeH"], vdb["Age"], 1) # first straight-line fit through all clusters
resid = vdb["Age"] - (slope * vdb["FeH"] + intercept) # residual of every cluster
kept = resid.abs() < 3 * resid.std() # leave out clusters more than 3 sigma away
slope, intercept = np.polyfit(vdb.loc[kept, "FeH"], vdb.loc[kept, "Age"], 1) # fit again without them
vdb["age_resid"] = vdb["Age"] - (slope * vdb["FeH"] + intercept) # final residual: age minus trend

# Spearman's rank correlation(test for whether two quantities rise or fall together) between distance and age offset. It ranks
# rho runs from -1 to +1 (negative = further out = younger), and p is the chance of a correlation this strong if there were no real connection.
rho, p_rho = spearmanr(vdb["R_gc"], vdb["age_resid"])
# put every cluster into its distance range
vdb["r_range"] = pd.cut(vdb["R_gc"], r_edges, right=False)
# number of clusters and mean age offset in each distance range
resid_by_range = vdb.groupby("r_range", observed=False)["age_resid"].agg(["count", "mean"])

print()
print("Age offset from the trend against distance from the Galactic centre") 
print(f"  Spearman rho = {rho:.2f}, p = {p_rho:.1e}  ({len(vdb)} clusters)") # the correlation result
print(resid_by_range.round(2).to_string()) # the table of mean offsets per distance range


# Two age-metallicity branches: Earlier studies (Marin-Franch et al. 2009; Leaman et al. 2013) found that above [Fe/H] = -1.8 the clusters follow 
# two parallel sequences: an old one, and one about 1 Gyr younger.
#
# Method we use here(uses only Age and [Fe/H]):
# a. start from one straight line: we say clusters above it are "old", clusters below it are "young"
# b. fit two parallel lines (same slope, different intercepts) through the two groups, weighting every cluster by its age error
# c. move every cluster to the line it is closer to
# d. repeat b and c until no cluster changes branch
# Check:
# distance from the Galactic centre and rotation are not used to make the split. If the two branches also differ in those, the split separates two 
# real populations.


def two_branches(table): # table of clusters with FeH, Age and Age_err
    x = table["FeH"].to_numpy() # [Fe/H] of each cluster
    y = table["Age"].to_numpy() # age of each cluster
    err = table["Age_err"].to_numpy() # error bar of each age

    # a.
    m, c = np.polyfit(x, y, 1) # one straight line through all of them
    old = y > m * x + c # True above the line (old), false below it (young)

    for _ in range(100): # repeat at most 100 times 
        # Dividing every row by the age error weights the fit.
        # The model is Age = m * [Fe/H] + c_old (for old clusters) or m * [Fe/H] + c_young (for young ones), with one shared slope m and two intercepts. 
        design = np.column_stack([x, old, ~old]).astype(float)
        # solve for m, c_old and c_young; dividing by err gives clusters with small error bars more weight
        m, c_old, c_young = np.linalg.lstsq(design / err[:, None], y / err, rcond=None)[0]

        # c.
        # True where a cluster is closer to the old line than to the young line
        new_old = np.abs(y - (m * x + c_old)) < np.abs(y - (m * x + c_young))

        # d.
        if (new_old == old).all(): # no cluster changed branch = the split is settled
            break 
        old = new_old

    return old, m, c_old, c_young # branch of every cluster, and the two fitted lines


branches = vdb[vdb["FeH"] > feh_branch_min].copy() # only clusters where the branches are separate
old, m, c_old, c_young = two_branches(branches) # find the two branches
branches["branch"] = np.where(old, "old", "young") # write old or young for each cluster
not_classified = vdb[vdb["FeH"] <= feh_branch_min] # too metal-poor

print()
print(f"Two branches (clusters with [Fe/H] > {feh_branch_min:g})")
print(f"  old branch:   Age = {m:.2f} [Fe/H] + {c_old:.2f} Gyr   ({old.sum()} clusters)") # old line and its size
print(f"  young branch: Age = {m:.2f} [Fe/H] + {c_young:.2f} Gyr   ({(~old).sum()} clusters)") # young line and its size
print(f"  the young branch is {c_old - c_young:.2f} Gyr younger at the same [Fe/H]") # gap between the lines
print(f"  not classified ([Fe/H] <= {feh_branch_min:g}, where the branches meet): " f"{len(not_classified)} clusters")
for name in ["old", "young"]:
    members = branches[branches["branch"] == name].sort_values("FeH") 
    print(f"  {name}: {', '.join(members['ID'])}") 

# Check 1: distance from the Galactic centre. The Mann-Whitney test asks whether the two sets of distances could come from the same distribution.
r_old = branches.loc[old, "R_gc"] # distances of the old-branch clusters
r_young = branches.loc[~old, "R_gc"] # distances of the young-branch clusters
# small p = very unlikely that both branches have the same spread of distances
p_r = mannwhitneyu(r_old, r_young).pvalue

print()
print("Check 1: distance from the Galactic centre")
print(f"  median R_gc: old {r_old.median():.1f} kpc, young {r_young.median():.1f} kpc" f"   (Mann-Whitney p = {p_r:.1e})")
# Check 2: rotation of each branch
with_branch = dyn.merge(branches[["ID", "branch"]], on="ID") # velocity clusters that have a branch

print("Check 2: rotation") 
report_rotation("old age branch", with_branch[with_branch["branch"] == "old"]) # rotation of the old branch
report_rotation("young age branch", with_branch[with_branch["branch"] == "young"]) # rotation of the young branch


# How many clusters could be accreted?
# The young branch is taken as accreted and the old branch as formed in the Milky Way. The vandenBerg age sample has fewer inner clusters than the Harris
# catalogue, and the inner clusters are mostly on the old branch. So the accreted fraction is worked out for each distance range separately and then 
# applied to the number of Harris clusters in that range. This assumes that at a given distance clusters without ages are like the clusters with ages.

# number of Harris clusters in each distance range
n_harris = pd.cut(harris1["R_gc"], r_edges, right=False).value_counts(sort=False)


def accreted_by_range(table): # table are clusters with a "branch" and an "R_gc" column
    young = table["branch"] == "young" # True for young-branch clusters
    r_range = pd.cut(table["R_gc"], r_edges, right=False) # the distance range of each cluster
    # per distance range: sum counts the True values (young clusters) and "count" counts all classified clusters
    counts = young.groupby(r_range, observed=False).agg(["sum", "count"])
    counts.columns = ["young", "classified"]
    counts["fraction young"] = counts["young"] / counts["classified"] # accreted fraction in each range
    counts["Harris"] = n_harris # number of Harris clusters in each range (matched by range)
    counts["accreted"] = counts["fraction young"] * counts["Harris"] # estimated accreted clusters in each range
    return counts 


counts = accreted_by_range(branches) # the estimate with the chosen [Fe/H] limit
n_accreted = counts["accreted"].sum() # total over all distance ranges

print()
print("How many clusters could be accreted (young branch = accreted)") 
print(counts.round(2).to_string()) 
print(f"  estimate: {n_accreted:.0f} accreted and {len(harris1) - n_accreted:.0f} formed in the" 
f" Milky Way, out of {len(harris1)} ({100 * n_accreted / len(harris1):.0f}% accreted)")

# Sensitivity: the same estimate with other [Fe/H] limits for the split.
# If the answer barely changes, it does not depend on our choice of -1.8.
print("  the same estimate with other [Fe/H] limits for the branch split:") 
for feh_min in [-1.7, -1.8, -1.9, -2.0]: # four nearby choices of the limit
    test = vdb[vdb["FeH"] > feh_min].copy() # clusters above this limit
    test["branch"] = np.where(two_branches(test)[0], "old", "young") # split them again
    n = accreted_by_range(test)["accreted"].sum() # and redo the estimate
    print(f"    [Fe/H] > {feh_min:g}: {n:.0f} of {len(harris1)} ({100 * n / len(harris1):.0f}%)") # result for this limit


# Figure 1: the two age-metallicity branches
plt.figure(figsize=(10, 6))

for name, color in [("old", "tab:red"), ("young", "tab:blue")]: # old branch in red, young in blue
    members = branches[branches["branch"] == name] # clusters on this branch
    plt.errorbar(members["FeH"], members["Age"], yerr=members["Age_err"], fmt="o", # points with age error bars
                 color=color, capsize=3, label=f"{name} branch ({len(members)} clusters)") # colour and legend text

# the clusters too metal-poor to classify, in faint grey
plt.errorbar(not_classified["FeH"], not_classified["Age"], yerr=not_classified["Age_err"],
             fmt="o", color="gray", alpha=0.5, capsize=3, # faint grey points with error bars
             label=f"not classified, [Fe/H] <= {feh_branch_min:g} ({len(not_classified)} clusters)") 

x_line = np.linspace(feh_branch_min, vdb["FeH"].max(), 100) # [Fe/H] values where the branches are fitted
plt.plot(x_line, m * x_line + c_old, color="tab:red") # the old-branch line
plt.plot(x_line, m * x_line + c_young, color="tab:blue") # the young-branch line
plt.axvline(feh_branch_min, color="gray", linestyle=":", linewidth=1) # dotted line at the [Fe/H] limit

# names of the clusters with offsets so that neighbouring labels do not overlap
label_offsets = {"NGC 362": (-8, -12), "NGC 1261": (8, -12), "NGC 6218": (8, 4),
                 "NGC 6723": (-4, 16), "NGC 6362": (8, 8), "Pal 12": (8, -2)} 
for _, row in vdb[vdb["ID"].isin(label_offsets)].iterrows():
    dx, dy = label_offsets[row["ID"]] 
    plt.annotate(row["ID"], (row["FeH"], row["Age"]), xytext=(dx, dy), 
                 textcoords="offset points", fontsize=8, ha="left" if dx > 0 else "right", 
                 arrowprops=dict(arrowstyle="-", color="gray", lw=0.6)) 

plt.xlabel("[Fe/H]") 
plt.ylabel("Age (Gyr)") 
plt.title("Two Age-Metallicity Branches of Milky Way Globular Clusters") 
plt.legend(loc="lower left")
plt.tight_layout() 
plt.show()


# Figure 2: age offset against distance from the Galactic centre
range_centres = vdb.groupby("r_range", observed=False)["R_gc"].median() # where to plot each range's mean: its median distance

plt.figure(figsize=(9, 6))
plt.errorbar(vdb["R_gc"], vdb["age_resid"], yerr=vdb["Age_err"], fmt="o", alpha=0.6, # every cluster with its age error
             capsize=3, label="Clusters (van den Berg ages)")
plt.plot(range_centres, resid_by_range["mean"], "s-", color="black", markersize=8, # mean offset per range, 
         label="Mean offset in each distance range") 
for edge in r_edges[1:-1]: # the inner edges only (5 and 10 kpc), not 0 and infinity
    plt.axvline(edge, color="gray", linestyle=":", linewidth=1) 
plt.axhline(0, color="black", linewidth=0.8) 

plt.xscale("log")
plt.xticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"]) 
plt.xlabel("Distance from the Galactic centre R_gc (kpc)") 
plt.ylabel("Age - trend (Gyr)   (negative = younger for its [Fe/H])")
plt.title("Age Offset From the Trend Against Distance From the Galactic Centre") 
plt.text(0.03, 0.04, f"Spearman rho = {rho:.2f}, p = {p_rho:.1e}", 
         transform=plt.gca().transAxes) 
plt.legend(loc="upper right") 
plt.tight_layout()
plt.show()


# Figure 3: rotation of the different groups
names = list(rotation_results) 
results = np.array([rotation_results[name] for name in names]) 

# same colours as figure 1, red = groups expected to have formed in the milky way, blue = groupsexpected to contain many accreted clusters
colors = ["black", "tab:red", "tab:blue", "tab:red", "tab:blue"]

plt.figure(figsize=(10, 6))
for i, (n, v_rot, v_rot_err, sigma) in enumerate(results): 
    plt.errorbar(i, v_rot, yerr=v_rot_err, fmt="o", color=colors[i], markersize=9, capsize=5) # rotation speed with its error
    plt.annotate(f"n = {n:.0f}\nsigma = {sigma:.0f} km/s", (i, v_rot + v_rot_err), # group size and random motion
                 xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8) 
plt.axhline(0, color="gray", linestyle="--", linewidth=1) # zero rotation

plt.xticks(range(len(names)), [name.replace(", ", "\n") for name in names]) 
plt.xlim(-0.5, len(names) - 0.5) 
plt.ylim((results[:, 1] - results[:, 2]).min() - 30, (results[:, 1] + results[:, 2]).max() + 50)
plt.ylabel("Rotation speed v_rot (km/s)")
plt.title("Rotation of Different Groups of Milky Way Globular Clusters") 
plt.tight_layout() 
plt.show() 

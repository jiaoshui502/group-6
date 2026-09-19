from pathlib import Path
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
vandenberg["difference"] = (
    vandenberg["Age"] - vandenberg["Age_fit"]
)

# Select clusters whose error bars doesn't cross the fitted line
outside_fit = vandenberg[
    np.abs(vandenberg["difference"]) > vandenberg["Age_err"]
].copy()

print("Clusters whose age error bars do not include the fitted line:")
print(
    outside_fit[
        ["#NGC", "Name", "FeH", "Age",
         "Age_err", "Age_fit", "difference"]
    ].sort_values("difference")
)
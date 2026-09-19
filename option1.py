from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data input

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
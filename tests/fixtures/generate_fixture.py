import os
import pickle
import numpy as np
from optiwindnet.api import WindFarmNetwork

# Define input data
cables =[(2, 10), (5, 20)]
turbinesC = np.array([[0, 0], [-1, 2]])
substationsC = np.array([[1, 1]])

# Provide a valid border to avoid empty geometry issues
borderC = np.array([
    [0, 0],
    [0, 10],
    [10, 10],
    [10, 0]
])

# Generate instance
try:
    wf_network = WindFarmNetwork(
        cables=cables,
        turbinesC=turbinesC,
        substationsC=substationsC,
        borderC=borderC
    )
    print("WindFarmNetwork instance created successfully.")

except Exception as e:
    print(f"Error during instance creation: {e}")
    exit(1)

# Ensure fixture directory exists
fixture_dir = os.path.join("tests", "fixtures")
os.makedirs(fixture_dir, exist_ok=True)

# Save to pickle
fixture_path = os.path.join(fixture_dir, "wf_network.pkl")
with open(fixture_path, "wb") as f:
    pickle.dump(wf_network, f)
    print(f"Fixture saved to {fixture_path}")

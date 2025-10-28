# test_single_pkl.py
import pandas as pd
import sys

# --- Configuration ---
# This path should point to the very first LiDAR file in your sequence.
LIDAR_FILE_PATH = "/mnt/d/saksham/college stuff/Munjal/Fifth Semester/classroom/DIP/DIP_DVMS/Pandaset 3 scenes/018/lidar/00.pkl"

print(f"--- Testing direct load of: {LIDAR_FILE_PATH} ---")
print(f"Using pandas version: {pd.__version__}")
print(f"Using Python version: {sys.version}")

try:
    # This is the critical line. We are calling pandas directly.
    lidar_df = pd.read_pickle(LIDAR_FILE_PATH)
    
    print("\n[SUCCESS] Successfully loaded the .pkl file!")
    print("This is unexpected, but good news. The problem might be how the SDK lists files.")
    print("\nDataframe info:")
    print(lidar_df.info())
    print("\nFirst 5 rows:")
    print(lidar_df.head())

except Exception as e:
    print(f"\n[FAILURE] Failed to load pickle file.")
    print("=============================================================")
    print("THIS IS THE REAL ERROR MESSAGE:")
    print("=============================================================")
    # The following line will print the full, detailed traceback.
    raise e
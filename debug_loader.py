# debug_loader.py

import pandaset

# --- Configuration ---
PANDASET_ROOT_PATH = "/mnt/d/saksham/college stuff/Munjal/Fifth Semester/classroom/DIP/DIP_DVMS/Pandaset 3 scenes"
SEQUENCE_ID = "018"

# --- Main Debug Logic ---
print("--- Starting PandaSet Loader Debug ---")
print(f"Attempting to load sequence '{SEQUENCE_ID}' from '{PANDASET_ROOT_PATH}'")

try:
    # 1. Initialize the dataset and select the sequence
    dataset = pandaset.DataSet(PANDASET_ROOT_PATH)
    sequence = dataset[SEQUENCE_ID]
    print("[SUCCESS] Sequence object created.")

    # 2. Explicitly load camera and lidar data
    print("Attempting to load camera data...")
    sequence.load_camera()
    print("[SUCCESS] sequence.load_camera() executed.")

    print("Attempting to load lidar data...")
    sequence.load_lidar()
    print("[SUCCESS] sequence.load_lidar() executed.")

    # 3. Select the primary LiDAR sensor
    print("Attempting to set LiDAR sensor to 0...")
    sequence.lidar.set_sensor(0)
    print("[SUCCESS] sequence.lidar.set_sensor(0) executed.")

    # 4. --- THE CRITICAL CHECK ---
    # Check the number of items the SDK has actually loaded
    print("\n--- Verification Step ---")
    num_camera_poses = len(sequence.camera['front_camera'].poses)
    print(f"Number of camera poses found: {num_camera_poses}")

    # The .data attribute holds the list of loaded frames
    num_lidar_frames = len(sequence.lidar.data)
    print(f"Number of LiDAR frames found: {num_lidar_frames}")

    if num_camera_poses > 0 and num_lidar_frames == 0:
        print("\n[DIAGNOSIS] The SDK loaded camera data but failed to load any LiDAR data.")
        print("This is the cause of the IndexError. The likely reason is a silent failure during the reading of .pkl files (e.g., pickle version mismatch or file corruption).")
    elif num_camera_poses != num_lidar_frames:
        print(f"\n[DIAGNOSIS] Mismatch between camera frames ({num_camera_poses}) and LiDAR frames ({num_lidar_frames}).")
    else:
        print("\n[DIAGNOSIS] Data appears to be loaded correctly. The problem may lie elsewhere.")

except Exception as e:
    print(f"\n[ERROR] An exception occurred during the loading process: {e}")

print("\n--- Debug Script Finished ---")
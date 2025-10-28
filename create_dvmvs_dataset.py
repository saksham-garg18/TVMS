import numpy as np
import cv2
from pandaset import DataSet
from pandaset import geometry
from scipy.spatial.transform import Rotation
import yaml
import os
from tqdm import tqdm
import pandas as pd

# --- 1. CONFIGURE YOUR PATHS AND SETTINGS ---
PANDASET_ROOT_PATH = "/mnt/d/saksham/college stuff/Munjal/Fifth Semester/classroom/DIP/DIP_DVMS/Pandaset 3 scenes"
SEQUENCE_ID = "047"

OUTPUT_PATH = f"./processed_pandaset/{SEQUENCE_ID}"

TRAIN_WIDTH = 640
TRAIN_HEIGHT = 384


# --- Main Script ---

print(f"Creating output directories at: {OUTPUT_PATH}")
os.makedirs(os.path.join(OUTPUT_PATH, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "depth"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "training_npz"), exist_ok=True)

print(f"Loading PandaSet data from: {PANDASET_ROOT_PATH}")
dataset = DataSet(PANDASET_ROOT_PATH)
sequence = dataset[SEQUENCE_ID]
sequence.load_camera() 
print(f"Successfully loaded sequence {SEQUENCE_ID} camera data.")


with open(os.path.join(PANDASET_ROOT_PATH, "static_extrinsic_calibration.yaml"), 'r') as f:
    calib_data = yaml.safe_load(f)
front_cam_k = calib_data['front_camera']['intrinsic']['K']
K_matrix = np.array(front_cam_k).reshape(3, 3)
np.savetxt(os.path.join(OUTPUT_PATH, 'K.txt'), K_matrix)
print("Saved K.txt")


num_frames = len(sequence.camera['front_camera'].poses)
print(f"Found {num_frames} frames. Starting processing...")
all_poses_for_txt = []

for i in tqdm(range(num_frames), desc=f"Processing Sequence {SEQUENCE_ID}"):

    image_pil = sequence.camera['front_camera'][i]
    camera_pose_data = sequence.camera['front_camera'].poses[i]
    camera_intrinsics = sequence.camera['front_camera'].intrinsics
    
    image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(OUTPUT_PATH, f"images/{i:05d}.png"), image_cv2)

    lidar_file_path = os.path.join(PANDASET_ROOT_PATH, SEQUENCE_ID, 'lidar', f'{i:02d}.pkl')
    lidar_df = pd.read_pickle(lidar_file_path)
    lidar_points_world = lidar_df.to_numpy()[:, :3]

    # Project LiDAR to Image to create Depth Map
    points2d, points3d_cam, _ = geometry.projection(
        lidar_points=lidar_points_world,
        camera_pose=camera_pose_data,
        camera_intrinsics=camera_intrinsics,
        camera_data=image_pil
    )

    image_h, image_w = image_pil.height, image_pil.width
    depth_map = np.zeros((image_h, image_w), dtype=np.float32)

    for p2d, p3d in zip(points2d, points3d_cam):
        u, v = int(p2d[0]), int(p2d[1])
        depth = p3d[2]
        if 0 <= u < image_w and 0 <= v < image_h:
            if depth_map[v, u] == 0 or depth < depth_map[v, u]:
                depth_map[v, u] = depth

    # Save the original full-resolution depth map (in mm)
    depth_map_mm = (depth_map * 1000).astype(np.uint16)
    cv2.imwrite(os.path.join(OUTPUT_PATH, f"depth/{i:05d}.png"), depth_map_mm)

    # --- CREATE AND SAVE .NPZ TRAINING ARCHIVE ---
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (TRAIN_WIDTH, TRAIN_HEIGHT), interpolation=cv2.INTER_LINEAR)

    depth_resized = cv2.resize(depth_map, (TRAIN_WIDTH, TRAIN_HEIGHT), interpolation=cv2.INTER_NEAREST)

    np.savez_compressed(
        os.path.join(OUTPUT_PATH, "training_npz", f"{i:05d}.npz"),
        image=image_resized,
        depth=depth_resized
    )

    pos = camera_pose_data['position']
    head = camera_pose_data['heading']
    rot_matrix = Rotation.from_quat([head['x'], head['y'], head['z'], head['w']]).as_matrix()
    pose_matrix = np.eye(4)
    pose_matrix[0:3, 0:3] = rot_matrix
    pose_matrix[0:3, 3] = [pos['x'], pos['y'], pos['z']]
    all_poses_for_txt.append(pose_matrix.flatten())

np.savetxt(os.path.join(OUTPUT_PATH, 'poses.txt'), np.array(all_poses_for_txt))

print(f"\nProcessing for sequence {SEQUENCE_ID} complete!")
print(f"Your dataset is ready at: {OUTPUT_PATH}")
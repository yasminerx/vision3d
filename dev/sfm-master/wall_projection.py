import numpy as np
import cv2
import os

def estimate_wall_plane(points_3d):
    """
    Estimate the best-fit plane for the 3D points.
    Returns the plane normal and distance from origin.
    """
    # Remove NaN and inf
    points = points_3d[np.isfinite(points_3d).all(axis=1)]
    
    if len(points) < 3:
        print("Warning: Not enough valid points to estimate plane")
        return np.array([0, 0, 1]), 10
    
    # Compute centroid
    centroid = points.mean(axis=0)
    
    # Center points
    centered_points = points - centroid
    
    # SVD to find the plane normal (direction of minimum variance)
    U, S, Vt = np.linalg.svd(centered_points)
    normal = Vt[-1]  # Smallest singular value direction
    
    # Ensure normal points in positive Z direction
    if normal[2] < 0:
        normal = -normal
    
    # Distance from origin
    distance = np.dot(normal, centroid)
    
    print(f"Estimated wall plane: normal={normal}, distance={distance:.2f}")
    
    return normal, distance

def project_image_to_wall(image, K, T, wall_normal, wall_distance):
    """
    Project an image onto a planar wall using proper 3D homography.
    """
    h, w = image.shape[:2]
    
    # Create a grid of image coordinates
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    img_coords = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=2).reshape(-1, 3)
    
    # Back-project to 3D rays in camera space
    K_inv = np.linalg.inv(K[:3, :3])
    rays_cam = img_coords @ K_inv.T
    
    # Convert rays to world coordinates
    T_inv = np.linalg.inv(T)
    cam_center_world = T_inv[:3, 3]
    R_inv = T_inv[:3, :3]
    
    rays_world = rays_cam @ R_inv.T
    
    # Ray-plane intersection: find t such that cam_center + t*ray intersects plane
    # Plane equation: normal · (P - origin) = distance
    # P = cam_center + t * ray
    # normal · (cam_center + t*ray) = distance
    # t = (distance - normal·cam_center) / (normal·ray)
    
    denominator = rays_world @ wall_normal
    numerator = wall_distance - np.dot(cam_center_world, wall_normal)
    
    # Avoid division by zero
    t = np.zeros(len(rays_world))
    valid = np.abs(denominator) > 1e-6
    t[valid] = numerator / denominator[valid]
    
    # Compute 3D points on plane
    points_3d = cam_center_world[np.newaxis, :] + t[:, np.newaxis] * rays_world
    
    # Project back to image to see the result
    points_img = K[:3, :3] @ points_3d.T
    points_img = points_img[:2] / points_img[2]
    points_img = points_img.T.reshape(h, w, 2).astype(np.float32)
    
    # Warp image
    warped = cv2.remap(image, points_img[:,:,0], points_img[:,:,1], cv2.INTER_LINEAR)
    
    return warped

def main():
    # Load data - chercher le fichier à plusieurs emplacements possibles
    possible_paths = [
        "results/out.npz",
        "../../sfm-master/results/out.npz",
        "../../../sfm-master/results/out.npz",
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print("Error: out.npz not found at any of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nRun: python3 run.py -i data/horizontal_4m80_cropped/ first")
        return
    
    print(f"Loading data from: {data_path}")
    data = np.load(data_path)
    K = data['K']
    Ts = data['T']
    Ps = data['P']  # Points 3D
    
    print(f"Loaded {len(Ts)} camera poses")
    print(f"Loaded {len(Ps)} 3D points")
    
    # Estimate wall plane from 3D points
    wall_normal, wall_distance = estimate_wall_plane(Ps)
    
    # Load images
    img_folder = "data/horizontal_4m80_cropped/"
    image_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.png')])
    
    # Extract image numbers
    img_nums = [int(f.split('_')[1].split('.')[0]) for f in image_files]
    
    # Important: avec --step 3, on a pris 1 image sur 3
    # Donc les poses correspondent aux images : [0, 3, 6, 9, 12, ...]
    STEP = 3
    
    # Sélectionner les poses (toutes, car elles correspondent déjà aux images subsampled)
    selected_indices = list(range(len(Ts)))
    print(f"Processing {len(selected_indices)} camera poses")
    print(f"Note: Images were subsampled with step={STEP}")
    
    # Create canvas for projection
    canvas = np.zeros((800, 1200, 3), dtype=np.uint8)
    weights = np.zeros((800, 1200), dtype=np.float32)
    
    for pose_idx, img_idx in enumerate(range(0, len(image_files), STEP)):
        if pose_idx >= len(Ts):
            break
        
        # Get corresponding image
        img_file = image_files[img_idx]
        img_path = os.path.join(img_folder, img_file)
        
        print(f"Processing pose {pose_idx}: image {img_idx} ({img_file})")
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"  Error: Could not load {img_path}")
            continue
        
        # Project to wall
        warped = project_image_to_wall(image, K, Ts[pose_idx], wall_normal, wall_distance)
        
        if warped is None:
            print(f"  Warning: Failed to project pose {pose_idx}")
            continue
        
        # Resize to fit canvas
        h, w = warped.shape[:2]
        if h > 0 and w > 0:
            scale = min(800/h, 1200/w)
            new_h, new_w = int(h * scale), int(w * scale)
            warped_resized = cv2.resize(warped, (new_w, new_h))
            
            # Blend onto canvas
            y_start = (800 - new_h) // 2
            x_start = (1200 - new_w) // 2
            
            y_end = min(y_start + new_h, 800)
            x_end = min(x_start + new_w, 1200)
            
            mask = (warped_resized[:y_end-y_start, :x_end-x_start] > 0).any(axis=2)
            canvas[y_start:y_end, x_start:x_end][mask] = warped_resized[:y_end-y_start, :x_end-x_start][mask]
            weights[y_start:y_end, x_start:x_end][mask] += 1
        
        print(f"  OK")
    
    # Save result
    cv2.imwrite("wall_projection.png", canvas)
    print("\nSaved to: wall_projection.png")
    
    # Also save a debug visualization
    cv2.imshow("Wall Projection", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

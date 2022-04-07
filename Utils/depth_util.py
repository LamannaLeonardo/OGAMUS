# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

import Configuration


def pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(np.int)
    y = np.linspace(0, height - 1, height).astype(np.int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


def intrinsic_from_fov(height, width, fov=Configuration.FOV):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])


def get_point_cloud(depth_matrix):
    # Get intrinsic parameters
    height, width = depth_matrix.shape
    K = intrinsic_from_fov(height, width, Configuration.FOV)
    K_inv = np.linalg.inv(K)

    # Get pixel coordinates
    pixel_coords = pixel_coord_np(width, height)  # [3, npoints]

    # Apply back-projection: K_inv @ pixels * depth
    cam_coords = K_inv[:3, :3] @ pixel_coords * depth_matrix.flatten()

    return cam_coords



# =========================================================
# Geometry
# Also look at https://math.stackexchange.com/questions/2796055/3d-coordinate-rotation-using-roll-pitch-yaw
# =========================================================
def rotation_from_euler(roll=0, pitch=0, yaw=0):
    """
    Get rotation matrix
    Args:
        roll, pitch, yaw:       In radians
    Returns:
        R:          [4, 4]
    """
    si, sj, sk = np.sin(roll), np.sin(pitch), np.sin(yaw)
    ci, cj, ck = np.cos(roll), np.cos(pitch), np.cos(yaw)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    R = np.identity(4)
    R[0, 0] = cj * ck  # cos(pitch) * cos(yaw)
    R[0, 1] = sj * sc - cs  # sin(pitch) * sin(roll) * cos(yaw) - cos(roll) * sin(yaw)
    R[0, 2] = sj * cc + ss
    R[1, 0] = cj * sk
    R[1, 1] = sj * ss + cc
    R[1, 2] = sj * cs - sc
    R[2, 0] = -sj
    R[2, 1] = cj * si
    R[2, 2] = cj * ci
    return R


def get_xyz_point_from_depth(depth_matrix, angle, cam_angle, pos):#, min_row, min_col, max_row, max_col):

        angle = (angle - 90) % 360  # rescale angle according to simulator reference system

        # Limit points to 150m in the z-direction for visualisation
        # cam_coords = cam_coords[:, np.where(cam_coords[2] <= 150)[0]]
        x, y, z = get_point_cloud(depth_matrix)

        # flip the y-axis to positive upwards
        cam_coords = np.array((x, -y, z))


        # Filter cam coordinates according to agent view horizon
        cam_coords = cam_coords[:, np.where(cam_coords[2] <= 30)[0]]

        # Filter cam coordinates according to agent height
        # cam_coords = cam_coords[:, np.where(cam_coords[1] <= 0)[0]]
        # cam_coords = cam_coords[:, np.where(cam_coords[1] >= -1.5)[0]]

        # # Visualize 3D point cloud
        # pcd_cam = o3d.geometry.PointCloud()
        # pcd_cam.points = o3d.utility.Vector3dVector(cam_coords.T[:, :3])
        # # Flip it, otherwise the pointcloud will be upside down
        # pcd_cam.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # o3d.visualization.draw_geometries([pcd_cam])

        # Do top view projection
        # Get camera points
        x, y, z = cam_coords

        # Get agent depth view occupancy points
        occupancy_points = np.column_stack((x, z, y))
        # rot_matrix = np.array(([np.cos(np.deg2rad(angle)), - np.sin(np.deg2rad(angle))],
        #                        [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]))
        rot_matrix = rotation_from_euler(yaw=np.deg2rad(angle), roll=np.deg2rad(cam_angle))

        # Rotate agent view according to agent orientation
        occupancy_points = np.dot(occupancy_points, rot_matrix.T[:3, :3])

        # Add agent offset position to agent view
        # occupancy_points[:, 0] += pos['x']
        # occupancy_points[:, 1] += pos['y']
        # considered_x = np.reshape(occupancy_points[:, 0], (224, 224))[min_col:max_col]
        # considered_y = np.reshape(occupancy_points[:, 0], (224, 224))[min_row:max_row]

        # x, y = np.mean(occupancy_points[:, 0]), np.mean(occupancy_points[:, 1])

        obj_x = [x + pos['x'] for x in occupancy_points[:, 0] if x != np.nan]
        obj_y = [y + pos['y'] for y in occupancy_points[:, 1] if y != np.nan]
        obj_z = [z + pos['z'] for z in occupancy_points[:, 2] if z != np.nan]
        x, y, z = np.mean(obj_x), np.mean(obj_y), np.mean(obj_z)

        return x, y, z


def get_xy_point_from_depth(depth_matrix, angle, pos):#, min_row, min_col, max_row, max_col):

        angle = (angle - 90) % 360  # rescale angle according to simulator reference system

        # Limit points to 150m in the z-direction for visualisation
        # cam_coords = cam_coords[:, np.where(cam_coords[2] <= 150)[0]]
        x, y, z = get_point_cloud(depth_matrix)

        # flip the y-axis to positive upwards
        cam_coords = np.array((x, -y, z))


        # Filter cam coordinates according to agent view horizon
        cam_coords = cam_coords[:, np.where(cam_coords[2] <= 30)[0]]

        # Filter cam coordinates according to agent height
        # cam_coords = cam_coords[:, np.where(cam_coords[1] <= 0)[0]]
        # cam_coords = cam_coords[:, np.where(cam_coords[1] >= -1.5)[0]]

        # # Visualize 3D point cloud
        # pcd_cam = o3d.geometry.PointCloud()
        # pcd_cam.points = o3d.utility.Vector3dVector(cam_coords.T[:, :3])
        # # Flip it, otherwise the pointcloud will be upside down
        # pcd_cam.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # o3d.visualization.draw_geometries([pcd_cam])

        # Do top view projection
        # Get camera points
        x, y, z = cam_coords

        # Get agent depth view occupancy points
        occupancy_points = np.column_stack((x, z))
        rot_matrix = np.array(([np.cos(np.deg2rad(angle)), - np.sin(np.deg2rad(angle))],
                               [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]))

        # Rotate agent view according to agent orientation
        occupancy_points = np.dot(occupancy_points, rot_matrix.T)

        # Add agent offset position to agent view
        # occupancy_points[:, 0] += pos['x']
        # occupancy_points[:, 1] += pos['y']
        # considered_x = np.reshape(occupancy_points[:, 0], (224, 224))[min_col:max_col]
        # considered_y = np.reshape(occupancy_points[:, 0], (224, 224))[min_row:max_row]

        # x, y = np.mean(occupancy_points[:, 0]), np.mean(occupancy_points[:, 1])

        obj_x = [x + pos['x'] for x in occupancy_points[:, 0] if x != np.nan]
        obj_y = [y + pos['y'] for y in occupancy_points[:, 1] if y != np.nan]
        x, y = np.mean(obj_x), np.mean(obj_y)

        return x, y


def get_xz_point_from_depth(depth_matrix, angle, pos):#, min_row, min_col, max_row, max_col):

        angle = (angle - 90) % 360  # rescale angle according to simulator reference system

        # Limit points to 150m in the z-direction for visualisation
        # cam_coords = cam_coords[:, np.where(cam_coords[2] <= 150)[0]]
        x, y, z = get_point_cloud(depth_matrix)

        # flip the y-axis to positive upwards
        cam_coords = np.array((x, -y, z))


        # Filter cam coordinates according to agent view horizon
        cam_coords = cam_coords[:, np.where(cam_coords[2] <= 20)[0]]  # ???

        # Filter cam coordinates according to agent height
        # cam_coords = cam_coords[:, np.where(cam_coords[1] <= 0)[0]]
        cam_coords = cam_coords[:, np.where(cam_coords[1] >= -1.5)[0]]

        # # Visualize 3D point cloud
        # pcd_cam = o3d.geometry.PointCloud()
        # pcd_cam.points = o3d.utility.Vector3dVector(cam_coords.T[:, :3])
        # # Flip it, otherwise the pointcloud will be upside down
        # pcd_cam.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # o3d.visualization.draw_geometries([pcd_cam])

        # Do top view projection
        # Get camera points
        x, y, z = cam_coords

        # Get agent depth view occupancy points
        occupancy_points = np.column_stack((x, y))
        # rot_matrix = np.array(([np.cos(np.deg2rad(angle)), - np.sin(np.deg2rad(angle))],
        #                        [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]))

        # Rotate agent view according to agent orientation
        # occupancy_points = np.dot(occupancy_points, rot_matrix.T)

        # Add agent offset position to agent view
        # occupancy_points[:, 0] += pos['x']
        # occupancy_points[:, 1] += pos['y']
        # considered_x = np.reshape(occupancy_points[:, 0], (224, 224))[min_col:max_col]
        # considered_y = np.reshape(occupancy_points[:, 0], (224, 224))[min_row:max_row]

        # x, y = np.mean(occupancy_points[:, 0]), np.mean(occupancy_points[:, 1])

        obj_x = [x + pos['x'] for x in occupancy_points[:, 0] if x != np.nan]
        obj_z = [z + pos['z'] for z in occupancy_points[:, 1] if z != np.nan]
        x, z = np.mean(obj_x), np.mean(obj_z)

        return x, z

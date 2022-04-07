# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

import Configuration
from OGAMUS.Learn.EnvironmentModels.MapModel import MapModel
from Utils.depth_util import rotation_from_euler


class Mapper:

    def __init__(self):
        # Set map model
        self.map_model = MapModel()

        # Optimize top view updates
        self.all_occupancy_pts = []
        self.all_pos = []
        self.all_angles = []


    def get_point_cloud(self, depth_matrix):

        # Get intrinsic parameters
        height, width = depth_matrix.shape
        K = self.intrinsic_from_fov(height, width, Configuration.FOV)
        K_inv = np.linalg.inv(K)

        # Get pixel coordinates
        pixel_coords = self.pixel_coord_np(width, height)  # [3, npoints]

        # Rotation matrix
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        R_inv = np.linalg.inv(R)

        # Apply back-projection: K_inv @ R_inv @ pixels * depth
        cam_coords = K_inv[:3, :3] @ R_inv @ pixel_coords * depth_matrix.flatten()

        return cam_coords


    def translation_matrix(self, vector):
        """
        Translation matrix
        Args:
            vector list[float]:     (x, y, z)
        Returns:
            T:      [4, 4]
        """
        M = np.identity(4)
        M[:3, 3] = vector[:3]
        return M

    # =========================================================
    # Geometry
    # Also look at https://math.stackexchange.com/questions/2796055/3d-coordinate-rotation-using-roll-pitch-yaw
    # =========================================================
    def rotation_from_euler(self, roll=0, pitch=0, yaw=0):
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


    def update_topview(self, depth_matrix, file_name, angle, cam_angle, pos, collision=False):

        angle = (angle - 90) % 360  # rescale angle according to simulator reference system

        x, y, z = self.get_point_cloud(depth_matrix)

        # flip the y-axis to positive upwards
        cam_coords = np.array((x, -y, z))

        # Filter cam coordinates according to agent view horizon
        cam_coords = cam_coords[:, np.where(cam_coords[2] <= 20)[0]]  # ???

        # Get camera points
        x, y, z = cam_coords

        # Get agent depth view occupancy points
        occupancy_points = np.column_stack((x, z, y))
        rot_matrix = rotation_from_euler(yaw=np.deg2rad(angle), roll=np.deg2rad(cam_angle))

        # Rotate agent view according to agent orientation
        occupancy_points = np.dot(occupancy_points, rot_matrix.T[:3, :3])

        # Add agent offset position to agent view
        occupancy_points[:, 0] += pos['x']
        occupancy_points[:, 1] += pos['y']
        # occupancy_points[:, 2] += pos['z']

        filtered_occ_pts = np.array([pt for pt in occupancy_points if pt[2] <= 0 and pt[2] >= -Configuration.CAMERA_HEIGHT])

        # Update map model occupancy points
        self.map_model.update_occupancy(filtered_occ_pts, pos, angle, file_name, collision)


    def pixel_coord_np(self, width, height):
        """
        Pixel in homogenous coordinate
        Returns:
            Pixel coordinate:       [3, width * height]
        """
        x = np.linspace(0, width - 1, width).astype(np.int)
        y = np.linspace(0, height - 1, height).astype(np.int)
        [x, y] = np.meshgrid(x, y)
        return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


    def intrinsic_from_fov(self, height, width, fov=Configuration.FOV):
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

        return np.array([[fx,  0, px, 0.],
                         [ 0, fy, py, 0.],
                         [ 0,  0, 1., 0.],
                         [0., 0., 0., 1.]])



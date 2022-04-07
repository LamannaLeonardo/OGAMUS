# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import matplotlib.pyplot as plt
import Configuration
import numpy as np

from PIL import Image


class MapModel:

    def __init__(self):

        # x min coordinate in centimeters
        self.x_min = Configuration.MAP_X_MIN

        # y min coordinate in centimeters
        self.y_min = Configuration.MAP_Y_MIN

        # x max coordinate in centimeters
        self.x_max = Configuration.MAP_X_MAX

        # y max coordinate in centimeters
        self.y_max = Configuration.MAP_Y_MAX

        # x and y centimeters per pixel in the resized grid obstacle map
        self.dx = Configuration.MAP_GRID_DX
        self.dy = Configuration.MAP_GRID_DY

        # Initial collision cells
        self.collision_cells = []

        # x axis length on map (centimeters)
        self.x_axis_len = abs(self.x_max - self.x_min)

        # y axis length on map (centimeters)
        self.y_axis_len = abs(self.y_max - self.y_min)

        # self.fig, self.axes = plt.subplots(figsize=(30,30))  # 3000 x 3000 pixel
        self.fig, self.axes = plt.subplots(figsize=(int(self.y_axis_len/100), # 1 is 100 pixel, 1 pixel is 1 centimeter
                                                    int(self.x_axis_len/100)))

        # Set figure window size
        window_x = (int(self.x_min/100), int(self.x_max/100))
        window_y = (int(self.y_min/100), int(self.y_max/100))
        self.axes.set_xlim(window_x)
        self.axes.set_ylim(window_y)

        self.grid = None


    def update_occupancy(self, occupancy_points, pos, angle, file_name, collision=False):

        # Draw depth occupancy points
        self.axes.plot(occupancy_points[:,0], occupancy_points[:,1], 'o', markerfacecolor='none',
                       markeredgecolor='k', alpha=0.5, markersize=1)

        if collision:
            self.print_top_view(pos, angle, file_name)


    def print_top_view(self, pos, angle, file_name):

        # Draw agent arrow
        agent_width = 0.2  # 20 centimeters
        agent_arrow = self.axes.arrow(pos['x'], pos['y'],
                        agent_width * np.cos(np.deg2rad(angle + 90)), agent_width * np.sin(np.deg2rad(angle + 90)),
                        head_width=agent_width, head_length=agent_width, # 1.25 for aesthetic reason
                        length_includes_head=True, fc="Red", ec="Red", alpha=0.9)

        plt.gca().set_aspect('equal')

        # Print agent top view map
        if Configuration.PRINT_TOP_VIEW_IMAGES:
            plt.savefig("{}/topview_hd.png".format("/".join((file_name.split("/")[:-1]))))

        # Remove agent arrow and white margins
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        # self.axes.artists = []
        agent_arrow.remove()
        plt.axis('off')
        self.fig.canvas.draw()
        self.set_grid()


    def set_grid(self):

        # Rescale agent top view
        self.grid = Image.frombytes('RGB', self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb()).convert('L')
        self.grid.thumbnail((round(self.y_axis_len/self.dy), round(self.x_axis_len/self.dx)), Image.ANTIALIAS)
        self.grid = np.array(self.grid)

        # Binarize rescaled agent top view
        self.grid[(self.grid < 245)] = 0  # 245 is an heuristic threshold
        self.grid[(self.grid >= 245)] = 1



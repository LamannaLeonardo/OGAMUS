# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import math
import random
import numpy as np

import Configuration
from OGAMUS.Plan.PathPlanner import PathPlanner
from OGAMUS.Plan.PDDLPlanner import PDDLPlanner
from Utils import Logger


class EventPlanner:

    def __init__(self, map_model):
        random.seed(0)

        # Set path planner
        self.path_planner = PathPlanner(map_model)

        # Set pddl planner
        self.pddl_planner = PDDLPlanner()

        # Set pddl plan
        self.pddl_plan = None

        # Set path plan
        self.path_plan = None

        # Set event plan
        self.event_plan = None

        # Current agent state
        self.all_objects = None
        self.visible_objects = None
        self.perceptions = None

        # Current event planner subgoal
        self.subgoal = None

        # Object goal position
        self.goal_obj_position = None

        # Agent goal pose, used to reach states from which the goal objects are visible and recognizable
        self.goal_pose = None

        # Rotate direction when moving agents
        self.rotate_dir = Configuration.ROTATE_DIRECTION

        # Goal cells already reached when executing the action 'INSPECT', i.e., goal cells from which the goal
        # object is not recognized.
        self.useless_goal_cells = []

        # Flag that indicates whether the agent has collided during rotation with an held object
        self.rotation_collision = False


    def plan(self, fsm_model, agent_pos):

        # Compute a pddl plan
        self.pddl_plan = self.pddl_planner.pddl_plan()

        # If no pddl plan can be computed explore the environment to learn new constants and predicates
        if self.pddl_plan is None:
            action = self.explore(agent_pos)
            self.subgoal = None

            # Adjust agent camera inclination
            if -int(self.perceptions[7]) < 0:
                action = 'LookUp'
            elif -int(self.perceptions[7]) > 0:
                action = 'LookDown'

        # Set first pddl plan action as an event planner subgoal
        else:

            if self.subgoal is None or self.subgoal != self.pddl_plan[0]:
                self.subgoal = self.pddl_plan.pop(0)
                Logger.write('Changing event planner subgoal to: {}'.format(self.subgoal))
                self.event_plan = None
                self.useless_goal_cells = []

            action = self.event_planning(fsm_model)

            # # If the previous subgoal has been reached, look for a new subgoal
            while self.subgoal is None:
                self.event_plan = None
                self.useless_goal_cells = []
                self.subgoal = self.pddl_plan.pop(0)
                Logger.write('Changing event planner subgoal to: {}'.format(self.subgoal))
                action = self.event_planning(fsm_model)

        return action


    def explore(self, agent_pos):

        # if self.path_plan is None:
        if self.path_plan is None or Configuration.STOCASTIC_AGENT:
            self.path_plan = self.path_planner.path_planning()

        # Adjust plan whether the agent has collided while rotating and holding an object
        if self.rotation_collision and self.path_plan is not None:
            self.adjust_path_plan_rotations()
            self.rotation_collision = False

        # while self.path_plan is None:
        agent_x_pos = agent_pos['x']
        agent_y_pos = agent_pos['y']
        sampling_distance = 150  # centimeters

        if Configuration.TASK == Configuration.TASK_OGN_ROBOTHOR:
            sampling_distance = 1000

        explorable = True
        while self.path_plan is None or len(self.path_plan) == 0:

            if explorable:
                self.goal_position = [(Configuration.MAP_X_MIN + (Configuration.MOVE_STEP * 100 * 2)) / 100,
                                      (Configuration.MAP_Y_MIN + (Configuration.MOVE_STEP * 100 * 2)) / 100]
            else:
                self.path_planner.goal_position = [random.randint(max(self.path_planner.map_model.x_min + (Configuration.MOVE_STEP*100*2),
                                                                      int(agent_x_pos*100) - sampling_distance),
                                                                  min(self.path_planner.map_model.x_max - (Configuration.MOVE_STEP*100*2),
                                                                      int(agent_x_pos*100) + sampling_distance)) / 100,
                                                   random.randint(max(self.path_planner.map_model.y_min + (Configuration.MOVE_STEP * 100 * 2),
                                                                      int(agent_y_pos * 100) - sampling_distance),
                                                                  min(self.path_planner.map_model.y_max - (Configuration.MOVE_STEP * 100 * 2),
                                                                      int(agent_y_pos * 100) + sampling_distance)) / 100]
            print("New goal position is: {}".format(self.path_planner.goal_position))
            self.path_plan = self.path_planner.path_planning()
            explorable = False

            if self.path_plan is None and self.agent_is_blocked(agent_pos):
                break

        if self.path_plan is None and self.agent_is_blocked(agent_pos):
            Logger.write('Warning: agent is blocked, clearing the area around the agent.')
            self.free_agent_area(agent_pos)
            return 'RotateRight'

        return self.path_plan.pop(0)


    def agent_is_blocked(self, agent_position):

        # Get occupancy grid
        grid = copy.deepcopy(self.path_planner.get_occupancy_grid())

        # Add agent starting position into occupancy grid
        start = [agent_position['x']*100, agent_position['y']*100]
        start_grid = (int(round((start[0]-self.path_planner.map_model.x_min)/self.path_planner.map_model.dx)),
                      int(round((start[1]-self.path_planner.map_model.y_min)/self.path_planner.map_model.dy)))
        start_grid = (start_grid[0], grid.shape[0] - start_grid[1])  # starting column and row of the grid

        agent_blocked = grid[start_grid[1] + 1, start_grid[0]] == 0 \
                        and grid[start_grid[1] - 1, start_grid[0]] == 0 \
                        and grid[start_grid[1], start_grid[0] + 1] == 0 \
                        and grid[start_grid[1], start_grid[0] - 1] == 0

        return agent_blocked


    def free_agent_area(self, agent_position):

        # Get occupancy grid
        grid = self.path_planner.get_occupancy_grid()

        # Add agent starting position into occupancy grid
        start = [agent_position['x']*100, agent_position['y']*100]
        start_grid = (int(round((start[0]-self.path_planner.map_model.x_min)/self.path_planner.map_model.dx)),
                      int(round((start[1]-self.path_planner.map_model.y_min)/self.path_planner.map_model.dy)))
        start_grid = (start_grid[0], grid.shape[0] - start_grid[1])  # starting column and row of the grid

        if not Configuration.DIAGONAL_MOVE:
            agent_area_cells = [(start_grid[1] + 1, start_grid[0]),
                                (start_grid[1], start_grid[0] + 1),
                                (start_grid[1] - 1, start_grid[0]),
                                (start_grid[1], start_grid[0] - 1)]
        else:
            agent_area_cells = [(start_grid[1] + 1, start_grid[0]),
                                (start_grid[1], start_grid[0] + 1),
                                (start_grid[1] - 1, start_grid[0]),
                                (start_grid[1], start_grid[0] - 1),
                                (start_grid[1] + 1, start_grid[0] - 1),
                                (start_grid[1] - 1, start_grid[0] - 1),
                                (start_grid[1] + 1, start_grid[0] + 1),
                                (start_grid[1] - 1, start_grid[0] + 1)]

        for cell in agent_area_cells:
            self.path_planner.map_model.grid[cell[0], cell[1]] = 1

            if [self.path_planner.map_model.grid.shape[0] - cell[0], cell[1]] in self.path_planner.map_model.collision_cells:
                self.path_planner.map_model.collision_cells.remove([self.path_planner.map_model.grid.shape[0] - cell[0], cell[1]])


# Set the object position as a goal one
    def set_goal_obj_position(self, goal_object_id):

        # Get goal object information from current agent state
        goal_object_type = goal_object_id.split("_")[0].strip()
        goal_object = [obj for obj in self.all_objects[goal_object_type] if obj['id'] == goal_object_id][0]

        # Set event planner goal object position
        self.goal_obj_position = [goal_object['map_x'], goal_object['map_y'], goal_object['map_z']]

# Set the goal position as the position of a state where the object is visible
    def set_goal_obj_state_position(self, goal_object_id, fsm_model):

        # Get goal object information from current agent state
        goal_object_type = goal_object_id.split("_")[0].strip()

        goal_object_states = [s for s in fsm_model.states
                              if goal_object_type in s.visible_objects.keys()
                              and goal_object_id in [obj['id'] for obj in s.visible_objects[goal_object_type]
                                                     if obj['distance'] < Configuration.CLOSE_TO_OBJ_DISTANCE]]

        if len(goal_object_states) == 0:
            Logger.write('WARNING: there are no states where the goal object is within the manipulation '
                         'distance of {} meters.'.format(Configuration.CLOSE_TO_OBJ_DISTANCE))
            return

        # Choose the state which minimizes the goal object distance
        goal_object_states_distances = [[obj['distance'] for obj in s.visible_objects[goal_object_type]
                                         if obj['id'] == goal_object_id][0] for s in goal_object_states]

        feasible_goal = False

        # Get occupancy grid
        grid = self.path_planner.get_occupancy_grid()

        while not feasible_goal and len(goal_object_states) > 0:
            goal_state_index = np.argmin(goal_object_states_distances)
            goal_state = goal_object_states[goal_state_index]

            # Add goal cell marker into occupancy grid
            goal = [goal_state.perceptions[0] * 100, goal_state.perceptions[1] * 100]
            goal_grid = [int(round((goal[0] - self.path_planner.map_model.x_min) / self.path_planner.map_model.dx)),
                         int(round((goal[1] - self.path_planner.map_model.y_min) / self.path_planner.map_model.dy))]

            if grid[grid.shape[0] - goal_grid[1]][goal_grid[0]] != 0:
                feasible_goal = True
            else:
                del goal_object_states[goal_state_index]
                del goal_object_states_distances[goal_state_index]

        if not feasible_goal:
            Logger.write('WARNING: There are no feasible states to reach within a distance of 1.5 meters from the goal'
                         'object. Probably the agent is holding an object which collides and obstacles the path,'
                         'or (unlikely) the goal object position has been updated and moved to a collision grid cell.')
            return

        # Set event planner goal object position
        self.goal_pose = [goal_state.perceptions[0], goal_state.perceptions[1],
                          goal_state.perceptions[6], -int(goal_state.perceptions[7])]  # last number is the camera tilt


    def get_obj_bb_centroid(self, goal_object_id):

        # Get goal object information from current agent state
        goal_object_type = goal_object_id.split("_")[0].strip()
        goal_object = [obj for obj in self.all_objects[goal_object_type] if obj['id'] == goal_object_id][0]

        # Return object bbox center relative to agent view
        obj_bb_centroid = [goal_object['bb']['center'][0] / Configuration.FRAME_WIDTH,
                           goal_object['bb']['center'][1] / Configuration.FRAME_HEIGHT]

        return obj_bb_centroid


    def get_obj_bb_free_point(self, goal_object_id):

        # Get goal object information from current agent state
        goal_object_type = goal_object_id.split("_")[0].strip()
        goal_object = [obj for obj in self.all_objects[goal_object_type] if obj['id'] == goal_object_id][0]

        goal_bbox_x_range = np.arange(int(goal_object['bb']['corner_points'][0]),
                                      int(goal_object['bb']['corner_points'][2]) + 1)
        goal_bbox_y_range = np.arange(int(goal_object['bb']['corner_points'][1]),
                                      int(goal_object['bb']['corner_points'][3]) + 1)  # -2, -2 for error margin when rounding to target coordinates
        xx, yy = np.meshgrid(goal_bbox_x_range, goal_bbox_y_range)
        goal_bb_free_points = np.vstack([xx.ravel(), yy.ravel()])
        goal_bb_free_points = [pt for pt in zip(goal_bb_free_points[0], goal_bb_free_points[1])]

        for obj_type in [obj_type for obj_type in self.visible_objects.keys() if obj_type != goal_object_type]:

            for obj in [obj for obj in self.visible_objects[obj_type] if obj['id'] != goal_object['id']]:

                # Determine the coordinates of the intersection rectangle
                bb1 = {'xmin': int(goal_object['bb']['corner_points'][0]),
                       'ymin': int(goal_object['bb']['corner_points'][1]),
                       'xmax': int(goal_object['bb']['corner_points'][2]),
                       'ymax': int(goal_object['bb']['corner_points'][3])}
                bb2 = {'xmin': int(obj['bb']['corner_points'][0] - 2),
                       'ymin': int(obj['bb']['corner_points'][1]) - 2,
                       'xmax': int(obj['bb']['corner_points'][2]) + 2,
                       'ymax': int(obj['bb']['corner_points'][3]) + 2}
                x_left = max(bb1['xmin'], bb2['xmin'])
                y_top = max(bb1['ymin'], bb2['ymin'])
                x_right = min(bb1['xmax'], bb2['xmax'])
                y_bottom = min(bb1['ymax'], bb2['ymax'])

                # If the two bboxes intersect, and the intersection area is lower than a given threshold,
                # then remove intersection points from free space ones
                if x_right >= x_left and y_bottom >= y_top:

                    intersection_area = abs(x_right-x_left)*abs(y_bottom-y_top)\
                                        /int(goal_object['bb']['size'][0]*goal_object['bb']['size'][1])

                    if intersection_area < 0.75:

                        # Compute intersection points
                        intersect_bbox_x_range = np.arange(x_left, x_right + 1)
                        intersect_bbox_y_range = np.arange(y_top, y_bottom + 1)
                        xx, yy = np.meshgrid(intersect_bbox_x_range, intersect_bbox_y_range)
                        intersect_bb_points = np.vstack([xx.ravel(), yy.ravel()])
                        intersect_bb_points = [pt for pt in zip(intersect_bb_points[0], intersect_bb_points[1])]

                        # DEBUG
                        # print('Overlapping with: {} in {} points'.format(obj['id'], len(intersect_bb_points)))

                        # Remove intersection points from goal object bbox free points
                        goal_bb_free_points = [pt for pt in goal_bb_free_points if pt not in intersect_bb_points]

        # Select the goal object bbox free point which minimizes the distance from the bbox center
        goal_obj_bb_centroid = [int(goal_object['bb']['center'][0]), int(goal_object['bb']['center'][1])]
        free_points_distances = [np.linalg.norm(np.array(free_point)-np.array(goal_obj_bb_centroid))
                                 for free_point in goal_bb_free_points]

        # Check if there is at least one free point on goal object bbox
        if len(goal_bb_free_points) > 0:
            minimum_distance_free_point = list(goal_bb_free_points[np.argmin(free_points_distances)])
            minimum_distance_free_point = [minimum_distance_free_point[0] / Configuration.FRAME_WIDTH,
                                    minimum_distance_free_point[1] / Configuration.FRAME_HEIGHT]
        else:
            print('WARNING: Cannot find any free point on bbox of goal object:{}, using bbox center'.format(goal_object_id))
            goal_obj_bb_centroid = [goal_object['bb']['center'][0] / Configuration.FRAME_WIDTH,
                                    goal_object['bb']['center'][1] / Configuration.FRAME_HEIGHT]
            return goal_obj_bb_centroid

        return minimum_distance_free_point


    def event_planning(self, fsm_model):

        if self.subgoal.split("(")[0].strip() == "GET_CLOSE_AND_LOOK_AT_RECEPTACLE"\
                or self.subgoal.split("(")[0].strip() == "GET_CLOSE_AND_LOOK_AT_PICKUPABLE"\
                or self.subgoal.split("(")[0].strip() == "GET_CLOSE_AND_LOOK_AT_OPENABLE"\
                or self.subgoal.split("(")[0].strip() == "GET_CLOSE_AND_LOOK_AT":
            self.goal_pose = None
            goal_object_id = self.subgoal.split("(")[1].strip()[:-1].lower()
            self.set_goal_obj_state_position(goal_object_id, fsm_model)

            # Choose the state which minimizes the distance from the goal object and set the state position as a goal one
            if self.goal_pose is not None:
                self.path_planner.goal_position = self.goal_pose[:-2]  # last two numbers are agent and cam rotations

            # Set the object position as a goal one
            else:
                self.set_goal_obj_position(goal_object_id)
                self.path_planner.goal_position = self.goal_obj_position[:-1]  # last number is the z position

            if self.event_plan is None:
                if self.goal_pose is not None:
                    pass
                else:
                    Logger.write('Warning: the goal object has already been inspected, but the inspection state cannot '
                                 'be reached. It is likely that the agent holds an object which collides during the path')
                    self.useless_goal_cells = list(range(Configuration.MAX_USELESS_GOAL_CELLS))
                    agent_x = self.perceptions[0]
                    agent_y = self.perceptions[1]
                    start_position = {'x': agent_x, 'y': agent_y}
                    return self.explore(start_position)

                self.event_plan = self.path_planner.path_planning()

                # Adjust plan whether the agent has collided while rotating and holding an object
                if self.rotation_collision and self.event_plan is not None:
                    self.adjust_event_plan_rotations()
                    self.rotation_collision = False

                # Check if goal object is still reachable. E.g., if the agent collides while holding an other object
                # there could be no more path towards the goal object, hence it is removed from the knowledge base.
                if self.event_plan is None:
                    self.useless_goal_cells = list(range(Configuration.MAX_USELESS_GOAL_CELLS))
                    agent_x = self.perceptions[0]
                    agent_y = self.perceptions[1]
                    start_position = {'x': agent_x, 'y': agent_y}
                    return self.explore(start_position)

                # Add fictitious action, otherwise the subgoal is considered achieved because the event plan length is zero.
                self.event_plan.append('fictitious action')


            if len(self.event_plan) == 1:
                print('Close to goal object, planning to look at it.')

                self.event_plan = []

                # Get agent position
                agent_x = self.perceptions[0]
                agent_y = self.perceptions[1]
                goal_object_id = self.subgoal.split("(")[1].strip()[:-1].lower()
                goal_object_type = goal_object_id.split("_")[0].strip()
                goal_object = [obj for obj in self.all_objects[goal_object_type] if obj['id'] == goal_object_id][0]
                goal_obj_x = goal_object['map_x']
                goal_obj_y = goal_object['map_y']

                # Get angle between agent and goal object position
                agent_angle = int(round(self.perceptions[6]))  # rescale agent angle according to reference system
                cam_angle = -int(round(self.perceptions[7]))  # rescale agent angle according to reference system
                agent2obj_angle = int(
                    (math.degrees(math.atan2(goal_obj_y - agent_y, goal_obj_x - agent_x)) - agent_angle)) % 360

                agent2obj_z_angle = None
                if self.goal_pose is not None:
                    agent2obj_angle = int(self.goal_pose[-2] - agent_angle) % 360
                    agent2obj_z_angle = int(self.goal_pose[-1] - cam_angle)
                else:
                    print("WARNING: agent goal pose is None, check 'GET_CLOSE_AND_LOOK_AT' action in EventPlanner.py")

                # Set angle according to agent rotation direction (left or right)
                if self.rotate_dir == Configuration.ROTATE_RIGHT:
                    agent2obj_angle = 360 - agent2obj_angle

                while agent2obj_angle > 0:
                    if self.rotate_dir == Configuration.ROTATE_LEFT:
                        self.event_plan.append("RotateLeft")
                        agent2obj_angle -= Configuration.ROTATION_STEP
                    elif self.rotate_dir == Configuration.ROTATE_RIGHT:
                        self.event_plan.append("RotateRight")
                        agent2obj_angle -= Configuration.ROTATION_STEP

                if agent2obj_z_angle is None:
                    Logger.write('Warning: agent-object angle in xz is None, check GET_CLOSE_AND_LOOK_AT in EventPlanner.py')
                if agent2obj_z_angle is not None and agent2obj_z_angle > 0:
                    [self.event_plan.append('LookUp') for _ in range(agent2obj_z_angle // 30)] # assume 30 degrees of lookup
                elif agent2obj_z_angle is not None and agent2obj_z_angle < 0:
                    [self.event_plan.append('LookDown') for _ in range(agent2obj_z_angle // -30)] # assume 30 degrees of lookdown

            if len(self.event_plan) == 0:
                print('Goal object is in agent view.')
                self.subgoal = None
                self.event_plan = None
                self.goal_obj_position = None
            else:
                return self.event_plan.pop(0)


        elif self.subgoal.split("(")[0].strip() == "GET_CLOSE_TO_OGN":

            goal_object_id = self.subgoal.split("(")[1].strip()[:-1].lower()
            self.set_goal_obj_position(goal_object_id)
            self.path_planner.goal_position = self.goal_obj_position[:-1]

            self.event_plan = self.path_planner.path_planning_greedy_OGN(self.goal_obj_position)

            # Check if goal object is still reachable. E.g., if the agent collides while holding an other object
            # there could be no more path towards the goal object, hence it is removed from the knowledge base.
            if self.event_plan is None:
                self.useless_goal_cells = list(range(Configuration.MAX_USELESS_GOAL_CELLS))
                agent_x = self.perceptions[0]
                agent_y = self.perceptions[1]
                start_position = {'x': agent_x, 'y': agent_y}
                return self.explore(start_position)

            if len(self.event_plan) == 0:
                print('Close to goal object.')
                self.subgoal = None
                self.event_plan = None
                self.goal_obj_position = None
            else:
                return self.event_plan.pop(0)


        elif self.subgoal.split("(")[0].strip() == "LOOK_AT_OGN":

            if self.event_plan is None:
                self.event_plan = []

                # Get agent position
                agent_x = self.perceptions[0]
                agent_y = self.perceptions[1]
                goal_object_id = self.subgoal.split("(")[1].strip()[:-1].lower()
                goal_object_type = goal_object_id.split("_")[0].strip()
                goal_object = [obj for obj in self.all_objects[goal_object_type] if obj['id'] == goal_object_id][0]
                goal_obj_x = goal_object['map_x']
                goal_obj_y = goal_object['map_y']

                # Get angle between agent and goal object position
                agent_angle = int(round(self.perceptions[6]))
                # rescale agent angle according to ref sys
                agent2obj_angle = int((math.degrees(math.atan2(goal_obj_y - agent_y, goal_obj_x - agent_x)) - agent_angle)) % 360

                # Set angle according to agent rotation direction (left or right)
                if self.rotate_dir == Configuration.ROTATE_RIGHT:
                    agent2obj_angle = 360 - agent2obj_angle

                while agent2obj_angle > 0:
                    if self.rotate_dir == Configuration.ROTATE_LEFT:
                        self.event_plan.append("RotateLeft")
                        agent2obj_angle -= Configuration.ROTATION_STEP
                    elif self.rotate_dir == Configuration.ROTATE_RIGHT:
                        self.event_plan.append("RotateRight")
                        agent2obj_angle -= Configuration.ROTATION_STEP

                self.event_plan.extend(['LookDown', 'LookDown', 'LookUp', 'LookUp'])

            if len(self.event_plan) == 1:
                print('Cannot see goal object.')
                self.event_plan = None
                self.goal_obj_position = None
                self.useless_goal_cells = list(range(Configuration.MAX_USELESS_GOAL_CELLS))
                return 'LookUp'
            else:
                return self.event_plan.pop(0)


        elif self.subgoal.split("(")[0].strip() == "INSPECT":

            # Get agent position
            agent_x = self.perceptions[0]
            agent_y = self.perceptions[1]
            start_position = {'x': agent_x, 'y': agent_y}
            start = [start_position['x'] * 100, start_position['y'] * 100]
            start_grid = (int(round((start[0] - self.path_planner.map_model.x_min) / self.path_planner.map_model.dx)),
                          int(round((start[1] - self.path_planner.map_model.y_min) / self.path_planner.map_model.dy)))
            grid = self.path_planner.get_occupancy_grid()
            start_grid = (start_grid[0], grid.shape[0] - start_grid[1])  # starting column and row of the grid

            self.inspect()

            # Check if goal object inspection is feasible
            if self.event_plan is None:
                self.useless_goal_cells = list(range(Configuration.MAX_USELESS_GOAL_CELLS))
                return self.explore(start_position)

            if len(self.event_plan) == 1:

                # Check if goal object has been seen in a state where its distance is lower than the manipulation one
                goal_object_id = self.subgoal.split("(")[1].strip()[:-1].lower()
                goal_object_type = goal_object_id.split("_")[0].strip()
                goal_object_states = [s for s in fsm_model.states
                                      if goal_object_type in s.visible_objects.keys()
                                      and goal_object_id in [obj['id'] for obj in s.visible_objects[goal_object_type]
                                                             if obj['distance'] < Configuration.CLOSE_TO_OBJ_DISTANCE]]
                if len(goal_object_states) == 0:
                    self.event_plan = None
                    self.useless_goal_cells.append((start_grid[1], start_grid[0]))
                    Logger.write('Adding a useless goal cell: {}'.format(len(self.useless_goal_cells)))
                    self.inspect()

                    # Check if goal object inspection is feasible
                    if self.event_plan is None:
                        self.useless_goal_cells = list(range(Configuration.MAX_USELESS_GOAL_CELLS))
                        return self.explore(start_position)

                    return self.event_plan.pop(0)

                else:
                    print('Goal object has been successfully inspected.')
                    self.subgoal = None
                    self.event_plan = None
                    self.goal_obj_position = None
                    self.goal_pose = None
                    self.useless_goal_cells = []
            else:
                return self.event_plan.pop(0)

        elif self.subgoal.split("(")[0].strip() == "PICKUP":

            # Set goal object position
            if self.goal_obj_position is None:
                goal_object_id = self.subgoal.split("(")[1].strip()[:-1].lower()
                self.set_goal_obj_position(goal_object_id)

            if self.event_plan is None:
                # Pick up object
                goal_object_id = self.subgoal.split("(")[1].strip()[:-1].lower()
                goal_obj_bb_centroid = self.get_obj_bb_centroid(goal_object_id)

                self.event_plan = ["PickupObject|{:.2f}|{:.2f}".format(goal_obj_bb_centroid[0], goal_obj_bb_centroid[1])]

            if len(self.event_plan) == 0:
                self.subgoal = None
                self.event_plan = None
                self.goal_obj_position = None
            else:
                return self.event_plan.pop(0)

        elif self.subgoal.split("(")[0].strip() == "OPEN":

            # Set goal object position
            if self.goal_obj_position is None:
                goal_object_id = self.subgoal.split("(")[1].strip()[:-1].lower()
                self.set_goal_obj_position(goal_object_id)

            if self.event_plan is None:
                # Open object
                goal_object_id = self.subgoal.split("(")[1].strip()[:-1].lower()
                goal_obj_bb_centroid = self.get_obj_bb_centroid(goal_object_id)

                self.event_plan = ["OpenObject|{:.2f}|{:.2f}".format(goal_obj_bb_centroid[0], goal_obj_bb_centroid[1])]

            if len(self.event_plan) == 0:
                self.subgoal = None
                self.event_plan = None
                self.goal_obj_position = None
            else:
                return self.event_plan.pop(0)

        elif self.subgoal.split("(")[0].strip() == "CLOSE":

            # Set goal object position
            if self.goal_obj_position is None:
                goal_object_id = self.subgoal.split("(")[1].strip()[:-1].lower()
                self.set_goal_obj_position(goal_object_id)

            if self.event_plan is None:
                # Close object
                goal_object_id = self.subgoal.split("(")[1].strip()[:-1].lower()
                goal_obj_bb_centroid = self.get_obj_bb_centroid(goal_object_id)

                self.event_plan = ["CloseObject|{:.2f}|{:.2f}".format(goal_obj_bb_centroid[0], goal_obj_bb_centroid[1])]

            if len(self.event_plan) == 0:
                self.subgoal = None
                self.event_plan = None
                self.goal_obj_position = None
            else:
                return self.event_plan.pop(0)


        # Put an object into another one (e.g. put an apple into a box)
        elif self.subgoal.split("(")[0].strip() == "PUTINTO":

            # Set goal object position
            if self.goal_obj_position is None:
                goal_object_id = self.subgoal.split("(")[1].strip()[:-1].lower().split(",")[1]
                self.set_goal_obj_position(goal_object_id)

            if self.event_plan is None:
                # Look at object
                goal_object_id = self.subgoal.split("(")[1].strip()[:-1].lower().split(',')[1]
                goal_obj_bb_centroid = self.get_obj_bb_centroid(goal_object_id)
                # Put object
                self.event_plan = ["PutObject|{:.2f}|{:.2f}".format(goal_obj_bb_centroid[0], goal_obj_bb_centroid[1])]

            if len(self.event_plan) == 2:
                print("Ready to put down object.")
                return self.event_plan.pop(0)
            elif len(self.event_plan) == 0:
                self.subgoal = None
                self.event_plan = None
            else:
                return self.event_plan.pop(0)

        # Put an object on another one (e.g. put an apple on a table)
        elif self.subgoal.split("(")[0].strip() == "PUTON":

            # Set goal object position
            goal_object_id = self.subgoal.split("(")[1].strip()[:-1].lower().split(",")[1]
            self.set_goal_obj_position(goal_object_id)

            if self.event_plan is None:
                goal_object_id = self.subgoal.split("(")[1].strip()[:-1].lower().split(',')[1]
                goal_obj_bb_free_point = self.get_obj_bb_free_point(goal_object_id)
                # Put object
                self.event_plan = ["PutObject|{:.2f}|{:.2f}".format(goal_obj_bb_free_point[0], goal_obj_bb_free_point[1])]

            if len(self.event_plan) == 0:
                self.subgoal = None
                self.event_plan = None
            else:
                return self.event_plan.pop(0)

        elif self.subgoal.split("(")[0].strip() == "STOP":

            if self.event_plan is None:
                self.event_plan = ["Stop"]

            if len(self.event_plan) == 0:
                self.subgoal = None
                self.event_plan = None
            else:
                return self.event_plan.pop(0)


# Adjust plan whether the agent has collided while rotating and holding an object
    def adjust_event_plan_rotations(self):

            rotate_left = []
            rotate_right = []
            for action in self.event_plan:
                if action == 'RotateLeft':
                    rotate_left.append('RotateLeft')
                elif action == 'RotateRight':
                    rotate_right.append('RotateRight')
                else:
                    break

            self.event_plan = self.event_plan[len(rotate_left) + len(rotate_right):]

            assert len(rotate_right) == 0 or len(rotate_left) == 0, 'Check event_planning() in EventPlanner.py'

            complete_rotations = int(360 / Configuration.ROTATION_STEP)
            if len(rotate_right) > 0:
                for i in range(complete_rotations - len(rotate_right)):
                    self.event_plan = ['RotateLeft'] + self.event_plan
            elif len(rotate_left) > 0:
                for i in range(complete_rotations - len(rotate_left)):
                    self.event_plan = ['RotateRight'] + self.event_plan


# Adjust plan whether the agent has collided while rotating and holding an object
    def adjust_path_plan_rotations(self):

            rotate_left = []
            rotate_right = []
            for action in self.path_plan:
                if action == 'RotateLeft':
                    rotate_left.append('RotateLeft')
                elif action == 'RotateRight':
                    rotate_right.append('RotateRight')
                else:
                    break

            self.path_plan = self.path_plan[len(rotate_left) + len(rotate_right):]

            assert len(rotate_right) == 0 or len(rotate_left) == 0, 'Check adjust_path_plan_rotations() in EventPlanner.py'

            complete_rotations = int(360 / Configuration.ROTATION_STEP)
            if len(rotate_right) > 0:
                for i in range(complete_rotations - len(rotate_right)):
                    self.path_plan = ['RotateLeft'] + self.path_plan
            elif len(rotate_left) > 0:
                for i in range(complete_rotations - len(rotate_left)):
                    self.path_plan = ['RotateRight'] + self.path_plan


    def inspect(self):

        # Get agent position
        agent_x = self.perceptions[0]
        agent_y = self.perceptions[1]

        # Get goal object position
        goal_object_id = self.subgoal.split("(")[1].strip()[:-1].lower()
        goal_object_type = goal_object_id.split("_")[0].strip()
        goal_object = [obj for obj in self.all_objects[goal_object_type] if obj['id'] == goal_object_id][0]
        goal_obj_x = goal_object['map_x']
        goal_obj_y = goal_object['map_y']
        goal_obj_z = goal_object['map_z']
        goal_position = [goal_obj_x, goal_obj_y, goal_obj_z]
        start_position = {'x': agent_x, 'y': agent_y}

        # if self.event_plan is None:
        self.event_plan = self.path_planner.path_planning_greedy_inspect(start_position, goal_position,
                                                                         non_goal_grid_cells=self.useless_goal_cells)
        if self.event_plan is None:
            Logger.write('WARNING: goal object is not reachable, deleting it from the agent knowledge.')
            return

        # Append a fictitious action to see last event plan action results before confirming subgoal success
        if 'fictitious action' not in self.event_plan:
            self.event_plan.append('fictitious action')

        # Adjust plan whether the agent has collided while rotating and holding an object
        if self.rotation_collision and len([a for a in self.event_plan if a != 'fictitious action']) > 0:
            self.adjust_event_plan_rotations()
            self.rotation_collision = False

        # If the agent has reached a position close to the goal object, look at the goal object
        if len(self.event_plan) == 1:

            if not 'fictitious action' in self.event_plan:
                Logger.write('ERROR: look at inspect() in EventPlanner.py')
                exit()
            self.event_plan = []

            # Get angle between agent and goal object position
            agent_angle = int(round(self.perceptions[6]))  # rescale agent angle according to ref sys
            agent2obj_angle = int(
                (math.degrees(math.atan2(goal_obj_y - agent_y, goal_obj_x - agent_x)) - agent_angle)) % 360

            rotate_right = agent2obj_angle >= 180
            # Set angle according to agent rotation direction (left or right)
            if rotate_right:
                agent2obj_angle = 360 - agent2obj_angle

            while agent2obj_angle not in range(int(-Configuration.FOV / 2), int(Configuration.FOV / 2) + 1) \
                    and agent2obj_angle not in range(int(360 - Configuration.FOV / 2),
                                                     int(360 + Configuration.FOV / 2) + 1):

                if not rotate_right:
                    self.event_plan.append("RotateLeft")  # Assume a rotation of 90 degrees
                    agent2obj_angle -= Configuration.ROTATION_STEP
                else:
                    self.event_plan.append("RotateRight")  # Assume a rotation of 90 degrees
                    agent2obj_angle -= Configuration.ROTATION_STEP

            # Adjust plan whether the agent has collided while rotating and holding an object
            if self.rotation_collision:
                self.adjust_event_plan_rotations()
                self.rotation_collision = False

            camera_angle = -int(self.perceptions[7])
            agent2obj_z_angle = int((math.degrees(math.atan2((goal_obj_z) - Configuration.CAMERA_HEIGHT,
                                                             (goal_obj_y - agent_y)**2 + (goal_obj_x - agent_x)**2)) - camera_angle))

            if agent2obj_z_angle < -30 and camera_angle - 30 >= -Configuration.MAX_CAM_ANGLE:  # Assume an inclination step of 30 degrees
                look_down = []
                look_up = []
                while agent2obj_z_angle < -30:
                    look_down.append("LookDown")  # Assume an inclination of 30 degrees
                    agent2obj_z_angle += 30
                    # look_up.append('LookUp')
                self.event_plan = self.event_plan + look_down + look_up

            # elif agent2obj_z_angle > 30:
            elif agent2obj_z_angle > 30 and camera_angle + 30 <= Configuration.MAX_CAM_ANGLE:  # Assume an inclination step of 30 degrees
                look_down = []
                look_up = []
                while agent2obj_z_angle > 30:
                    look_up.append("LookUp")  # Assume an inclination of 30 degrees
                    agent2obj_z_angle -= 30
                    # look_down.append('LookDown')
                self.event_plan = self.event_plan + look_up + look_down

            # Append a fictitious action to see last event plan action results before confirming subgoal success
            if 'fictitious action' not in self.event_plan:
                self.event_plan.append('fictitious action')


    def look_at_object(self):

        # Event plan actions to look at a goal object, given its absolute position on the map
        event_plan_actions = []
        dtheta_agent = 90
        dtheta_camera = 30

        # Get agent position on map from state perceptions
        agent_x, agent_y, camera_z = self.perceptions[0], self.perceptions[1], self.perceptions[2]
        agent_angle = int(round(self.perceptions[6]))
        camera_angle = int(round(self.perceptions[7]))

        # Get goal object position on map
        goal_obj_x = self.goal_obj_position[0]
        goal_obj_y = self.goal_obj_position[1]
        goal_obj_z = self.goal_obj_position[2]

        # Compute angle between agent position and object goal point
        obj_agent_angle = np.rad2deg(math.atan2(goal_obj_y - agent_y, goal_obj_x - agent_x))

        # Add agent orientation
        obj_agent_angle = (obj_agent_angle - agent_angle) % 360

        # Compute number of necessary z-rotations to see goal object in agent view
        if obj_agent_angle > 0:
            # Rotate left
            [event_plan_actions.append("RotateLeft") for _ in range(int(obj_agent_angle / dtheta_agent))]
            event_plan_actions.append("RotateLeft|{}".format(obj_agent_angle % dtheta_agent))
        else:
            # Rotate right
            [event_plan_actions.append("RotateRight") for _ in range(abs(int(obj_agent_angle / dtheta_agent)))]
            event_plan_actions.append("RotateRight|{}".format(obj_agent_angle % dtheta_agent))

        # Get angle between object and camera on the z axis
        obj_camera_angle = np.rad2deg(math.atan2(goal_obj_z - camera_z, goal_obj_x - agent_x))

        # Add agent orientation
        obj_camera_angle = obj_camera_angle - camera_angle

        if - 180 - 45 <= obj_camera_angle <= - 180 + 45:  # 45 is vertical FOV / 2
            obj_camera_angle = -(obj_camera_angle + 180)

        # Compute number of necessary camera pitch rotations to see goal object in agent view
        if obj_camera_angle < 0:
            # Look down
            [event_plan_actions.append("LookDown") for _ in range(abs(int(obj_camera_angle / dtheta_camera)))]
            event_plan_actions.append("LookDown|{}".format(abs(obj_camera_angle % -dtheta_camera)))
        else:
            # Look up
            [event_plan_actions.append("LookUp") for _ in range(int(obj_camera_angle / dtheta_camera))]
            event_plan_actions.append("LookUp|{}".format(obj_camera_angle % dtheta_camera))

        self.goal_obj_position = None

        return event_plan_actions

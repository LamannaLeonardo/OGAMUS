# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import datetime
import os
import random
from collections import defaultdict

import numpy as np
from ai2thor.controller import Controller

import Configuration
from OGAMUS.Learn.Learner import Learner
from OGAMUS.Learn.EnvironmentModels.State import State
from OGAMUS.Plan.EventPlanner import EventPlanner
from Utils import Logger, PddlParser
from Utils.Evaluator import Evaluator

import matplotlib.pyplot as plt

class Agent:


    def __init__(self, scene="FloorPlan_Train1_1", position=None, init_rotation=None, init_horizon=None, shortest_path=None):

        # Set learner
        self.learner = Learner()

        # Set event planner
        self.event_planner = EventPlanner(self.learner.mapper.map_model)

        if Configuration.TASK == Configuration.TASK_OGN_ROBOTHOR:
            agent_mode = 'locobot'  # ai2thor 3.3.1
        else:
            agent_mode = 'default'

        hfov = Configuration.FOV / 360. * 2. * np.pi
        vfov = 2. * np.arctan(np.tan(hfov / 2) * Configuration.FRAME_HEIGHT / Configuration.FRAME_WIDTH)
        vfov = np.rad2deg(vfov)

        if Configuration.TASK == Configuration.TASK_OGN_ROBOTHOR:
            self.controller = Controller(renderDepthImage=Configuration.RENDER_DEPTH_IMG,
                                         renderObjectImage=True,
                                         visibilityDistance=Configuration.VISIBILITY_DISTANCE,
                                         gridSize=Configuration.MOVE_STEP,
                                         rotateStepDegrees=Configuration.ROTATION_STEP,
                                         scene=scene,
                                         # camera properties
                                         width=Configuration.FRAME_WIDTH,
                                         height=Configuration.FRAME_HEIGHT,
                                         fieldOfView=vfov,
                                         continuousMode=True,
                                         snapToGrid=False,
                                         agentMode=agent_mode
                                         )
        elif Configuration.TASK == Configuration.TASK_OGN_ITHOR:
            self.controller = Controller(renderDepthImage=Configuration.RENDER_DEPTH_IMG,
                                         renderObjectImage=True,
                                         visibilityDistance=Configuration.VISIBILITY_DISTANCE,
                                         gridSize=Configuration.MOVE_STEP,
                                         rotateStepDegrees=Configuration.ROTATION_STEP,
                                         scene=scene,
                                         continuousMode=True,
                                         snapToGrid=False,
                                         # camera properties
                                         width=Configuration.FRAME_WIDTH,
                                         height=Configuration.FRAME_HEIGHT,
                                         fieldOfView=vfov,
                                         agentMode=agent_mode
                                         )
        else:
            self.controller = Controller(renderDepthImage=Configuration.RENDER_DEPTH_IMG,
                                         renderObjectImage=True,
                                         visibilityDistance=Configuration.VISIBILITY_DISTANCE,
                                         gridSize=Configuration.MOVE_STEP,
                                         rotateStepDegrees=Configuration.ROTATION_STEP,
                                         scene=scene,
                                         # camera properties
                                         width=Configuration.FRAME_WIDTH,
                                         height=Configuration.FRAME_HEIGHT,
                                         fieldOfView=vfov,
                                         agentMode=agent_mode
                                         )

        # Initialize event (i.e. the observation after action execution)
        self.event = self.controller.step("Pass")

        if Configuration.TASK == Configuration.TASK_OGN_ROBOTHOR or Configuration.TASK == Configuration.TASK_OGN_ITHOR:
            self.controller.step(action="MakeAllObjectsStationary")

        if position is not None or init_rotation is not None or init_horizon is not None:
            assert position is not None and init_rotation is not None and init_horizon is not None, \
                " If you do not want to use the default initial agent pose, you should set: initial position, " \
                "rotation and horizon. See Agent.py constructor."

            if agent_mode == 'locobot':
                self.controller.step(
                    action="TeleportFull",
                    position=position,
                    rotation=dict(x=0, y=init_rotation, z=0),
                    horizon=init_horizon
                )
                Configuration.CAMERA_HEIGHT = 0.8
            else:
                self.controller.step(
                    action="TeleportFull",
                    position=position,
                    rotation=dict(x=0, y=init_rotation, z=0),
                    horizon=init_horizon,
                    standing=True
                )
                Configuration.CAMERA_HEIGHT = 1.5

            self.event = self.controller.step("Pass")

        # Set initial agent angle
        self.init_angle = self.event.metadata['agent']['rotation']['y']

        # Perceive the environment
        perceptions = self.perceive()

        # Update agent position in agent state and path planner state
        self.pos = {"x": int(), "y": int(), "z": int()}
        self.hand_pos = {"x": int(), "y": int(), "z": int()}
        self.angle = None
        # Update agent xyz position
        self.pos['x'], self.pos['y'], self.pos['z'] = perceptions[0], perceptions[1], perceptions[2]
        # Update agent xyz hand position
        self.hand_pos['x'], self.hand_pos['y'], self.hand_pos['z'] = perceptions[3], perceptions[4], perceptions[5]
        # Update agent y rotation
        self.angle = int(round(perceptions[6])) % 360

        # Set current iteration
        self.iter = 0

        # Create initial top view map
        self.learner.update_topview(os.path.join(Logger.LOG_DIR_PATH, "topview_{}.png".format(self.iter)),
                                    self.event.depth_frame, self.angle, int(self.event.metadata['agent']['cameraHorizon']), self.pos, collision=True)

        self.last_action_effects = None

        # Initialize initial state
        self.state = self.create_state(perceptions)
        self.learner.add_state(self.state)

        # Create evaluator of agent belief state
        self.evaluator = Evaluator()

        self.goal_achieved = False
        self.update()

        self.collision = False

        self.hide_magnitude = 0

        # Necessary to compute Success weighted by Path Length (SPL) for the Object Goal Navigation task
        # using the evaluation function "compute_single_spl" provided by ai2thor.util.metrics
        self.path = []
        if shortest_path is not None:
            self.evaluator.shortest_path = shortest_path

        # Prepare environment for open/close goal
        if Configuration.TASK == Configuration.TASK_OPEN:
            for obj in [o for o in self.event.metadata['objects'] if o['objectType'].lower() in Configuration.GOAL_OBJECTS]:
                action_result = self.controller.step(action='CloseObject', objectId=obj['objectId'], forceAction=True)
                if not action_result.metadata['lastActionSuccess']:
                    print('Error when closing all the objects instances of the goal type, check Agent.py constructor.')
                    exit()
        elif Configuration.TASK == Configuration.TASK_CLOSE:
            for obj in [o for o in self.event.metadata['objects'] if o['objectType'].lower() in Configuration.GOAL_OBJECTS]:
                action_result = self.controller.step(action='OpenObject', objectId=obj['objectId'], forceAction=True)
                if not action_result.metadata['lastActionSuccess']:
                    print('Error when opening all the objects instances of the goal type, check Agent.py constructor.')
                    exit()


    def run(self, n_iter=Configuration.MAX_ITER):

        start = datetime.datetime.now()

        # Iterate for a maximum number of steps
        for i in range(n_iter):

            if self.goal_achieved:
                break

            # Set current iteration number
            self.iter = i

            if (self.event_planner.event_plan is None or self.event_planner.subgoal is None or self.event_planner.subgoal.lower().startswith("pickup"))\
                    and -int(self.state.perceptions[7]) < 0:
                # Adjust agent camera inclination
                event_action = 'LookUp'
            elif (self.event_planner.event_plan is None or self.event_planner.subgoal is None or self.event_planner.subgoal.lower().startswith("pickup"))\
                    and -int(self.state.perceptions[7]) > 0:
                event_action = 'LookDown'
            else:
                # Get low level action to execute from current first plan action (if any) or exploration
                event_action = self.event_planner.plan(self.learner.abstract_model, self.pos)

            # Call stop action for object goal navigation task
            if self.iter == Configuration.MAX_ITER - 1 and (Configuration.TASK == Configuration.TASK_OGN_ROBOTHOR
                                                            or Configuration.TASK == Configuration.TASK_OGN_ITHOR):
                event_action = "Stop"

            # DEBUG
            Logger.write('{}:{}'.format(self.iter + 1, event_action))

            # Execute the chosen action
            self.event = self.step(event_action)
            # Necessary to sync unity window frame, otherwise it shows one step before (but executes in the simulator)
            self.controller.step("Pass")

            # Detect collision when moving forward (and eventually update occupancy map)
            if event_action == "MoveAhead" and not self.event.metadata["lastActionSuccess"]:
                Logger.write("Collision detected")
                self.update_collision_map(self.angle)
                self.event_planner.path_plan = None
                self.event_planner.event_plan = None
                self.collision = True

            # Detect collision when rotating (and eventually change rotation direction)
            if event_action.startswith("Rotate") and not self.event.metadata["lastActionSuccess"]:
                Logger.write("Collision detected")
                self.event_planner.path_plan = None
                self.event_planner.event_plan = None
                self.event_planner.rotation_collision = True

                # This is necessary for 'LOOK_AT' symbolic action
                if self.event_planner.rotate_dir == Configuration.ROTATE_RIGHT:
                    self.event_planner.rotate_dir = Configuration.ROTATE_LEFT
                else:
                    self.event_planner.rotate_dir = Configuration.ROTATE_RIGHT

            # Detect failure in following cases:
            # CASE 1: when picking or putting or opening/closing objects (and remove them from the knowledge base)
            # CASE 2: when inspecting an object, a failure occurs whether the agent cannot see the
            # inspected object from n states which are close to the inspected object (while the agent
            # is looking to the inspected object direction)
            if ((event_action.startswith("Pick") or event_action.startswith("Put")
                or event_action.startswith("Open") or event_action.startswith("Close")) \
                    and not self.event.metadata["lastActionSuccess"]) \
                    or (len(self.event_planner.useless_goal_cells) >= Configuration.MAX_USELESS_GOAL_CELLS):
                Logger.write("NOT successfully executed action: {}".format(self.event_planner.subgoal))

                # DEBUG
                # if len(event_action.split('|')) > 1:
                    # Logger.write("INFO: Failed {} action on {}. {}".format(event_action.split('|')[0],
                    #                                                  self.controller.step('GetObjectInFrame',
                    #                                                                       x=event_action.split('|')[1],
                    #                                                                       y=event_action.split('|')[2]).metadata['actionReturn'],
                    #                                                  "Failure message: {}".format(self.event.metadata['errorMessage'])))

                removed_obj_id = self.event_planner.subgoal.lower().split('(')[1].strip()[:-1].split(',')[-1]
                self.learner.remove_object(removed_obj_id, self.evaluator)

                # Reset event plan in event planner since the event planner subgoal is failed
                self.event_planner.subgoal = None
                self.event_planner.event_plan = None
                self.event_planner.path_plan = None
                self.event_planner.useless_goal_cells = []

                if 'is not a valid' in self.event.metadata['errorMessage'].lower():
                    Logger.write('WARNING: the action cannot be executed since the contained object is not a valid'
                                 ' object type for the container one')

                elif 'no valid positions' in self.event.metadata['errorMessage'].lower():
                    Logger.write('WARNING: the action cannot be executed since there are no valid positions '
                                 'to place the held object.')

            # Look if pddl action has been successfully executed
            elif self.event.metadata["lastActionSuccess"] and self.event_planner.event_plan is not None \
                    and len(self.event_planner.event_plan) == 0 and self.event_planner.subgoal is not None:

                # DEBUG
                Logger.write("Successfully executed action: {}".format(self.event_planner.subgoal))

                # Apply action effects regardless of last observation
                if Configuration.TRUST_PDDL:
                    self.apply_action_effects(self.event_planner.subgoal)
                else:
                    print('If the agent does not trust the PDDL action effects, '
                          'you have to manage the predicates: HAND_FREE and PICKED(OBJ) from sensory perceptions')
                    exit()

                # Reset useless_goal_cells used by 'INSPECT' pddl action
                self.event_planner.useless_goal_cells = []

                # Check if goal has been reached
                if self.event_planner.subgoal == 'STOP()' or self.event_planner.pddl_plan[0] == 'STOP()':
                    self.goal_achieved = True

            # Save agent view image
            if Configuration.PRINT_CAMERA_VIEW_IMAGES:
                Logger.save_img("view_{}.png".format(i), self.event.frame)

            # Save agent depth view image
            if Configuration.PRINT_CAMERA_DEPTH_VIEW_IMAGES:
                Logger.save_img("depth_view_{}.png".format(i), (self.event.depth_frame/np.max(self.event.depth_frame)*255).astype('uint8'))

            # Perceive the environment
            perceptions = self.perceive()

            # Check if the current state has already been visited
            all_states_perceptions = np.array([state.perceptions[8 + Configuration.FRAME_WIDTH*Configuration.FRAME_HEIGHT*3:]
                                               for state in self.learner.abstract_model.states])
            all_states_diff = np.sum(all_states_perceptions -
                                     perceptions[8 + Configuration.FRAME_WIDTH*Configuration.FRAME_HEIGHT*3:], axis=1)
            candidate_states = [i for i, state in enumerate(all_states_diff) if state == 0]

            assert len(candidate_states) <= 1, 'There are more future possible states already visited, check Agent.py'

            if len(candidate_states) == 0:
                new_state = self.create_state(perceptions)
                # Add state in abstract model
                self.learner.add_state(new_state)
            else:
                # DEBUG
                # print('Coming back to state:{}'.format(candidate_states[0]))

                new_state = self.learner.abstract_model.states[candidate_states[0]]

                # Get visible object relationships and update predicates
                rgb_img = perceptions[8:8 + Configuration.FRAME_WIDTH * Configuration.FRAME_HEIGHT * 3] \
                    .reshape((Configuration.FRAME_HEIGHT, Configuration.FRAME_WIDTH, 3)) \
                    .astype(np.uint8)
                self.update_predicates(new_state.visible_objects, rgb_img)

                # Update objects bbox in knowledge manager
                for obj_type, obj_instances in new_state.visible_objects.items():
                    for obj_inst in obj_instances:
                        obj = [obj for obj in self.learner.knowledge_manager.all_objects[obj_type]
                               if obj['id'] == obj_inst['id']][0]
                        obj['bb'] = obj_inst['bb']

            # If a PDDL action has been successfully executed, apply action effects
            if self.last_action_effects is not None:
                # Update pddl state
                pos_effect = [e for e in self.last_action_effects if not e.startswith("(not ")]
                neg_effect = [e.replace('(not ', '').strip()[:-1] for e in self.last_action_effects if
                              e.startswith("(not ")]
                [self.learner.knowledge_manager.add_predicate(e) for e in pos_effect]
                [self.learner.knowledge_manager.remove_predicate(e) for e in neg_effect]
                self.last_action_effects = None

                self.learner.knowledge_manager.update_pddl_state()

            # Add transition
            self.learner.add_transition(self.state, event_action, new_state)

            # Update current state
            self.state = new_state

            # Update agent position in agent state and path planner state
            self.update()

            # Update top view map <==> the agent is in a new state
            if len(candidate_states) == 0 or self.collision:
                self.learner.update_topview(os.path.join(Logger.LOG_DIR_PATH, "topview_{}.png".format(self.iter)),
                                            self.event.depth_frame, self.angle,
                                            int(self.event.metadata['agent']['cameraHorizon']),
                                            self.pos, collision=self.collision)
                self.collision = False

            # Update followed path, this is required to compute SPL when solving the Object Goal Navigation task.
            if len(self.path) == 0 or self.pos != self.path[-1]:
                self.path.append(copy.deepcopy(self.pos))

        # Evaluate metrics
        self.evaluator.evaluate_state(self)

        if Configuration.TASK == Configuration.TASK_ON:
            self.evaluator.eval_goal_achievement_on(self.state.visible_objects, self.event)
        elif Configuration.TASK == Configuration.TASK_OPEN:
            self.evaluator.eval_goal_achievement_open(self.state.visible_objects, self.event)
        elif Configuration.TASK == Configuration.TASK_CLOSE:
            self.evaluator.eval_goal_achievement_close(self.state.visible_objects, self.event)
        elif Configuration.TASK == Configuration.TASK_OGN_ROBOTHOR or Configuration.TASK == Configuration.TASK_OGN_ITHOR:
            self.evaluator.eval_goal_achievement_objectnav(self.state.visible_objects, self.event, self.path,
                                                           self.controller)

        if self.goal_achieved:
            Logger.write('Episode succeeds.')
        else:
            Logger.write('Episode fails.')

        # DEBUG
        end = datetime.datetime.now()
        Logger.write("Episode computational time: {} seconds".format((end-start).seconds))

        # Release resources
        self.controller.stop()
        plt.close(self.event_planner.path_planner.map_model.fig)


    def step(self, action):
        action_result = None

        if action.startswith("Rotate") or action.startswith("Look"):
            if len(action.split("|")) > 1:
                degrees = round(float(action.split("|")[1]), 1)
                action_result = self.controller.step(action=action.split("|")[0], degrees=degrees)
            else:
                action_result = self.controller.step(action=action)

        elif action.startswith("OpenObject") or action.startswith("CloseObject"):
            # If xy camera coordinates are used to perform the action, i.e., are in the action name
            if len(action.split("|")) > 2:
                x_pos = round(float(action.split("|")[1]), 2)
                y_pos = round(float(action.split("|")[2]), 2)

                action_result = self.controller.step(action=action.split("|")[0], x=x_pos, y=y_pos)

        elif action.startswith("PickupObject") or action.startswith("PutObject"):
            # Store held objects
            old_inventory = copy.deepcopy(self.event.metadata['inventoryObjects'])

            # If xy camera coordinates are used to perform the action, i.e., are in the action name
            if len(action.split("|")) > 2:
                x_pos = round(float(action.split("|")[1]), 2)
                y_pos = round(float(action.split("|")[2]), 2)

                # The held object is hidden, however the simulator persistently sees it, hence if the target point
                # overlaps with the held (and hidden) object, the pickup action fails
                # ==> move up/down the held object in order to avoid overlapping with target point of pickup action.
                if action.startswith("PutObject"):
                    if y_pos >= 0.5:
                        move_magnitude = 0.7
                        moved = False
                        while not moved and move_magnitude > 0:
                            moved = self.controller.step("MoveHandUp", moveMagnitude=move_magnitude).metadata['lastActionSuccess']
                            move_magnitude -= 0.1
                        action_result = self.controller.step(action=action.split("|")[0], x=x_pos, y=y_pos)

                        # Check whether the object is within the visibility distance but cannot be placed into
                        # the container one due to object categories constraints
                        if 'cannot be placed in' in action_result.metadata['errorMessage']:
                            action_result = self.controller.step(action=action.split("|")[0], x=x_pos, y=y_pos, forceAction=True)

                        if moved:
                            self.controller.step("MoveHandDown", moveMagnitude=move_magnitude)

                    else:
                        move_magnitude = 0.7
                        moved = False
                        while not moved and move_magnitude > 0:
                            moved = self.controller.step("MoveHandDown", moveMagnitude=move_magnitude).metadata['lastActionSuccess']
                            move_magnitude -= 0.1
                        action_result = self.controller.step(action=action.split("|")[0], x=x_pos, y=y_pos)

                        # Check whether the object is within the visibility distance but cannot be placed into
                        # the container one due to object categories constraints
                        if 'cannot be placed in' in action_result.metadata['errorMessage']:
                            action_result = self.controller.step(action=action.split("|")[0], x=x_pos, y=y_pos, forceAction=True)

                        if moved:
                            self.controller.step("MoveHandUp", moveMagnitude=move_magnitude)
                else:
                    action_result = self.controller.step(action=action.split("|")[0], x=x_pos, y=y_pos)

                # Hide picked up objects
                if Configuration.HIDE_PICKED_OBJECTS and action.startswith("PickupObject") \
                        and action_result.metadata['lastActionSuccess']:
                    for picked_obj in action_result.metadata['inventoryObjects']:
                        self.hide_magnitude = 0.7
                        debug = self.controller.step("MoveHandDown", moveMagnitude=self.hide_magnitude)
                        while not debug.metadata['lastActionSuccess'] and self.hide_magnitude > 0:
                            self.hide_magnitude -= 0.1
                            debug = self.controller.step("MoveHandDown", moveMagnitude=self.hide_magnitude)
                        self.controller.step('HideObject', objectId=picked_obj['objectId'])
                        action_result = self.controller.step('Pass')

                # Unhide put down objects
                elif Configuration.HIDE_PICKED_OBJECTS and action.startswith("PutObject") \
                        and action_result.metadata['lastActionSuccess']:
                    for released_obj in old_inventory:
                        self.controller.step('UnhideObject', objectId=released_obj['objectId'])
                    if self.hide_magnitude > 0:
                        debug = self.controller.step("MoveHandUp", moveMagnitude=self.hide_magnitude)
                        if not debug.metadata['lastActionSuccess']:
                            Logger.write('WARNING: MoveHandUp failed after putting object down. Check step() function'
                                         'in Agent.py.')
                    action_result = self.controller.step('Pass')

            # Other cases
            else:
                print('You should manage the case where a pickup/putdown action is performed without '
                      'passing input xy camera coordinates. Look at step() method in Agent.py .')
                exit()

        elif action == "Stop":
            action_result = self.step("Pass")
            self.goal_achieved = True

        else:
            # Execute "move" action in the environment
            action_result = self.controller.step(action=action)

        self.learner.knowledge_manager.update_all_obj_position(action, action_result, self.event_planner.subgoal)

        return action_result

    def perceive(self):

        # Get perceptions
        x_pos = self.event.metadata['agent']['position']['x']
        y_pos = self.event.metadata['agent']['position']['z']
        camera_z_pos = self.event.metadata['cameraPosition']['y']
        # camera_z_pos = self.event.metadata['agent']['position']['y']
        # hand_x_pos = self.event.metadata['hand']['position']['x']
        # hand_y_pos = self.event.metadata['hand']['position']['z']
        # hand_z_pos = self.event.metadata['hand']['position']['y']
        hand_x_pos = self.event.metadata['heldObjectPose']['position']['x']
        hand_y_pos = self.event.metadata['heldObjectPose']['position']['z']
        hand_z_pos = self.event.metadata['heldObjectPose']['position']['y']
        angle = (360 - self.event.metadata['agent']['rotation']['y'] + 90) % 360
        camera_angle = self.event.metadata['agent']['cameraHorizon']  # tilt angle of the camera
        rgb_img = self.event.frame
        depth_img = self.event.depth_frame
        perceptions = np.concatenate((np.array([x_pos, y_pos, camera_z_pos,
                                                hand_x_pos, hand_y_pos, hand_z_pos,
                                                angle, camera_angle]),
                                      rgb_img.flatten(),
                                      depth_img.flatten()), dtype=np.float32)
        return perceptions

    def create_state(self, perceptions):

        x_pos = perceptions[0]
        y_pos = perceptions[1]
        camera_z_pos = perceptions[2]
        angle = perceptions[6]

        rgb_img = perceptions[8:8 + Configuration.FRAME_WIDTH*Configuration.FRAME_HEIGHT*3]\
            .reshape((Configuration.FRAME_HEIGHT, Configuration.FRAME_WIDTH, 3))\
            .astype(np.uint8)

        depth_img = perceptions[8 + Configuration.FRAME_WIDTH*Configuration.FRAME_HEIGHT*3:]\
            .reshape((Configuration.FRAME_HEIGHT, Configuration.FRAME_WIDTH))\
            .astype(np.float32)

        agent_pos = {'x': x_pos, 'y': y_pos, 'z': camera_z_pos}
        visible_objects = self.learner.get_visible_objects(rgb_img, depth_img, agent_pos, angle, self.event)

        # Update overall knowledge about objects
        visible_objects = self.learner.knowledge_manager.update_objects(visible_objects, agent_pos)

        # Get visible object relationships and update predicates
        self.update_predicates(visible_objects, rgb_img)

        # If a PDDL action has been successfully executed, apply action effects
        if self.last_action_effects is not None:
            # Update pddl state
            pos_effect = [e for e in self.last_action_effects if not e.startswith("(not ")]
            neg_effect = [e.replace('(not ','').strip()[:-1] for e in self.last_action_effects if e.startswith("(not ")]
            [self.learner.knowledge_manager.add_predicate(e) for e in pos_effect]
            [self.learner.knowledge_manager.remove_predicate(e) for e in neg_effect]
            self.last_action_effects = None

            self.learner.knowledge_manager.update_pddl_state()

        # Create new state
        s_new = State(len(self.learner.abstract_model.states), perceptions, visible_objects)

        return s_new


    def update_predicates(self, visible_objects, rgb_img):
        visible_predicates = self.learner.scene_classifier.get_visible_predicates(visible_objects, rgb_img)

        # Update overall knowledge about predicates
        self.learner.knowledge_manager.update_all_predicates(visible_predicates, visible_objects, self.learner.abstract_model,
                                                             self.event_planner.path_planner.get_occupancy_grid())

        # Update pddl state
        self.learner.knowledge_manager.update_pddl_state()


    def update_collision_map(self, agent_theta):

        # Map agent position into grid
        start = [self.pos['x'] * 100, self.pos['y'] * 100]
        start_grid = (int(round((start[0] - self.learner.mapper.map_model.x_min) / self.learner.mapper.map_model.dx)),
                      int(round((start[1] - self.learner.mapper.map_model.y_min) / self.learner.mapper.map_model.dy)))

        collision_cell = None

        NOISE_THRSH = 30  # noise threshold between two subsequent cells
        if 0 - NOISE_THRSH <= agent_theta <= 0 + NOISE_THRSH or 360 - NOISE_THRSH < agent_theta <= 360 + NOISE_THRSH:
            collision_cell = [start_grid[1], start_grid[0] + 1]
        elif NOISE_THRSH < agent_theta <= 90 - NOISE_THRSH:
            collision_cell = [start_grid[1] + 1, start_grid[0] + 1]
        elif 90 - NOISE_THRSH < agent_theta <= 90 + NOISE_THRSH:
            collision_cell = [start_grid[1] + 1, start_grid[0]]
        elif 90 + NOISE_THRSH < agent_theta <= 180 - NOISE_THRSH:
            collision_cell = [start_grid[1] + 1, start_grid[0] - 1]
        elif 180 - NOISE_THRSH < agent_theta <= 180 + NOISE_THRSH:
            collision_cell = [start_grid[1], start_grid[0] - 1]
        elif 180 + NOISE_THRSH < agent_theta <= 270 - NOISE_THRSH:
            collision_cell = [start_grid[1] - 1, start_grid[0] - 1]
        elif 270 - NOISE_THRSH < agent_theta <= 270 + NOISE_THRSH:
            collision_cell = [start_grid[1] - 1, start_grid[0]]
        elif 270 + NOISE_THRSH < agent_theta <= 360 - NOISE_THRSH:
            collision_cell = [start_grid[1] - 1, start_grid[0] + 1]

        assert collision_cell is not None, "Cannot add null collision cell"

        self.learner.mapper.map_model.collision_cells.append(collision_cell)

        # Check if the goal position on the grid is the same as the collision cell just added. If this is the case,
        # change the goal position in the path planner.
        goal = [self.event_planner.path_planner.goal_position[0]*100,
                self.event_planner.path_planner.goal_position[1]*100]
        goal_grid = [int(round((goal[1]-self.learner.mapper.map_model.y_min)/self.learner.mapper.map_model.dy)),
                     int(round((goal[0]-self.learner.mapper.map_model.x_min)/self.learner.mapper.map_model.dx))]

        if goal_grid == collision_cell:
            print('changing goal position')

            self.goal_position = [(Configuration.MAP_X_MIN + (Configuration.MOVE_STEP * 100 * 2)) / 100,
                                  (Configuration.MAP_Y_MIN + (Configuration.MOVE_STEP * 100 * 2)) / 100]


    def update(self):
        """
        Update agent position in agent state and in path planner state, then update ground truth state required
        for evaluation.
        :return: None
        """

        # Update agent xyz position
        self.pos['x'], self.pos['y'], self.pos['z'] = self.state.perceptions[0], self.state.perceptions[1], self.state.perceptions[2]

        # Update agent xyz position
        self.hand_pos['x'], self.hand_pos['y'], self.hand_pos['z'] = self.state.perceptions[3], self.state.perceptions[4], self.state.perceptions[5]

        # Update agent y rotation
        self.angle = int(round(self.state.perceptions[6]))

        # Update path planner state
        self.event_planner.path_planner.agent_position = self.pos
        self.event_planner.path_planner.agent_angle = self.angle

        # Update event planner state
        self.event_planner.all_objects = self.learner.knowledge_manager.all_objects
        self.event_planner.visible_objects = self.state.visible_objects
        self.event_planner.perceptions = self.state.perceptions

        # Update ground truth state for evaluation
        self.evaluator.update_gt_state(self.state.visible_objects,
                                       self.event)


    def apply_action_effects(self, action_name):

        self.last_action_effects = []

        # Get operator name
        op_name = action_name.split("(")[0].strip().lower()

        # Get operator objects
        op_objs = {"?param_{}".format(i + 1): obj
                   for i, obj in enumerate(action_name.split("(")[1][:-1].strip().lower().split(","))}

        # Get operator effects
        op_effects = PddlParser.get_operator_effects(op_name)

        # Update predicates with positive effects
        for pred in [pred for pred in op_effects if not pred.startswith("(not ")]:
            # Replace predicate variables with pddl action input objects
            for k, v in op_objs.items():
                pred = pred.replace(k, v)
            # Get predicate name and grounded input objects
            pred_name = pred[1:-1].split()[0].strip()
            pred_objs = [obj.strip() for obj in pred[1:-1].split()[1:]]
            pred_renamed = "{}({})".format(pred_name, ",".join(pred_objs))
            if pred_renamed not in self.learner.knowledge_manager.all_predicates:
                # DEBUG
                print("Adding {}".format(pred_renamed))
                self.last_action_effects.append(pred_renamed)

        # Update predicates with negative effects
        for pred in [pred for pred in op_effects if pred.startswith("(not ")]:
            pred = pred.replace("(not ", "").strip()[:-1].strip()
            # Replace predicate variables with pddl action input objects
            for k, v in op_objs.items():
                pred = pred.replace(k, v)
            # Get predicate name and grounded input objects
            pred_name = pred[1:-1].split()[0].strip()
            pred_objs = [obj.strip() for obj in pred[1:-1].split()[1:]]
            pred_renamed = "{}({})".format(pred_name, ",".join(pred_objs))
            if pred_renamed in self.learner.knowledge_manager.all_predicates:
                # DEBUG
                print("Removing {}".format(pred_renamed))
                self.last_action_effects.append("(not {})".format(pred_renamed))



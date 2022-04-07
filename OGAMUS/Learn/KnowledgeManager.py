# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
from collections import defaultdict

import Configuration
from Utils import PddlParser
import numpy as np



class KnowledgeManager:

    def __init__(self):
        self.all_objects = defaultdict(list)
        self.all_predicates = []

        self.objects_counting = defaultdict(int)
        self.objects_avg_score = defaultdict(float)

    def update_objects(self, new_objects, agent_pos):
        position_threshold = 0.2  # if some x y z coordinate is above threshold, then the object instance is a new one

        # List of objects merged with the current knowledge ones, i.e., if a visible object is already existing in the
        # current knowledge, its ID is updated with the already existing one.
        merged_objects = {obj_type: [] for obj_type in new_objects.keys()}

        # Add new objects to current objects dictionary
        for obj_type in new_objects.keys():

            for new_obj_type_inst in new_objects[obj_type]:

                new_obj_x, new_obj_y, new_obj_z = new_obj_type_inst['map_x'], new_obj_type_inst['map_y'], new_obj_type_inst['map_z']

                new_obj_exists = False
                merged_object = None

                # Check if an object instance does not exist yet in the current objects dictionary
                for existing_obj in self.all_objects[obj_type]:
                    exist_obj_x, exist_obj_y, exist_obj_z = existing_obj['map_x'], existing_obj['map_y'], existing_obj['map_z']

                    if np.linalg.norm(np.array([exist_obj_x, exist_obj_y, exist_obj_z])
                                      - np.array([new_obj_x, new_obj_y, new_obj_z])) < position_threshold:

                        # Update existing object average score
                        self.objects_avg_score[existing_obj['id']] = new_obj_type_inst['score']/(self.objects_counting[existing_obj['id']] + 1) \
                                                + (self.objects_counting[existing_obj['id']]*self.objects_avg_score[existing_obj['id']])/(self.objects_counting[existing_obj['id']] + 1)

                        # Update objects counting
                        self.objects_counting[existing_obj['id']] += 1

                        merged_object = copy.deepcopy(new_obj_type_inst)
                        merged_object['id'] = existing_obj['id']

                        # Update existing object bounding box
                        existing_obj['bb'] = merged_object['bb']

                        # Update existing object position
                        existing_obj['map_x'] = merged_object['map_x']/(self.objects_counting[existing_obj['id']] + 1) \
                                                + (self.objects_counting[existing_obj['id']]*existing_obj['map_x'])/(self.objects_counting[existing_obj['id']] + 1)
                        existing_obj['map_y'] = merged_object['map_y']/(self.objects_counting[existing_obj['id']] + 1) \
                                                + (self.objects_counting[existing_obj['id']]*existing_obj['map_y'])/(self.objects_counting[existing_obj['id']] + 1)
                        existing_obj['map_z'] = merged_object['map_z']/(self.objects_counting[existing_obj['id']] + 1) \
                                                + (self.objects_counting[existing_obj['id']]*existing_obj['map_z'])/(self.objects_counting[existing_obj['id']] + 1)

                        new_obj_exists = True

                        break

                # If new object instance does not exist yet in the current objects dictionary
                if not new_obj_exists:
                    # Update new object id
                    new_obj_id = "{}_{}".format(obj_type, len(self.all_objects[obj_type]))

                    # object instance is a new one
                    new_obj_type_inst['id'] = new_obj_id

                    # Update objects counting
                    self.objects_counting[new_obj_id] += 1
                    self.objects_avg_score[new_obj_id] = new_obj_type_inst['score']

                    self.all_objects[obj_type].append(new_obj_type_inst)
                    merged_object = copy.deepcopy(new_obj_type_inst)

                # Add a new object to merged ones
                if merged_object is not None:
                    existing_obj_ids = [obj['id'] for obj in merged_objects[obj_type]]
                    if merged_object['id'] not in existing_obj_ids:
                        merged_objects[obj_type].append(merged_object)

        self.update_all_obj_distances(agent_pos)

        return merged_objects


    def update_objects_ground_truth(self, new_objects):
        position_threshold = 0.2  # if some x y z coordinate is above threshold, then the object instance is a new one

        objects_id_mapping = dict()

        # Add new objects to current objects dictionary
        for obj_type in new_objects.keys():
            # The object type is a new one
            if len(self.all_objects[obj_type]) == 0:
                self.all_objects[obj_type].extend(new_objects[obj_type])

                # Update objects id mapping
                for obj in new_objects[obj_type]:
                    objects_id_mapping[obj['id']] = {'id': obj['id'], 'name': obj['name']}

            # There are already some object instances of object type
            else:
                for new_obj_type_inst in new_objects[obj_type]:

                    new_obj_x, new_obj_y, new_obj_z = new_obj_type_inst['map_x'], new_obj_type_inst['map_y'], new_obj_type_inst['map_z']

                    new_obj_exists = False

                    # Check if an object instance does not exist yet in the current objects dictionary
                    for existing_obj in self.all_objects[obj_type]:
                        exist_obj_x, exist_obj_y, exist_obj_z = existing_obj['map_x'], existing_obj['map_y'], existing_obj['map_z']

                        if (new_obj_x - exist_obj_x) < position_threshold \
                            and (new_obj_y - exist_obj_y) < position_threshold \
                            and (new_obj_z - exist_obj_z) < position_threshold:
                            # Change (not) new object instance id to already existing one
                            objects_id_mapping[new_obj_type_inst['id']] = {'id': existing_obj['id'],
                                                                           'name':existing_obj['name']}
                            new_obj_exists = True
                            break

                    # If new object instance does not exist yet in the current objects dictionary
                    if not new_obj_exists:
                        # Update new object id
                        new_obj_id = "{}_{}".format(obj_type, len(self.all_objects[obj_type]))
                        objects_id_mapping[new_obj_type_inst['id']] = {'id':new_obj_id, 'name':new_obj_type_inst['name']}
                        # object instance is a new one
                        new_obj_type_inst['id'] = new_obj_id
                        self.all_objects[obj_type].append(new_obj_type_inst)

        # Return visible objects with updated id
        return objects_id_mapping


    def add_predicate(self, new_predicate):
        self.all_predicates.append(new_predicate)


    def remove_predicate(self, removed_predicate):
        self.all_predicates.remove(removed_predicate)


    def update_all_predicates(self, new_predicates, visible_objects, fsm_model, occupancy_grid):

        self.all_predicates = list(set(self.all_predicates + new_predicates))

        # Check "hand_free" predicate
        hand_free = len([o for o in self.all_predicates if o.strip().startswith("holding")]) == 0
        if hand_free and "hand_free()" not in self.all_predicates:
            self.all_predicates.append("hand_free()")

        # Update "viewing(object)" predicate for all visible objects
        self.all_predicates = [pred for pred in self.all_predicates if not pred.startswith("viewing")]
        [self.all_predicates.append("viewing({})".format(obj['id']))
        for obj_type in list(visible_objects.keys())
        for obj in visible_objects[obj_type]]

        removed_close_to = []
        for pred in [p for p in self.all_predicates if p not in new_predicates]:
            if pred.startswith('close_to'):
                obj_id = pred.split('(')[1].lower().strip()[:-1]
                obj_type = obj_id.split('_')[0]

                obj = [obj for obj in self.all_objects[obj_type] if obj['id'] == obj_id][0]

                if obj['distance'] > Configuration.CLOSE_TO_OBJ_DISTANCE:
                    removed_close_to.append(pred)

        self.all_predicates = [pred for pred in self.all_predicates if pred not in removed_close_to]

        [self.all_predicates.append("close_to({})".format(obj['id']))
         for obj_type in list(self.all_objects.keys())
         for obj in self.all_objects[obj_type]
         if obj['distance'] <= Configuration.CLOSE_TO_OBJ_DISTANCE]

        # Update "inspected(object)" predicate for all objects
        self.all_predicates = [pred for pred in self.all_predicates if not pred.startswith("inspected")]
        [self.all_predicates.append("inspected({})".format(obj['id']))
         for obj_type in list(self.all_objects.keys())
         for obj in self.all_objects[obj_type]
         if len([s for s in fsm_model.states
                 if obj_type in s.visible_objects.keys()
                 and obj['id'] in [obj['id'] for obj in s.visible_objects[obj_type]
                                   if obj['distance'] <= Configuration.CLOSE_TO_OBJ_DISTANCE
                                   and occupancy_grid[occupancy_grid.shape[0] - int(round((s.perceptions[1] * 100 - Configuration.MAP_Y_MIN) / Configuration.MAP_GRID_DY))]
                                   [int(round((s.perceptions[0] * 100 - Configuration.MAP_X_MIN) / Configuration.MAP_GRID_DX))] != 0]]) > 0]
        [self.all_predicates.append("inspected({})".format(obj['id']))
         for obj_type in list(self.all_objects.keys())
         for obj in self.all_objects[obj_type]
         if "close_to({})".format(obj['id']) in self.all_predicates and "viewing({})".format(obj['id']) in self.all_predicates]


    def update_pddl_state(self):
        PddlParser.update_pddl_state(self.all_objects, self.all_predicates,
                                     self.objects_counting, self.objects_avg_score)



    def update_obj_position(self, obj_id, pos):
        obj_type = obj_id.split("_")[0]
        obj = [obj for obj in self.all_objects[obj_type] if obj['id'] == obj_id][0]

        # Update object coordinates
        obj['map_x'] = pos['x']
        obj['map_y'] = pos['y']
        # obj['map_z'] = pos['z']


    def update_all_obj_position(self, micro_action_name, micro_action_result, macro_action_name):

        # If action has failed then no updates are done
        if not micro_action_result.metadata['lastActionSuccess']:
            return

        # Get updated (after executing action) agent hand position
        # hand_pos = {'x':micro_action_result.metadata['hand']['position']['x'],
        #             'y':micro_action_result.metadata['hand']['position']['z'],
        #             'z':micro_action_result.metadata['hand']['position']['y']}
        hand_pos = {'x': micro_action_result.metadata['heldObjectPose']['position']['x'],
                    'y': micro_action_result.metadata['heldObjectPose']['position']['z'],
                    'z': micro_action_result.metadata['heldObjectPose']['position']['y']}


        # Update held object coordinates
        if micro_action_name.startswith("Move") or micro_action_name.startswith("Rotate") \
                or micro_action_name.startswith("Home") or micro_action_name.startswith("Look"):
            held_obj_id = [pred.split("(")[1][:-1].strip() for pred in self.all_predicates if pred.startswith("holding(")]
            if len(held_obj_id) > 0:
                held_obj_id = held_obj_id[0]
                self.update_obj_position(held_obj_id, hand_pos)

                # Update coordinates of all objects contained into held one
                contained_objs = [pred.split("(")[1][:-1].strip().split(",")[0] for pred in self.all_predicates
                                  if pred.startswith("on(") and pred.split("(")[1][:-1].strip().split(",")[1] == held_obj_id]
                for contained_obj_id in contained_objs:
                    self.update_obj_position(contained_obj_id, hand_pos)

        # Update picked object coordinates
        elif micro_action_name.startswith("Pickup"):

            obj_id = macro_action_name.split("(")[1][:-1].lower().strip()
            self.update_obj_position(obj_id, hand_pos)

            # Update coordinates of all objects contained into picked one
            contained_objs = [pred.split("(")[1][:-1].strip().split(",")[0] for pred in self.all_predicates
                              if pred.startswith("on(") and pred.split("(")[1][:-1].strip().split(",")[1] == obj_id]
            for contained_obj_id in contained_objs:
                self.update_obj_position(contained_obj_id, hand_pos)

        # Update put down object coordinates
        elif micro_action_name.startswith("PutObject"):

            contained_obj_id = macro_action_name.split("(")[1][:-1].lower().strip().split(",")[0].strip()
            container_obj_id = macro_action_name.split("(")[1][:-1].lower().strip().split(",")[1].strip()
            container_obj_type = container_obj_id.split("_")[0]
            container_obj = [obj for obj in self.all_objects[container_obj_type] if obj['id'] == container_obj_id][0]
            # container_pos = {'x': container_obj['map_x'], "y": container_obj['map_y'], "z":container_obj['map_z']}
            container_pos = {'x': container_obj['map_x'], "y": container_obj['map_y']}
            self.update_obj_position(contained_obj_id, container_pos)

            # Update coordinates of all objects contained into picked one
            contained_objs = [pred.split("(")[1][:-1].strip().split(",")[0] for pred in self.all_predicates
                              if pred.startswith("on(") and pred.split("(")[1][:-1].strip().split(",")[1] == contained_obj_id]
            for contained_obj_id in contained_objs:
                self.update_obj_position(contained_obj_id, container_pos)

    def update_all_obj_distances(self, agent_pos):
        agent_pos = [agent_pos['x'], agent_pos['y'], Configuration.CAMERA_HEIGHT]

        for obj_type, obj_instances in self.all_objects.items():
            for obj in obj_instances:
                obj_pos = [obj['map_x'], obj['map_y'], obj['map_z']]
                obj['distance'] = np.linalg.norm(np.array(agent_pos) - np.array(obj_pos))

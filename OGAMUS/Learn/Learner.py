# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
from collections import defaultdict

import Configuration
from OGAMUS.Learn.EnvironmentModels.AbstractModel import AbstractModel
from OGAMUS.Learn.KnowledgeManager import KnowledgeManager
from OGAMUS.Learn.Mapper import Mapper
from OGAMUS.Learn.ObjectDetector import ObjectDetector
from OGAMUS.Learn.ObjectDetector_robothor_ogn import ObjectDetector_robothor_ogn
from OGAMUS.Learn.SceneClassifier import SceneClassifier
import numpy as np

from Utils.depth_util import get_xyz_point_from_depth


class Learner:

    def __init__(self):

        # Abstract model (Finite state machine)
        self.abstract_model = AbstractModel()

        # Depth mapper
        self.mapper = Mapper()

        # Object detector
        self.object_detector = ObjectDetector()
        if Configuration.TASK == Configuration.TASK_OGN_ROBOTHOR:
            self.object_detector = ObjectDetector_robothor_ogn()

        # Scene classifier
        self.scene_classifier = SceneClassifier()

        # Knowledge manager
        self.knowledge_manager = KnowledgeManager()


    def add_state(self, state_new):
        self.abstract_model.add_state(state_new)


    def add_transition(self, state_src, action, state_dest):
        self.abstract_model.add_transition(state_src, action, state_dest)


    def update_topview(self, file_name, depth_matrix, angle, cam_angle, pos, collision=False):
        self.mapper.update_topview(depth_matrix, file_name, angle, -cam_angle, pos, collision)  # depth_matrix in meters


    def get_visible_objects(self, rgb_img, depth_img, agent_pos, agent_angle, event):

        if not Configuration.GROUND_TRUTH_OBJS:
            pred_objects = self.object_detector.get_visible_objects(rgb_img)
        else:
            pred_objects = self.object_detector.get_visible_objects_ground_truth(event)

        pred_objects['labels'] = [obj_type.lower() for obj_type in pred_objects['labels']]

        visible_objects = defaultdict(list)

        for obj_type, obj_bb, obj_score in zip(pred_objects['labels'], pred_objects['boxes'], pred_objects['scores']):
            # Get object centroid from bbox = [x0, y0, x1, y1]. Notice that y0 and y1 are from top to bottom
            obj_centroid = [int(round((obj_bb[2] + obj_bb[0]) / 2)),  # columns (x-axis)
                            int(round((obj_bb[3] + obj_bb[1]) / 2))]  # rows (y-axis)

            # Get object distance by averaging object reduced bbox depth values
            obj_bb_size = [obj_bb[2] - obj_bb[0], obj_bb[3] - obj_bb[1]]  # [height, width]

            # Filter non object bbox values
            depth_matrix = copy.deepcopy(depth_img)

            # Remove first rows
            min_row = max(0, int(round(obj_centroid[1]) - (obj_bb_size[1] * 0.25)) - 1)
            depth_matrix[:min_row, :] = np.nan
            # Remove first cols
            min_col = max(0, int(round(obj_centroid[0]) - (obj_bb_size[0] * 0.25)) - 1)
            depth_matrix[:, :min_col] = np.nan
            # Remove last rows
            max_row = max(0, int(round(obj_centroid[1]) + (obj_bb_size[1] * 0.25)) + 1)
            depth_matrix[max_row:, :] = np.nan
            # Remove last cols
            max_col = max(0, int(round(obj_centroid[0]) + (obj_bb_size[0] * 0.25)) + 1)
            depth_matrix[:, max_col:] = np.nan

            cam_angle = int(event.metadata['agent']['cameraHorizon'])
            x_obj, y_obj, z_obj = get_xyz_point_from_depth(depth_matrix, agent_angle, -cam_angle, agent_pos)

            obj_distance = np.linalg.norm(np.array([agent_pos['x'], agent_pos['y'], Configuration.CAMERA_HEIGHT])
                                          - np.array([x_obj, y_obj, z_obj]))

            visible_objects[obj_type].append({'id': '{}_{}'.format(obj_type, len(visible_objects[obj_type])),
                                              'map_x': x_obj,
                                              'map_y': y_obj,
                                              'map_z': z_obj,
                                              'distance': obj_distance,
                                              'bb': {'center': obj_centroid,
                                                     'corner_points': list(obj_bb),
                                                     'size': [obj_bb[2] - obj_bb[0], obj_bb[3] - obj_bb[1]]},
                                              'score': obj_score
                                              })
        return visible_objects


    def remove_object(self, removed_obj_id, evaluator):

        # Remove object from all objects list
        removed_obj_type = removed_obj_id.split('_')[0]
        self.knowledge_manager.all_objects[removed_obj_type] = [obj for obj in self.knowledge_manager.all_objects[removed_obj_type]
                                                                if not obj['id'] == removed_obj_id]

        # Remove all predicates involving the removed object
        self.knowledge_manager.all_predicates = [el for el in self.knowledge_manager.all_predicates
                                                 if removed_obj_id not in el.split('(')[1][:-1].strip().split(',')]

        # Rename remaining objects in increasing order
        renamed_objects = dict()
        for i in range(len(self.knowledge_manager.all_objects[removed_obj_type])):
            renamed_objects[self.knowledge_manager.all_objects[removed_obj_type][i]['id']] = '{}_{}'.format(removed_obj_type, i)
            self.knowledge_manager.all_objects[removed_obj_type][i]['id'] = '{}_{}'.format(removed_obj_type, i)

        # Update predicates with renamed objects
        for i in range(len(self.knowledge_manager.all_predicates)):
            for old_obj_name, new_obj_name in renamed_objects.items():
                self.knowledge_manager.all_predicates[i] = self.knowledge_manager.all_predicates[i].replace(old_obj_name + ',', new_obj_name + ',')
                self.knowledge_manager.all_predicates[i] = self.knowledge_manager.all_predicates[i].replace(old_obj_name + ')', new_obj_name + ')')

        # Update objects counting with renamed objects
        old_objs_counting = copy.deepcopy(self.knowledge_manager.objects_counting)
        self.knowledge_manager.objects_counting = defaultdict(int)
        old_objs_counting = {k: v for k, v in old_objs_counting.items() if k != removed_obj_id}
        for k, v in old_objs_counting.items():
            if k not in renamed_objects.keys():
                self.knowledge_manager.objects_counting[k] = v
            else:
                self.knowledge_manager.objects_counting[renamed_objects[k]] = old_objs_counting[k]

        # Update objects scores with renamed objects
        old_objs_scores = copy.deepcopy(self.knowledge_manager.objects_avg_score)
        self.knowledge_manager.objects_avg_score = defaultdict(float)
        old_objs_scores = {k: v for k, v in old_objs_scores.items() if k != removed_obj_id}
        for k, v in old_objs_scores.items():
            if k not in renamed_objects.keys():
                self.knowledge_manager.objects_avg_score[k] = v
            else:
                self.knowledge_manager.objects_avg_score[renamed_objects[k]] = old_objs_scores[k]

        # Remove the objects from all abstract model states and update remaining objects with new names
        for state in self.abstract_model.states:
            if removed_obj_type in state.visible_objects.keys():
                state.visible_objects[removed_obj_type] = [obj for obj in state.visible_objects[removed_obj_type]
                                                           if obj['id'] != removed_obj_id]

                if len(state.visible_objects[removed_obj_type]) > 0:
                    for old_obj_name, new_obj_name in renamed_objects.items():
                        if old_obj_name in [obj['id'] for obj in state.visible_objects[removed_obj_type]]:
                            old_obj_renamed = copy.deepcopy([obj for obj in state.visible_objects[removed_obj_type]
                                       if obj['id'] == old_obj_name][0])
                            old_obj_renamed['id'] = new_obj_name
                            state.visible_objects[removed_obj_type] = [obj for obj in state.visible_objects[removed_obj_type]
                                                                   if obj['id'] != old_obj_name]
                            state.visible_objects[removed_obj_type].append(old_obj_renamed)

        # Update objects id mapping (to ground truth ones) for evaluation purposes
        evaluator.objs_id_mapping = {k: v for k, v in evaluator.objs_id_mapping.items() if k != removed_obj_id}
        evaluator.objs_id_mapping = {(k if k not in renamed_objects else renamed_objects[k]): v
                                     for k, v in evaluator.objs_id_mapping.items()}

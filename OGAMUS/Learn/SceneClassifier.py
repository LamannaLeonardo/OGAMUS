# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import pickle
from collections import defaultdict

import numpy as np

import Configuration
from OGAMUS.Learn.PredicateClassifiers.OnPredicateClassifier import OnPredicateClassifier
from OGAMUS.Learn.PredicateClassifiers.OpenPredicateClassifier import OpenPredicateClassifier
from Utils.util import get1_hot_vector


class SceneClassifier:


    def __init__(self):
        self.scene_objects = None

        # Set open predicate classifier
        self.open_classifier = OpenPredicateClassifier(input_model_path=Configuration.OPEN_CLASSIFIER_PATH)
        self.on_classifier = OnPredicateClassifier(input_model_path=Configuration.ON_CLASSIFIER_PATH)

        # Set object classes
        self.obj_classes = [obj_class.lower() for obj_class in pickle.load(open(Configuration.OBJ_CLASSES_PATH, "rb"))]


    def get_visible_predicates(self, visible_objects, rgb_img):

        # visible_objects_id = defaultdict(list)
        visible_predicates = defaultdict(dict)

        for obj_type in visible_objects:
            for obj in visible_objects[obj_type]:
                obj_name = obj['id']
                obj_type = obj_name.split("_")[0]
                obj_visible = True
                obj_close = bool(obj['distance'] <= Configuration.CLOSE_TO_OBJ_DISTANCE)
                # The difference among 'inspected' and 'close' is that 'inspected' is persistent
                obj_inspected = bool(obj['distance'] <= Configuration.CLOSE_TO_OBJ_DISTANCE)

                # Crop image to object bbox
                obj_bbox = [int(coord) for coord in obj['bb']['corner_points']]
                obj_img_rgb = rgb_img[
                              max(0, obj_bbox[1] - 3): min(rgb_img.shape[0], obj_bbox[3] + 4),
                              max(0, obj_bbox[0] - 3): min(rgb_img.shape[1], obj_bbox[2] + 4),
                              :
                              ]

                obj_openable = obj_type in Configuration.OPENABLE_OBJS
                obj_pickable = obj_type in Configuration.PICKUPABLE_OBJS
                obj_receptacle = obj_type in Configuration.RECEPTACLE_OBJS

                # Predict whether an object is open by means of a neural network classifier
                if obj_openable:
                    obj_open = bool(self.open_classifier.predict(obj_img_rgb))
                else:
                    obj_open = False

                containers = []  # e.g. containers = [obj1_id, obj2_id]
                containers_scores = []
                # Predict whether an object is contained into another one by means of a neural network classifier
                for container_obj in [container_obj for obj_type in visible_objects
                                      for container_obj in visible_objects[obj_type]
                                      if container_obj['id'].split("_")[0] in Configuration.RECEPTACLE_OBJS
                                         and container_obj['id'] != obj_name]:
                    contained_type = obj_type
                    container_type = container_obj['id'].split("_")[0]
                    contained_class_index = self.obj_classes.index(contained_type)
                    container_class_index = self.obj_classes.index(container_type)

                    contained_bbox = [round(el, 2) for el in obj['bb']['corner_points']]
                    container_bbox = [round(el, 2) for el in container_obj['bb']['corner_points']]

                    example = []
                    example.extend(list(get1_hot_vector(contained_class_index, len(self.obj_classes))))
                    example.extend(list(get1_hot_vector(container_class_index, len(self.obj_classes))))
                    example.extend(contained_bbox)
                    example.extend(container_bbox)

                    on, score = self.on_classifier.predict(example)
                    on, score = bool(on), float(score)

                    if on and contained_type.lower() not in Configuration.NOT_CONTAINED_OBJS:
                        containers.append(container_obj['id'])
                        containers_scores.append(score)

                # Update object visible predicates
                if Configuration.TASK == Configuration.TASK_ON and obj_name.split('_')[0] == Configuration.GOAL_OBJECTS[1]:
                    obj_pickable = False
                elif Configuration.TASK == Configuration.TASK_ON and obj_name.split('_')[0] == Configuration.GOAL_OBJECTS[0]:
                    obj_receptacle = False

                visible_predicates[obj_name] = {'open': obj_open,
                                                'discovered': obj_visible,
                                                'close_to': obj_close,
                                                'inspected': obj_inspected,
                                                'on': containers,
                                                'openable': obj_openable,
                                                'pickupable': obj_pickable,
                                                'receptacle': obj_receptacle}

        # Generate visible predicates list
        visible_predicates = dict(visible_predicates)
        # Get unary predicates
        visible_predicates_list = ["{}({})".format(k2, k) for k, v in visible_predicates.items()
                                   for k2, v2 in v.items() if type(v2) == type(True) and v2]

        # Get 'ON' binary predicates
        visible_predicates_list.extend(["{}({},{})".format(k2, k, v3) for k, v in visible_predicates.items()
                                        for k2, v2 in v.items() if type(v2) != type(True) for v3 in v2])

        return visible_predicates_list


    def get_visible_predicates_ground_truth(self, metadata, objects_id_mapping):

        visible_objects_id = defaultdict(list)
        visible_predicates = defaultdict(dict)

        for obj in metadata['objects']:
            if obj['visible']:
                obj_name = "{}_{}".format(obj['objectType'].lower(), len(visible_objects_id[obj['objectType'].lower()]))
                obj_picked = obj['isPickedUp']
                obj_open = obj['isOpen']
                obj_openable = obj['openable']
                obj_pickable = obj['pickupable']
                obj_receptacle = obj['receptacle']
                obj_visible = obj['visible']
                obj_close = obj['distance'] < 1.5

                # Look if object is contained in another one
                containers = []
                if obj['parentReceptacles'] is not None:
                    for container_name in obj['parentReceptacles']:
                        container_name = container_name
                        [containers.append(k) for k, v in objects_id_mapping.items() if v['name'] == container_name]

                visible_objects_id[obj['objectType'].lower()].append({"id":obj_name})
                visible_predicates[obj_name] = {'holding':obj_picked,
                                                'open':obj_open,
                                                'discovered':obj_visible,
                                                'close_to':obj_close,
                                                'on':containers,
                                                'openable':obj_openable,
                                                'pickupable':obj_pickable,
                                                'receptacle':obj_receptacle}

        # Update visible predicates (object) id
        for obj_id in objects_id_mapping.keys():
            visible_predicates[objects_id_mapping[obj_id]['id']] = visible_predicates.pop(obj_id)

        # Generate visible predicates list
        visible_predicates = dict(visible_predicates)
        # Get unary predicates
        visible_predicates_list = ["{}({})".format(k2,k) for k,v in visible_predicates.items()
                                   for k2, v2 in v.items() if type(v2) == type(True) and v2]
        # Get binary predicates
        visible_predicates_list.extend(["{}({},{})".format(k2,k,v3) for k,v in visible_predicates.items()
                                        for k2, v2 in v.items() if type(v2) != type(True) for v3 in v2])

        return visible_predicates_list


    def get_iou(self, bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import pickle
import re
from collections import defaultdict
import numpy as np
import Configuration

from Utils import Logger

from ai2thor.util.metrics import (
    compute_single_spl,
    get_shortest_path_to_object_type,
    path_distance
)


all_success = []
all_spl = []
all_dts = []

class Evaluator:

    def __init__(self):
        self.objs_id_mapping = dict()
        self.gt_objs_seen = []
        self.gt_predicates = []

        # Set object classes
        self.obj_classes = [obj_class.lower() for obj_class in pickle.load(open(Configuration.OBJ_CLASSES_PATH, "rb"))]

        # Used for Object Goal Navigation task when computing SPL.
        self.shortest_path = None


    def update_visible_objects_ground_truth(self, event):

        [self.gt_objs_seen.append(obj['objectId'])
         for obj in event.metadata['objects']
         if obj['objectId'] in event.instance_detections2D
         and obj['objectId'] not in self.gt_objs_seen
         and obj['objectType'].lower() not in Configuration.IRRELEVANT_CLASSES]


    def update_objs_id_mapping(self, visible_objects, event):

        # Get list of belief state objects
        belief_objects = []
        [belief_objects.extend(v) for k,v in visible_objects.items()]

        for belief_obj in belief_objects:

            # Check if the predicted object id has already been mapped into a ground truth object id
            if belief_obj['id'] not in self.objs_id_mapping.keys():
                obj_type = belief_obj['id'].split("_")[0].lower()
                x_min = int(round(belief_obj['bb']['corner_points'][0]))
                y_min = int(round(belief_obj['bb']['corner_points'][1]))
                x_max = int(round(belief_obj['bb']['corner_points'][2]))
                y_max = int(round(belief_obj['bb']['corner_points'][3]))
                belief_bbox = {'x1': x_min, 'y1': y_min, 'x2': x_max, 'y2': y_max}

                visible_gt = []
                [visible_gt.append(obj['objectId'])
                 for obj in event.metadata['objects']
                 if obj['objectId'] in event.instance_detections2D
                 and obj['objectId'] not in visible_gt
                 and obj['objectType'].lower() not in Configuration.IRRELEVANT_CLASSES
                 and obj['objectType'].lower()==obj_type]

                for obj in visible_gt:
                    bbox_pts = event.instance_detections2D[obj]
                    gt_bbox = {'x1': bbox_pts[0], 'y1': bbox_pts[1], 'x2': bbox_pts[2], 'y2': bbox_pts[3]}

                    if gt_bbox['x2'] > gt_bbox['x1'] and gt_bbox['y2'] > gt_bbox['y1'] \
                            and self.get_iou(belief_bbox, gt_bbox) > Configuration.OBJ_IOU_MATCH_THRSH:
                        self.objs_id_mapping[belief_obj['id']] = obj
                        break

                if belief_obj['id'] not in self.objs_id_mapping and Configuration.GROUND_TRUTH_OBJS:
                    Logger.write('Warning: cannot map detected object {} into a ground truth one, even if '
                                 'using ground truth detection. Check update_objs_id_mapping in Evaluator.py')


    def evaluate_state(self, agent):

        metrics = {
            "objects":{
                "belief": None,
                "gt": self.gt_objs_seen,
                "belief2gt": self.objs_id_mapping,
                "belief_counting": None,
                "belief_avg_scores": None},
            "predicates":{
                "belief": None,
                "gt": self.gt_predicates,
                "tp": None,
                "fp": None,
                "fn": None},
            "global": {
                "belief": None,
                "gt": self.gt_predicates,
                "tp": None,
                "fp": None,
                "fn": None},
        }

        # Update and store ground-truth belief objects
        belief_objects = []
        [belief_objects.extend(v) for k,v in agent.learner.knowledge_manager.all_objects.items()]
        belief_objects = [obj['id'] for obj in belief_objects]
        mapped_belief_objects = [self.objs_id_mapping[obj_id] if obj_id in self.objs_id_mapping else "unknown_" + obj_id
                                 for obj_id in belief_objects]
        metrics['objects']['belief'] = list(set(mapped_belief_objects))

        # Map and store belief predicates
        belief_predicates = agent.learner.knowledge_manager.all_predicates
        mapped_belief_predicates = []
        for pred in belief_predicates:
            for belief_obj_id in belief_objects:
                if belief_obj_id in self.objs_id_mapping:
                    pred = pred.replace(belief_obj_id + ')', self.objs_id_mapping[belief_obj_id] + ')')
                    pred = pred.replace(belief_obj_id + ',', self.objs_id_mapping[belief_obj_id] + ',')
                else:
                    pred = pred.replace(belief_obj_id, "unknown_" + belief_obj_id)

            mapped_belief_predicates.append(pred)

        mapped_belief_predicates_global = [pred for pred in mapped_belief_predicates
                                           if not pred.split("(")[0] in ["pickupable", "receptacle", "openable"]]
        mapped_belief_predicates = [pred for pred in mapped_belief_predicates
                                    if not pred.split("(")[0] in ["pickupable", "receptacle", "openable"]
                                    and not "unknown" in pred]

        # Compute predicate metrics
        metrics['predicates']['belief'] = list(set(mapped_belief_predicates))
        metrics['predicates']['tp'] = len([pred for pred in mapped_belief_predicates
                                           if pred in metrics['predicates']['gt']])
        metrics['predicates']['fp'] = len([pred for pred in mapped_belief_predicates
                                           if pred not in metrics['predicates']['gt']])
        metrics['predicates']['fn'] = len([pred for pred in metrics['predicates']['gt']
                                           if pred not in mapped_belief_predicates])
        metrics['predicates']['precision'] = metrics['predicates']['tp'] / \
                                             (metrics['predicates']['tp'] + metrics['predicates']['fp'])
        metrics['predicates']['recall'] = metrics['predicates']['tp'] / \
                                             (metrics['predicates']['tp'] + metrics['predicates']['fn'])

        # Compute object metrics
        metrics['objects']['tp'] = len([obj for obj in metrics['objects']['belief']
                                           if obj in metrics['objects']['gt']])
        metrics['objects']['fp'] = len([obj for obj in metrics['objects']['belief']
                                           if obj not in metrics['objects']['gt']])
        metrics['objects']['fn'] = len([obj for obj in metrics['objects']['gt']
                                           if obj not in metrics['objects']['belief']])
        metrics['objects']['precision'] = metrics['objects']['tp'] / \
                                             (metrics['objects']['tp'] + metrics['objects']['fp'])
        metrics['objects']['recall'] = metrics['objects']['tp'] / \
                                             (metrics['objects']['tp'] + metrics['objects']['fn'])
        metrics['objects']['belief_counting'] = agent.learner.knowledge_manager.objects_counting
        metrics['objects']['belief_avg_scores'] = {k:str(v) for k, v in agent.learner.knowledge_manager.objects_avg_score.items()}

        # Compute predicate global metrics, i.e. predicates on all objects (even not existing ones in the simulator)
        metrics['global']['belief'] = list(set(mapped_belief_predicates_global))
        metrics['global']['tp'] = len([pred for pred in mapped_belief_predicates_global
                                           if pred in metrics['predicates']['gt']])
        metrics['global']['fp'] = len([pred for pred in mapped_belief_predicates_global
                                           if pred not in metrics['predicates']['gt']])
        metrics['global']['fn'] = len([pred for pred in metrics['predicates']['gt']
                                           if pred not in mapped_belief_predicates_global])
        metrics['global']['precision'] = metrics['global']['tp'] / \
                                             (metrics['global']['tp'] + metrics['global']['fp'])
        metrics['global']['recall'] = metrics['global']['tp'] / \
                                             (metrics['global']['tp'] + metrics['global']['fn'])

        # Save metrics
        self.save_metrics(metrics)


    def save_metrics(self, metrics):
        with open("{}/metrics.json".format(Logger.LOG_DIR_PATH), 'w') as fp:
            json.dump(metrics, fp, indent=2)


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


    def update_gt_state(self, visible_objects, last_event):

        # Update mapping among belief object ids and ground truth ones (required for evaluation)
        # self.update_objs_id_mapping(belief_state.visible_objects, last_event)
        self.update_objs_id_mapping(visible_objects, last_event)

        # Update ground truth objects for evaluation
        self.update_visible_objects_ground_truth(last_event)

        # Update ground truth predicates
        gt_predicates = defaultdict(dict)

        hand_free = True
        for obj_id in self.gt_objs_seen:
            obj = [obj for obj in last_event.metadata['objects'] if obj['objectId'] == obj_id][0]
            obj_picked = obj['isPickedUp']
            obj_open = obj['isOpen']
            obj_visible = obj['visible']

            hand_free = hand_free and not obj_picked

            # Compute object distance by adding the object height, otherwise the simulator uses the lower point
            # of the object as its height (y-coordinate in the simulator)
            obj_pos = [obj['position']['x'],
                       obj['position']['y'] + obj['axisAlignedBoundingBox']['size']['y'],
                       obj['position']['z']]
            agent_pos = [last_event.metadata['agent']['position']['x'],
                         last_event.metadata['agent']['position']['y'],
                         last_event.metadata['agent']['position']['z']]
            obj_distance = np.linalg.norm(np.array(obj_pos) - np.array(agent_pos))
            obj_close = bool(obj_distance < 1.5)

            # Check if the object has been inspected, i.e., if it has been reached a state where the object is
            # visible and the distance is lower than the manipulation distance (1.5 meters)
            obj_inspected = bool(obj_distance < 1.5)
            for pred in self.gt_predicates:
                if pred.split('(')[0].lower() == 'inspected' and pred.split('(')[1].strip()[:-1].lower() == obj_id:
                    obj_inspected = True

            # Look if object is contained in another one
            containers = []
            if obj['parentReceptacles'] is not None:
                for container_name in obj['parentReceptacles']:
                    # Filter ground truth predicates by removing useless predicates, i.e., predicates which involve
                    # not considered objects such as floors and walls.
                    if not container_name.split('|')[0].lower().strip() in Configuration.IRRELEVANT_CLASSES:
                        containers.append(container_name)

            gt_predicates[obj_id] = {'holding': obj_picked,
                                     'open': obj_open,
                                     'discovered': True,
                                     'viewing': obj_visible,
                                     'close_to': obj_close,
                                     'inspected': obj_inspected,
                                     'on': containers}

        # Generate visible predicates list
        gt_predicates = dict(gt_predicates)

        # Get unary predicates
        gt_predicates_list = ["{}({})".format(k2,k) for k,v in gt_predicates.items()
                                   for k2, v2 in v.items() if type(v2) == type(True) and v2]
        if hand_free:
            gt_predicates_list.append("hand_free()")

        # Get binary predicates
        gt_predicates_list.extend(["{}({},{})".format(k2,k,v3) for k,v in gt_predicates.items()
                                        for k2, v2 in v.items() if type(v2) != type(True) for v3 in v2])

        self.gt_predicates = gt_predicates_list



    def eval_goal_achievement_objectnav(self, all_objects, last_event, agent_path, controller):

        assert self.shortest_path is not None, "Set shortest_path in Evaluator.py for computing SPL for " \
                                               "the Object Goal Navigation task."

        # Update ground truth state
        self.update_gt_state(all_objects, last_event)

        # Get current goal from pddl problem file
        with open(Configuration.PDDL_PROBLEM_PATH, 'r') as f:
            data = [el.strip() for el in f.read().split("\n") if not el.strip().startswith(";")]
            goal = re.findall(":goal.*","".join(data))[0]

        goal_facts = sorted(re.findall("\([^()]*\)", goal))
        goal_objects = goal_facts[0]

        goal_objects = [el for el in goal_objects.strip()[1:-1].split() if el.strip() != "-"]
        goal_objects_id = {goal_objects[i]: goal_objects[i+1] for i in range(len(goal_objects)) if i % 2 == 0}
        goal_objects_types = list(goal_objects_id.values())
        goal_object_type = goal_objects_types[0]


        objects_visibles_and_near_to_agent = [o['objectType'].lower() for o in last_event.metadata['objects'] if o['visible']]

        success = 0
        if goal_object_type in objects_visibles_and_near_to_agent:
            success = 1
        else:
            visible_objs_instances = []
            [visible_objs_instances.append(obj['objectId'])
             for obj in controller.last_event.metadata['objects']
             if obj['objectId'] in controller.last_event.instance_detections2D
             and obj['objectId'] not in visible_objs_instances]

            goal_objs = [o for o in controller.last_event.metadata['objects'] if goal_object_type.lower() == o['objectType'].lower()]

            for obj in goal_objs:
                if Configuration.TASK == Configuration.TASK_OGN_ROBOTHOR and obj['distance'] < 1:
                    success = 1
                    break
                elif Configuration.TASK == Configuration.TASK_OGN_ITHOR and obj['distance'] < 1.5:
                    success = 1
                    break

        # Set agent height in the path to the floor one, since metrics in ithor dataset uses floor y-coordinates
        # rather than agent height.
        # Similar issue in RoboTHOR: https://github.com/allenai/robothor-challenge/issues/24
        if Configuration.TASK == Configuration.TASK_OGN_ITHOR:
            y_floor = self.shortest_path[0]['y']
            agent_path = [{'x': p['x'], 'y': y_floor, 'z': p['z']} for p in agent_path]

        spl = compute_single_spl(agent_path, self.shortest_path, bool(success))

        uppercase_goal_obj_type = [o['objectType'] for o in last_event.metadata['objects']
                                   if o['objectType'].lower() == goal_object_type][0]

        try:
            dts_distance = path_distance(get_shortest_path_to_object_type(controller, uppercase_goal_obj_type,
                                                                          last_event.metadata['agent']['position'],
                                                                          initial_rotation=last_event.metadata['agent']['rotation']))
        except:
            Logger.write("WARNING: computing distance to success as the distance from nearest goal object instance of the goal class,"
                         " since ai2thor metrics cannot find a valid path.")
            dts_distance = min([o['distance'] for o in last_event.metadata['objects'] if o['objectType'].lower() == goal_object_type])
            dts_distance = max(0, dts_distance - 1)  # Remove the maximum distance from the object to succeeds


        if success == 1 and dts_distance != 0:
            Logger.write('WARNING: distance to success is greater than 0 even if success is True. Check Evaluator.py')
            dts_distance = 0



        metrics = json.load(open("{}/metrics.json".format(Logger.LOG_DIR_PATH),))

        metrics['goal'] = {"success": success,
                           "spl": spl,
                           "distance_to_success": dts_distance}

        all_success.append(success)
        all_spl.append(spl)
        all_dts.append(dts_distance)
        episodes = len(all_success)

        Logger.write("Average metrics over {} episodes:\n\t AVG success: {}\n\t AVG SPL: {}\n\t AVG dts: {}"
                     .format(episodes, sum(all_success)/episodes,
                             sum(all_spl)/episodes, sum(all_dts)/episodes))

        with open("{}/metrics.json".format(Logger.LOG_DIR_PATH), 'w') as f:
            json.dump(metrics, f, indent=2)


    def eval_goal_achievement_open(self, all_objects, last_event):

        # Update ground truth state
        self.update_gt_state(all_objects, last_event)

        # Get current goal from pddl problem file
        with open(Configuration.PDDL_PROBLEM_PATH, 'r') as f:
            data = [el.strip() for el in f.read().split("\n") if not el.strip().startswith(";")]
            goal = re.findall(":goal.*","".join(data))[0]

        goal_facts = sorted(re.findall("\([^()]*\)", goal))
        goal_objects = goal_facts[0]
        goal_facts = goal_facts[1:]

        goal_predicate_type = [goal_facts[i].strip()[1:-1].split()[0] for i in range(len(goal_facts))
                                if not 'inspected' in goal_facts[i].strip()[1:-1].split()[0]
                               and not 'manipulated' in goal_facts[i].strip()[1:-1].split()[0]][0]

        goal_objects = [el for el in goal_objects.strip()[1:-1].split() if el.strip() != "-"]
        goal_objects_id = {goal_objects[i]: goal_objects[i+1] for i in range(len(goal_objects)) if i % 2 == 0}
        goal_objects_types = list(goal_objects_id.values())

        gt_satisfied_facts = None
        success = 0
        for sat_goal_fact in [el for el in self.gt_predicates if el.startswith(goal_predicate_type)]:
            if bool(np.all(['(' + obj + '|' in sat_goal_fact.lower()
                    or ',' + obj + '|' in sat_goal_fact.lower()
                    or '|' + obj + ')' in sat_goal_fact.lower()
                    for obj in goal_objects_types])) is True:
                gt_satisfied_facts = sat_goal_fact
                success = 1
                break

        # Compute distance to success, i.e., minimum distance from an instance of the goal "contained" object to
        # an instance of the goal "container" object
        dts_distance = None
        if success:
            dts_distance = 0
        else:

            if Configuration.TASK == Configuration.TASK_ON:
                contained_type = goal_objects_types[0]
                container_type = goal_objects_types[1]

                contained_instances = [o for o in last_event.metadata['objects'] if o['objectType'].lower() == contained_type.lower()]
                container_instances = [o for o in last_event.metadata['objects'] if o['objectType'].lower() == container_type.lower()]
                dts_distance = 10000
                for contained_obj_inst in contained_instances:
                    for container_obj_inst in container_instances:
                        contained_center_3D = np.array([contained_obj_inst['axisAlignedBoundingBox']['center']['x'],
                                                        contained_obj_inst['axisAlignedBoundingBox']['center']['y'],
                                                        contained_obj_inst['axisAlignedBoundingBox']['center']['z']])
                        container_center_3D = np.array([container_obj_inst['axisAlignedBoundingBox']['center']['x'],
                                                        container_obj_inst['axisAlignedBoundingBox']['center']['y'],
                                                        container_obj_inst['axisAlignedBoundingBox']['center']['z']])
                        distance = np.linalg.norm(contained_center_3D - container_center_3D)

                        if distance < dts_distance:
                            dts_distance = distance

            elif Configuration.TASK == Configuration.TASK_OPEN or Configuration.TASK == Configuration.TASK_CLOSE:
                goal_obj_type = goal_objects_types[0]
                goal_obj_instances = [o for o in last_event.metadata['objects'] if o['objectType'].lower() == goal_obj_type.lower()]
                dts_distance = 10000
                for goal_obj_inst in goal_obj_instances:
                    agent_center = np.array([last_event.metadata['agent']['position']['x'],
                                             last_event.metadata['agent']['position']['z']])
                    goal_obj_center = np.array([goal_obj_inst['axisAlignedBoundingBox']['center']['x'],
                                                goal_obj_inst['axisAlignedBoundingBox']['center']['z']])
                    distance = np.linalg.norm(agent_center - goal_obj_center)

                    if distance < dts_distance:
                        dts_distance = distance

        metrics = json.load(open("{}/metrics.json".format(Logger.LOG_DIR_PATH),))

        metrics['goal'] = {"success": success,
                           "distance_to_success": dts_distance,
                           "gt_satisfied_facts": gt_satisfied_facts}

        with open("{}/metrics.json".format(Logger.LOG_DIR_PATH), 'w') as f:
            json.dump(metrics, f, indent=2)


    def eval_goal_achievement_close(self, all_objects, last_event):

        # Update ground truth state
        self.update_gt_state(all_objects, last_event)

        # Get current goal from pddl problem file
        with open(Configuration.PDDL_PROBLEM_PATH, 'r') as f:
            data = [el.strip() for el in f.read().split("\n") if not el.strip().startswith(";")]
            goal = re.findall(":goal.*","".join(data))[0]

        goal_facts = sorted(re.findall("\([^()]*\)", goal))
        goal_objects = goal_facts[0]
        goal_facts = goal_facts[1:]

        goal_objects = [el for el in goal_objects.strip()[1:-1].split() if el.strip() != "-"]

        goal_objects_id = {goal_objects[i]: goal_objects[i+1] for i in range(len(goal_objects)) if i % 2 == 0}
        goal_objects_types = list(goal_objects_id.values())
        goal_object_type = goal_objects_types[0]

        success = 0
        gt_satisfied_facts = None
        for obj in last_event.metadata['objects']:
            if obj['objectType'].lower() == goal_object_type and not obj['isOpen']:
                success = 1
                gt_satisfied_facts = "Closed({})".format(obj['objectId'])
                break

        # Compute distance to success, i.e., minimum distance from an instance of the goal "contained" object to
        # an instance of the goal "container" object
        dts_distance = None
        if success:
            dts_distance = 0
        else:

            if Configuration.TASK == Configuration.TASK_ON:
                contained_type = goal_objects_types[0]
                container_type = goal_objects_types[1]

                contained_instances = [o for o in last_event.metadata['objects'] if o['objectType'].lower() == contained_type.lower()]
                container_instances = [o for o in last_event.metadata['objects'] if o['objectType'].lower() == container_type.lower()]
                dts_distance = 10000
                for contained_obj_inst in contained_instances:
                    for container_obj_inst in container_instances:
                        contained_center_3D = np.array([contained_obj_inst['axisAlignedBoundingBox']['center']['x'],
                                                        contained_obj_inst['axisAlignedBoundingBox']['center']['y'],
                                                        contained_obj_inst['axisAlignedBoundingBox']['center']['z']])
                        container_center_3D = np.array([container_obj_inst['axisAlignedBoundingBox']['center']['x'],
                                                        container_obj_inst['axisAlignedBoundingBox']['center']['y'],
                                                        container_obj_inst['axisAlignedBoundingBox']['center']['z']])
                        distance = np.linalg.norm(contained_center_3D - container_center_3D)

                        if distance < dts_distance:
                            dts_distance = distance

            elif Configuration.TASK == Configuration.TASK_OPEN or Configuration.TASK == Configuration.TASK_CLOSE:
                goal_obj_type = goal_objects_types[0]
                goal_obj_instances = [o for o in last_event.metadata['objects'] if o['objectType'].lower() == goal_obj_type.lower()]
                dts_distance = 10000
                for goal_obj_inst in goal_obj_instances:
                    agent_center = np.array([last_event.metadata['agent']['position']['x'],
                                             # last_event.metadata['agent']['position']['y'],
                                             last_event.metadata['agent']['position']['z']])
                    goal_obj_center = np.array([goal_obj_inst['axisAlignedBoundingBox']['center']['x'],
                                                # container_obj_inst['axisAlignedBoundingBox']['center']['y'],
                                                goal_obj_inst['axisAlignedBoundingBox']['center']['z']])
                    distance = np.linalg.norm(agent_center - goal_obj_center)

                    if distance < dts_distance:
                        dts_distance = distance

        metrics = json.load(open("{}/metrics.json".format(Logger.LOG_DIR_PATH),))

        metrics['goal'] = {"success": success,
                           "distance_to_success": dts_distance,
                           # "belief_satisfied_facts": belief_satisfied_facts,
                           "gt_satisfied_facts": gt_satisfied_facts}

        with open("{}/metrics.json".format(Logger.LOG_DIR_PATH), 'w') as f:
            json.dump(metrics, f, indent=2)


    def eval_goal_achievement_on(self, all_objects, last_event):

        # Update ground truth state
        self.update_gt_state(all_objects, last_event)

        # Get current goal from pddl problem file
        with open(Configuration.PDDL_PROBLEM_PATH, 'r') as f:
            data = [el.strip() for el in f.read().split("\n") if not el.strip().startswith(";")]
            goal = re.findall(":goal.*","".join(data))[0]
            facts = re.findall(":init.*\(:goal","".join(data))[0]

        goal_facts = sorted(re.findall("\([^()]*\)", goal))
        goal_objects = goal_facts[0]
        goal_facts = goal_facts[1:]
        goal_predicate_type = goal_facts[0].strip()[1:-1].split()[0]
        goal_objects = [el for el in goal_objects.strip()[1:-1].split() if el.strip() != "-"]
        goal_objects_id = {goal_objects[i]: goal_objects[i+1] for i in range(len(goal_objects)) if i % 2 == 0}
        goal_objects_types = list(goal_objects_id.values())

        gt_satisfied_facts = None
        success = 0
        for sat_goal_fact in [el for el in self.gt_predicates if el.startswith(goal_predicate_type)]:
            if bool(np.all(['(' + obj + '|' in sat_goal_fact.lower()
                    or ',' + obj + '|' in sat_goal_fact.lower()
                    or '|' + obj + ')' in sat_goal_fact.lower()
                    for obj in goal_objects_types])) is True:
                gt_satisfied_facts = sat_goal_fact
                success = 1
                break

        # Compute distance to success, i.e., minimum distance from an instance of the goal "contained" object to
        # an instance of the goal "container" object
        dts_distance = None
        if success:
            dts_distance = 0
        else:

            if Configuration.TASK == Configuration.TASK_ON:
                contained_type = goal_objects_types[0]
                container_type = goal_objects_types[1]

                contained_instances = [o for o in last_event.metadata['objects'] if o['objectType'].lower() == contained_type.lower()]
                container_instances = [o for o in last_event.metadata['objects'] if o['objectType'].lower() == container_type.lower()]
                dts_distance = 10000
                for contained_obj_inst in contained_instances:
                    for container_obj_inst in container_instances:
                        contained_center_3D = np.array([contained_obj_inst['axisAlignedBoundingBox']['center']['x'],
                                                        contained_obj_inst['axisAlignedBoundingBox']['center']['y'],
                                                        contained_obj_inst['axisAlignedBoundingBox']['center']['z']])
                        container_center_3D = np.array([container_obj_inst['axisAlignedBoundingBox']['center']['x'],
                                                        container_obj_inst['axisAlignedBoundingBox']['center']['y'],
                                                        container_obj_inst['axisAlignedBoundingBox']['center']['z']])
                        distance = np.linalg.norm(contained_center_3D - container_center_3D)

                        if distance < dts_distance:
                            dts_distance = distance

            elif Configuration.TASK == Configuration.TASK_OPEN or Configuration.TASK == Configuration.TASK_CLOSE:
                goal_obj_type = goal_objects_types[0]
                goal_obj_instances = [o for o in last_event.metadata['objects'] if o['objectType'].lower() == goal_obj_type.lower()]
                dts_distance = 10000
                for goal_obj_inst in goal_obj_instances:
                    agent_center = np.array([last_event.metadata['agent']['position']['x'],
                                             # last_event.metadata['agent']['position']['y'],
                                             last_event.metadata['agent']['position']['z']])
                    goal_obj_center = np.array([goal_obj_inst['axisAlignedBoundingBox']['center']['x'],
                                                # container_obj_inst['axisAlignedBoundingBox']['center']['y'],
                                                goal_obj_inst['axisAlignedBoundingBox']['center']['z']])
                    distance = np.linalg.norm(agent_center - goal_obj_center)

                    if distance < dts_distance:
                        dts_distance = distance

        metrics = json.load(open("{}/metrics.json".format(Logger.LOG_DIR_PATH),))

        metrics['goal'] = {"success": success,
                           "distance_to_success": dts_distance,
                           # "belief_satisfied_facts": belief_satisfied_facts,
                           "gt_satisfied_facts": gt_satisfied_facts}

        with open("{}/metrics.json".format(Logger.LOG_DIR_PATH), 'w') as f:
            json.dump(metrics, f, indent=2)

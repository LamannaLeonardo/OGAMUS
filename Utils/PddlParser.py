# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import random
import re

from ai2thor.controller import Controller

import numpy as np

import Configuration
from Utils import Logger

np.random.seed(Configuration.RANDOM_SEED)
random.seed(Configuration.RANDOM_SEED)


# Update PDDL state with max-score ones and consider only objects involved in the goal or objects related to goal ones.
def update_pddl_state(objects, predicates, objects_counting, objects_scores):

    inspected_objs = [pred.split('(')[1].strip()[:-1] for pred in predicates if pred.startswith("inspected")]

    score_threshold = 0.2

    # Initialize filtered predicates (i.e. predicates involving goal objects) with all nullary ones.
    considered_goal_objs = []
    holding_obj = [p for p in predicates if 'holding' in p]
    if len(holding_obj) > 0:
        holding_obj = holding_obj[0].replace("holding(", "").strip()[:-1]
    else:
        holding_obj = None

    for goal_obj_type in Configuration.GOAL_OBJECTS:

        if holding_obj is not None and goal_obj_type == holding_obj.split('_')[0]:
            considered_goal_objs.append(holding_obj)
        else:
            considered_object_instances = {k: v for k, v in objects_counting.items()
                                     if v >= Configuration.OBJ_COUNT_THRSH and k.split('_')[0] == goal_obj_type}
            goal_object_instances = {k: v for k, v in objects_scores.items()
                                     if k in considered_object_instances.keys() and k.split('_')[0] == goal_obj_type}

            if len(goal_object_instances.keys()) > 0:
                goal_obj_inst = max(goal_object_instances, key=goal_object_instances.get)

                # Get all object instances which have been inspected
                goal_object_instances_inspected = {k: v for k, v in goal_object_instances.items()
                                                   if k in inspected_objs}
                if len(goal_object_instances_inspected) > 0:
                    goal_obj_inst_inspected = max(goal_object_instances_inspected, key=goal_object_instances_inspected.get)

                    goal_obj_inst_score = goal_object_instances[goal_obj_inst]
                    goal_obj_inst_inspected_score = goal_object_instances_inspected[goal_obj_inst_inspected]

                    if np.abs(goal_obj_inst_inspected_score - goal_obj_inst_score) <= score_threshold:
                        considered_goal_objs.append(goal_obj_inst_inspected)
                    else:
                        considered_goal_objs.append(goal_obj_inst)

                else:
                    considered_goal_objs.append(goal_obj_inst)

    removed_objs = [obj['id'] for obj_type in Configuration.GOAL_OBJECTS for obj in objects[obj_type]
                    if obj['id'] not in considered_goal_objs]

    filtered_objs = [obj['id'] for obj_type in objects.keys() for obj in objects[obj_type]
                     if obj['id'] not in removed_objs]
    filtered_preds = [pred for pred in predicates
                      if set([el for el in pred.split('(')[1].strip()[:-1].split(',') if el.strip() != '']).issubset(set(filtered_objs))
                      or len([el for el in pred.split('(')[1].strip()[:-1].split(',') if el.strip() != '']) == 0]

    with open("./OGAMUS/Plan/PDDL/facts.pddl", "r") as f:
        old_pddl_state = [el.strip() for el in f.read().split("\n") if el.strip() != '']
        old_facts = re.findall(":init.*\(:goal", "".join(old_pddl_state))[0]

        old_objs = [el for el in re.findall(":objects(.*?)\)", "++".join(old_pddl_state))[0].split("++") if el.strip() != ""]
        old_state = "\n" + "\n".join(re.findall("\([^()]*\)", old_facts))

        # Replace facts
        new_facts = "\n" + "\n".join(sorted(["({} {})".format(p.strip().split("(")[0],
                                                              " ".join(p.strip().split("(")[1][:-1].split(",")))
                                             for p in filtered_preds]))

        if old_state!="\n":
            new_pddl = "\n".join(old_pddl_state).replace(old_state, new_facts)
        else:
            new_pddl = "\n".join(old_pddl_state).replace("(:init", "(:init" + new_facts)

        # Replace objects
        new_objs = ["{} - {}".format(obj_id, obj_id.split('_')[0]) for obj_id in filtered_objs]

        if len(old_objs) > 0:
            new_pddl = new_pddl.replace("\n".join(old_objs), "\n".join(new_objs))
        else:
            new_pddl = new_pddl.replace("(:objects", "(:objects\n" + "\n".join(new_objs))

    with open("./OGAMUS/Plan/PDDL/facts.pddl", "w") as f:
        f.write(new_pddl)


def get_operator_effects(op_name):

    with open("OGAMUS/Plan/PDDL/domain.pddl", "r") as f:
        data = [el.strip() for el in f.read().split("\n")]

    all_action_schema = " ".join(data)[" ".join(data).index(":action"):]
    action_schema = re.findall(":action {}(.*?)(?::action|$)".format(op_name), all_action_schema)[0]
    effect_neg = re.findall("\(not[^)]*\)\)", action_schema[action_schema.find("effect"):])
    effect_pos = [el for el in re.findall("\([^()]*\)", action_schema[action_schema.find("effect"):])
                     if el not in [el.replace("(not", "").strip()[:-1] for el in effect_neg]
                     and "".join(el.split()) != "(and)" and "".join(el.split()) != "()"]

    effect_pos = [e for e in effect_pos if '?x' not in e]
    effect_neg = [e for e in effect_neg if '?x' not in e]

    return effect_pos + effect_neg


def goal_predicate_open(scene_name, scene_objects, controller):
    openable_objects = [obj for obj in scene_objects if obj['openable']
                        and obj['objectType'].lower().strip() in Configuration.OPENABLE_OBJS
                        and obj['axisAlignedBoundingBox']['center']['y'] > 0.4
                        and (obj['parentReceptacles'] is None or
                             len([o for o in obj['parentReceptacles'] if o.lower().split('|')[0]
                                  in Configuration.OPENABLE_OBJS]) == 0)]
    goal = None

    sampling_openable_objects = copy.deepcopy(openable_objects)

    while len(sampling_openable_objects) > 0:
        feasible = True
        openable_object = np.random.choice(sampling_openable_objects)

        if len([o for o in scene_objects if o['objectType'] == openable_object['objectType'] and o['isOpen']]) <= 3:

            for obj in [o for o in scene_objects if
                        o['objectType'] == openable_object['objectType'] and o['isOpen']]:
                action_result = controller.step(action='CloseObject', objectId=obj['objectId'], forceAction=True)
                if not action_result.metadata['lastActionSuccess']:
                    feasible = False

            if feasible:
                action_result = controller.step(action='OpenObject', objectId=openable_object['objectId'],
                                                forceAction=True)
                if action_result.metadata['lastActionSuccess']:
                    goal = '(exists (?o1 - {}) (and (inspected ?o1) (open ?o1)))'.format(openable_object['objectType'].lower())
                    break

        sampling_openable_objects = [o for o in sampling_openable_objects if o != openable_object]

    return goal


def goal_predicate_close(scene_name, scene_objects, controller):

    openable_objects = [obj for obj in scene_objects if obj['openable']
                        and obj['axisAlignedBoundingBox']['center']['y'] > 0.4
                        and obj['objectType'].lower().strip() in Configuration.OPENABLE_OBJS
                        and (obj['parentReceptacles'] is None or
                             len([o for o in obj['parentReceptacles'] if o.lower().split('|')[0]
                                  in Configuration.OPENABLE_OBJS]) == 0)]
    goal = None

    sampling_openable_objects = copy.deepcopy(openable_objects)

    while len(sampling_openable_objects) > 0:
        feasible = True
        openable_object = np.random.choice(sampling_openable_objects)

        if len([o for o in scene_objects if o['objectType'] == openable_object['objectType'] and not o['isOpen']]) <= 3:

            for obj in [o for o in scene_objects if o['objectType'] == openable_object['objectType'] and not o['isOpen']]:
                action_result = controller.step(action='OpenObject', objectId=obj['objectId'], forceAction=True)
                if not action_result.metadata['lastActionSuccess']:
                    feasible = False

            if feasible:
                action_result = controller.step(action='CloseObject', objectId=openable_object['objectId'], forceAction=True)
                if action_result.metadata['lastActionSuccess']:
                    goal = '(exists (?o1 - {}) (and (inspected ?o1) (not (open ?o1))))'.format(openable_object['objectType'].lower())
                    break

        sampling_openable_objects = [o for o in sampling_openable_objects if o != openable_object]

    return goal


def goal_predicate_on(scene_name, scene_objects, controller):
    satisfiable_goal = False

    # Randomly choose a receptable object
    receptacle_objects = [obj for obj in scene_objects
                          if obj['receptacle'] and not obj['openable'] and obj[
                              'objectType'].lower().strip() in Configuration.RECEPTACLE_OBJS
                          and obj['objectType'].lower() not in ['floor', 'wall', 'chair']  # chair should not be receptacle
                          and obj['axisAlignedBoundingBox']['center']['y'] > 0.4]

    random.shuffle(receptacle_objects)

    for receptacle_object in receptacle_objects:

        # Get pickupable objects
        pickupable_objects = [obj for obj in scene_objects
                              if obj['pickupable'] and obj != receptacle_object
                              and obj['axisAlignedBoundingBox']['center']['y'] > 0.4
                              and (obj['parentReceptacles'] is None or
                                   (receptacle_object['objectId'] not in obj['parentReceptacles']
                                   # and receptacle_object['objectId'].split('|') not in
                                   and receptacle_object['objectId'].split('|')[0] not in
                                    [obj_name.split('|')[0] for obj_name in obj['parentReceptacles']]))
                              ]

        random.shuffle(pickupable_objects)

        # Remove hidden objects from pickupable ones
        removed_pickupable = []
        for obj in pickupable_objects:
            if obj['parentReceptacles'] is not None:
                for receptacle_id in obj['parentReceptacles']:
                    receptacle_obj = [obj for obj in scene_objects if obj['objectId'] == receptacle_id][0]
                    if receptacle_obj['openable']:
                        removed_pickupable.append(obj)
                        break

        pickupable_objects = [obj for obj in pickupable_objects if obj not in removed_pickupable]

        for pickupable_object in pickupable_objects:
            action_result = controller.step(action='PickupObject', objectId=pickupable_object['objectId'], forceAction=True)
            if action_result.metadata['lastActionSuccess']:
                action_result = controller.step(action='PutObject', objectId=receptacle_object['objectId'], forceAction=True)
                if action_result.metadata['lastActionSuccess']:


                    contained_type = pickupable_object['objectType']
                    container_type = receptacle_object['objectType']

                    all_contained_type_objs = [o for o in scene_objects if o['objectType'] == contained_type]

                    already_satisfied = False

                    for o in [o for o in all_contained_type_objs if o['parentReceptacles'] is not None]:
                        if container_type in [el.split('|')[0] for el in o['parentReceptacles']]\
                                or container_type in [el.split('|')[-1] for el in o['parentReceptacles']]:
                            already_satisfied = True

                    if not already_satisfied:
                        contained_obj = copy.deepcopy(pickupable_object)
                        container_obj = copy.deepcopy(receptacle_object)
                        satisfiable_goal = True
                        break
                    else:
                        controller.reset(scene=scene_name, headless=True)
                else:
                    controller.reset(scene=scene_name, headless=True)

        if satisfiable_goal:
            break


    if not satisfiable_goal:
        Logger.write('Cannot find a satisfiable goal in PddlParser.py')
        exit()

    goal = '(exists (?o1 - {} ?o2 - {}) (on ?o1 ?o2))'.format(contained_obj['objectType'].lower(),
                                                              container_obj['objectType'].lower())

    return goal


def set_goal(goal):

    Configuration.GOAL_OBJECTS = [el.split()[0] for el in re.findall("\([^()]*\)", goal)[0][1:-1].split('-')[1:]]

    # Update goal in PDDL problem file
    with open("./OGAMUS/Plan/PDDL/facts.pddl", 'r') as f:
        data = f.read().split("\n")
        for i in range(len(data)):
            row = data[i]

            if row.strip().find("(:goal") != -1:
                end_index = i + 1

                if data[i].strip().startswith(")"):
                    data[i] = ")\n(:goal \n(and \n{}) \n))".format(goal)
                else:
                    data[i] = "(:goal \n(and \n{}) \n))".format(goal)

    with open("./OGAMUS/Plan/PDDL/facts.pddl", 'w') as f:
        [f.write(el + "\n") for el in data[:end_index]]


    # # Copy problem file in result directory
    # shutil.copyfile("./OGAMUS/Plan/PDDL/facts.pddl", os.path.join(Logger.LOG_DIR_PATH, "facts_{}.pddl".format(scene)))


def get_generated_goal_on(scene):

    controller = Controller(scene=scene, headless=True)
    event = copy.deepcopy(controller.step('Pass'))
    scene_objects = event.metadata['objects']

    goal = None
    goal = goal_predicate_on(scene, scene_objects, controller)

    assert goal is not None, 'Failed generating goal for scene {}'.format(scene)

    all_pos = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    random_start_pos = np.random.choice(all_pos)
    event.metadata['agent']['position']['x'] = random_start_pos['x']
    event.metadata['agent']['position']['y'] = random_start_pos['y']
    event.metadata['agent']['position']['z'] = random_start_pos['z']
    controller.stop()

    return goal, event


def get_generated_goal_open(scene):

    controller = Controller(scene=scene, headless=True)
    event = copy.deepcopy(controller.step('Pass'))
    scene_objects = event.metadata['objects']
    goal = goal_predicate_open(scene, scene_objects, controller)

    all_pos = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    random_start_pos = np.random.choice(all_pos)
    event.metadata['agent']['position']['x'] = random_start_pos['x']
    event.metadata['agent']['position']['y'] = random_start_pos['y']
    event.metadata['agent']['position']['z'] = random_start_pos['z']

    controller.stop()

    return goal, event


def get_generated_goal_close(scene):

    controller = Controller(scene=scene, headless=True)
    event = copy.deepcopy(controller.step('Pass'))
    scene_objects = event.metadata['objects']
    goal = goal_predicate_close(scene, scene_objects, controller)

    all_pos = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    random_start_pos = np.random.choice(all_pos)
    event.metadata['agent']['position']['x'] = random_start_pos['x']
    event.metadata['agent']['position']['y'] = random_start_pos['y']
    event.metadata['agent']['position']['z'] = random_start_pos['z']

    controller.stop()

    return goal, event
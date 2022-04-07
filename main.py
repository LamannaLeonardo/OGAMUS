# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import json
import os
import random
import shutil

import numpy as np
import torch.cuda.random

import Configuration
from OGAMUS.Agent import Agent
from Utils import PddlParser, Logger


def main():

    # DEBUG
    starting_episode = 0

    if not os.path.exists('Utils/pretrained_models'):
        print('Please download the neural network models from '
              'https://drive.google.com/drive/folders/1UjADpBeBOMUKXQt-qSULIP3vM90zr_MR?usp=sharing , '
              'then move the downloaded models into a directory '
              'named "pretrained_models", and move the directory "pretrained_models" into the directory "Utils"')
    model_files = os.listdir('Utils/pretrained_models')
    if 'faster-rcnn_12classes.pth' not in model_files:
        print('File "faster-rcnn_12classes.pth" not found in "Utils/pretrained_models", please download the neural '
              'network models from  '
              'https://drive.google.com/drive/folders/1UjADpBeBOMUKXQt-qSULIP3vM90zr_MR?usp=sharing , '
              'then move the downloaded models into a directory '
              'named "pretrained_models", '
              'and move the directory "pretrained_models" into the directory "Utils"')
    elif 'faster-rcnn_118classes.pkl' not in model_files:
        print('File "faster-rcnn_118classes.pkl" not found in "Utils/pretrained_models", please download the neural '
              'network models from  '
              'https://drive.google.com/drive/folders/1UjADpBeBOMUKXQt-qSULIP3vM90zr_MR?usp=sharing , '
              'then move the downloaded models into a directory '
              'named "pretrained_models", '
              'and move the directory "pretrained_models" into the directory "Utils"')
    elif 'on_predictor.pth' not in model_files:
        print('File "on_predictor.pth" not found in "Utils/pretrained_models", please download the neural '
              'network models from  '
              'https://drive.google.com/drive/folders/1UjADpBeBOMUKXQt-qSULIP3vM90zr_MR?usp=sharing , '
              'then move the downloaded models into a directory '
              'named "pretrained_models", '
              'and move the directory "pretrained_models" into the directory "Utils"')
    elif 'open_predictor.pth' not in model_files:
        print('File "open_predictor.pth" not found in "Utils/pretrained_models", please download the neural '
              'network models from  '
              'https://drive.google.com/drive/folders/1UjADpBeBOMUKXQt-qSULIP3vM90zr_MR?usp=sharing , '
              'then move the downloaded models into a directory '
              'named "pretrained_models", '
              'and move the directory "pretrained_models" into the directory "Utils"')
    elif 'obj_classes_coco.pkl' not in model_files:
        print('File "obj_classes_coco.pkl" not found in "Utils/pretrained_models", please download the neural '
              'network models from  '
              'https://drive.google.com/drive/folders/1UjADpBeBOMUKXQt-qSULIP3vM90zr_MR?usp=sharing , '
              'then move the downloaded models into a directory '
              'named "pretrained_models", '
              'and move the directory "pretrained_models" into the directory "Utils"')
    elif 'obj_classes_robothor_ogn.pkl' not in model_files:
        print('File "obj_classes_robothor_ogn.pkl" not found in "Utils/pretrained_models", please download the neural '
              'network models from  '
              'https://drive.google.com/drive/folders/1UjADpBeBOMUKXQt-qSULIP3vM90zr_MR?usp=sharing , '
              'then move the downloaded models into a directory '
              'named "pretrained_models", '
              'and move the directory "pretrained_models" into the directory "Utils"')


    # Set input arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('-t', '--task', help="Task name,  e.g. on, close, open or ogn",
                             type=str, default='open')
    args_parser.add_argument('-obj', '--object_ground_truth', help="Use ground truth object detections",
                             default=None, action=argparse.BooleanOptionalAction)

    # Get input arguments
    args = args_parser.parse_args()
    task = args.task
    object_ground_truth = args.object_ground_truth

    if object_ground_truth is not None:
        Configuration.GROUND_TRUTH_OBJS = True

    if task is not None:
        assert task in [Configuration.TASK_ON, Configuration.TASK_OGN_ROBOTHOR, Configuration.TASK_OPEN,
                        Configuration.TASK_CLOSE, Configuration.TASK_OGN_ITHOR, Configuration.TASK_OGN_ROBOTHOR], \
            "Error: input task must be one of: on, open, close, ogn or ogn_ithor"
        Configuration.TASK = task
        Configuration.DATASET = 'test_set_{}'.format(Configuration.TASK)

        if Configuration.TASK == Configuration.TASK_OGN_ROBOTHOR:
            if not Configuration.MAX_ITER != 500:
                Logger.write("Warning: setting maximum number of steps to 500 according to the "
                             "RoboTHOR object navigation challenge.")
                Configuration.MAX_ITER = 500
        Configuration.RESULTS_DIR = 'Results/{}_steps{}'.format(Configuration.DATASET, Configuration.MAX_ITER)

    # Set random seed
    np.random.seed(Configuration.RANDOM_SEED)
    random.seed(Configuration.RANDOM_SEED)
    torch.manual_seed(Configuration.RANDOM_SEED)

    # Set environment path variables (for x server communication from WSL2 to Windows GUI)
    if Configuration.USING_WSL2_WINDOWS:
        os.environ['DISPLAY'] = "{}:0.0".format(Configuration.IP_ADDRESS)
        os.environ['LIBGL_ALWAYS_INDIRECT'] = "0"

    dataset = json.load(open(os.path.join(Configuration.DATASET_DIR, '{}.json'.format(Configuration.DATASET)), 'r'))

    if os.path.exists(Configuration.RESULTS_DIR):
        count = 1
        while os.path.exists(Configuration.RESULTS_DIR + '({})'.format(count)):
            count += 1
        Configuration.RESULTS_DIR = Configuration.RESULTS_DIR + '({})'.format(count)
    os.mkdir(Configuration.RESULTS_DIR)

    # Copy domain file for the task
    if Configuration.TASK == Configuration.TASK_ON:
        shutil.copyfile("OGAMUS/Plan/PDDL/domain_on.pddl", "OGAMUS/Plan/PDDL/domain.pddl")
    if Configuration.TASK == Configuration.TASK_OPEN or Configuration.TASK == Configuration.TASK_CLOSE:
        shutil.copyfile("OGAMUS/Plan/PDDL/domain_open_close.pddl", "OGAMUS/Plan/PDDL/domain.pddl")
    if Configuration.TASK == Configuration.TASK_OGN_ROBOTHOR or Configuration.TASK == Configuration.TASK_OGN_ITHOR:
        shutil.copyfile("OGAMUS/Plan/PDDL/domain_ogn.pddl", "OGAMUS/Plan/PDDL/domain.pddl")

    # Run agent on each scene
    for episode_data in dataset:

        episode = episode_data['episode']
        scene = episode_data['scene']
        goal = episode_data['goal']

        # Create log directories
        Logger.LOG_DIR_PATH = os.path.join(Configuration.RESULTS_DIR, "episode_{}"
                                           .format(len(os.listdir(Configuration.RESULTS_DIR))))
        os.mkdir(Logger.LOG_DIR_PATH)
        Logger.LOG_FILE = open(os.path.join(Logger.LOG_DIR_PATH, "log.txt"), "w")

        # Randomly generate a goal for the scene
        PddlParser.set_goal(goal)

        if episode in list(range(starting_episode, 5000)):

            init_position = episode_data['agent_position']
            init_rotation = episode_data['initial_orientation']
            init_horizon = episode_data['initial_horizon']


            if Configuration.TASK == Configuration.TASK_OGN_ROBOTHOR:
                if Configuration.ROTATION_STEP != 30:
                    Logger.write("Warning: setting rotation step to 30 degrees according to Object Goal Navigation standard setting.")
                    Configuration.ROTATION_STEP = 30
                if Configuration.MAX_CAM_ANGLE != 30:
                    Logger.write("Warning: setting maximum camera inclination to 30 degrees since 'Locobot' cannot "
                                 "look up or down more than 30 degrees.")
                    Configuration.MAX_CAM_ANGLE = 30
                if Configuration.FOV != 79:
                    Logger.write("Warning: setting field of view to 79 degrees according to Object Goal Navigation standard setting.")
                    Configuration.FOV = 79
                if Configuration.VISIBILITY_DISTANCE != 1:
                    Logger.write("Warning: setting visibility distance to 1 meter according to Object Goal Navigation standard setting.")
                    Configuration.VISIBILITY_DISTANCE = 1
                if Configuration.MAX_DISTANCE_MANIPULATION != 113:
                    Logger.write("Warning: setting max manipulation distance to 113 centimeters for solving the Object Goal "
                                 "Navigation task. In this way the agent is more robust to error in object position approximations.")
                    Configuration.MAX_DISTANCE_MANIPULATION = 113
                if Configuration.CLOSE_TO_OBJ_DISTANCE > 1:
                    Logger.write("Warning: setting distance threshold of predicate 'close_to(object)' to 100 centimeters "
                                 "for solving the Object Goal Navigation task. "
                                 "In this way the agent is more robust to error in object position approximations.")
                    Configuration.CLOSE_TO_OBJ_DISTANCE = 1
                if not Configuration.STOCASTIC_AGENT:
                    Logger.write("Warning: setting agent flag 'stocastic' as True, according to RoboTHOR object navigation challenge.")
                    Configuration.STOCASTIC_AGENT = True
                if not Configuration.DIAGONAL_MOVE:
                    Logger.write("Warning: setting diagonal movements possible in the path planner since running on the "
                                 "RoboTHOR object navigation challenge.")
                    Configuration.DIAGONAL_MOVE = True
                if not Configuration.MAX_ITER != 500:
                    Logger.write("Warning: setting maximum number of steps to 500 according to the "
                                 "RoboTHOR object navigation challenge.")
                    Configuration.MAX_ITER = 500

            if Configuration.TASK == Configuration.TASK_OGN_ITHOR:
                if Configuration.ROTATION_STEP != 45:
                    Logger.write("Warning: setting rotation step to 45 degrees according to Object Goal Navigation dataset in iTHOR.")
                    Configuration.ROTATION_STEP = 45
                if Configuration.MAX_CAM_ANGLE != 60:
                    Logger.write("Warning: setting maximum camera inclination to 30 degrees since 'default' robot in "
                                 "iTHOR cannot look up or down more than 60 degrees.")
                    Configuration.MAX_CAM_ANGLE = 60
                if Configuration.FOV != 79:
                    Logger.write("Warning: setting field of view to 79 degrees according to Object Goal Navigation standard setting.")
                    Configuration.FOV = 79
                if Configuration.VISIBILITY_DISTANCE != 1:
                    Logger.write("Warning: setting visibility distance to 1 meter according to Object Goal Navigation standard setting.")
                    Configuration.VISIBILITY_DISTANCE = 1
                if Configuration.MAX_DISTANCE_MANIPULATION != 163:
                    Logger.write("Warning: setting max manipulation distance to 163 (>150) centimeters for solving the Object Goal "
                                 "Navigation task in iTHOR. In this way the agent is more robust to error in object position approximations.")
                    Configuration.MAX_DISTANCE_MANIPULATION = 163
                if Configuration.CLOSE_TO_OBJ_DISTANCE != 1.5:
                    Logger.write("Warning: setting distance threshold of predicate 'close_to(object)' to 150 centimeters "
                                 "for solving the Object Goal Navigation task in iTHOR. "
                                 "In this way the agent is more robust to error in object position approximations.")
                    Configuration.CLOSE_TO_OBJ_DISTANCE = 1.5


            if Configuration.ROTATION_STEP > Configuration.FOV:
                Logger.write('Warning: agent rotation step ({}) is lower than its field of view ({}). '
                             'Therefore the agent may loop when trying to look at an object which cannot be seen due to'
                             ' blind spots. '.format(Configuration.ROTATION_STEP, Configuration.FOV))

            # DEBUG
            Logger.write('############# START CONFIGURATION #############\n'
                         'DATASET:{}\n'
                         'EPISODE:{}\n'
                         'SCENE:{}\n'
                         'TASK:{}\n'
                         'RANDOM SEED:{}\n'
                         'GOAL OBJECTS:{}\n'
                         'MAX ITER:{}\n'
                         'VISIBILITY DISTANCE:{}\n'
                         'MOVE STEP:{}\n'
                         'MOVE AND ROTATION RANDOMNESS:{}\n'
                         'ROTATION DEGREES:{}\n'
                         'FIELD OF VIEW:{}\n'
                         'MAX DISTANCE MANIPULATION (belief):{}\n'
                         'IoU THRESHOLD:{}\n'
                         'OBJECT DETECTOR GROUND TRUTH:{}\n'
                         'OPEN CLASSIFIER THRESHOLD:{}\n'
                         'ON CLASSIFIER THRESHOLD:{}\n'
                         'OBJECT DETECTOR:{}\n'
                         'OPEN CLASSIFIER:{}\n'
                         'ON CLASSIFIER:{}\n'
                         '###############################################\n'
                         .format(Configuration.DATASET, episode, scene, Configuration.TASK,
                                 Configuration.RANDOM_SEED, Configuration.GOAL_OBJECTS,
                                 Configuration.MAX_ITER, Configuration.VISIBILITY_DISTANCE,
                                 Configuration.MOVE_STEP, Configuration.TASK == Configuration.TASK_OGN_ROBOTHOR,
                                 Configuration.ROTATION_STEP, Configuration.FOV,
                                 Configuration.MAX_DISTANCE_MANIPULATION, Configuration.IOU_THRSH,
                                 Configuration.GROUND_TRUTH_OBJS, Configuration.OPEN_CLASSIFIER_THRSH,
                                 Configuration.OPEN_CLASSIFIER_THRSH, Configuration.OBJ_DETECTOR_PATH,
                                 Configuration.OPEN_CLASSIFIER_PATH, Configuration.ON_CLASSIFIER_PATH))

            # Necessary to compute SPL for Object Goal Navigation
            shortest_path = None
            if Configuration.TASK == Configuration.TASK_OGN_ROBOTHOR or Configuration.TASK == Configuration.TASK_OGN_ITHOR:
                shortest_path = episode_data['shortest_path']

            Agent(scene=scene, position=init_position, init_rotation=init_rotation, init_horizon=init_horizon,
                  shortest_path=shortest_path).run()

        # Copy problem file in result directory
        shutil.copyfile("OGAMUS/Plan/PDDL/facts.pddl", os.path.join(Logger.LOG_DIR_PATH, "facts_{}.pddl".format(scene)))


if __name__ == "__main__":
    main()

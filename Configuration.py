# Set random seed for exploration
RANDOM_SEED = 1

# Set which task to perform
TASK_ON = 'on'
TASK_OPEN = 'open'
TASK_CLOSE = 'close'
TASK_OGN_ROBOTHOR = 'ogn'  # Object goal navigation in RoboTHOR
TASK_OGN_ITHOR = 'ogn_ithor'  # Object goal navigation in iTHOR
TASK = TASK_OGN_ITHOR

DATASET = 'test_set_{}'.format(TASK)

# IP address of WSL2 in Windows
USING_WSL2_WINDOWS = False
IP_ADDRESS = "172.20.48.1"  # Set this for WSL2 by looking into /etc/resolv.conf


##########################################################
############### AGENT BELIEF CONFIGURATION ###############
##########################################################
TRUST_PDDL = True  # If set to true, apply action effects regardless of observation after action execution
GROUND_TRUTH_OBJS = False  # Use ground truth objects detection
STOCASTIC_AGENT = False  # An agent is stochastic if its movements and rotations are affected by actuation noise


##########################################################
################### RUN CONFIGURATION ####################
##########################################################
MAX_ITER = 200


##########################################################
############## ITHOR SIMULATOR CONFIGURATION #############
##########################################################
RENDER_DEPTH_IMG = 1
HIDE_PICKED_OBJECTS = 1
FRAME_WIDTH = 224
FRAME_HEIGHT = 224
FOV = 79
VISIBILITY_DISTANCE = 1.5  # 150 centimeters
MOVE_STEP = 0.25  # 25 centimeters
ROTATION_STEP = 30  # degrees
MAX_CAM_ANGLE = 60  # maximum degrees of camera when executing LookUp and LookDown actions


##########################################################
################## LOGGER CONFIGURATION ##################
##########################################################

# Print output information
VERBOSE = 1

# Save images
PRINT_IMAGES = 1

# Save agent camera view images
PRINT_CAMERA_VIEW_IMAGES = 0 and PRINT_IMAGES

# Save agent camera depth view images
PRINT_CAMERA_DEPTH_VIEW_IMAGES = 0 and PRINT_IMAGES

# Save top view images
PRINT_TOP_VIEW_IMAGES = 0 and PRINT_IMAGES

# Save top view images
PRINT_TOP_VIEW_GRID_PLAN_IMAGES = 0 and PRINT_IMAGES

# Save object predictions
PRINT_OBJS_PREDICTIONS = 0 and PRINT_IMAGES


##########################################################
################ MAP MODEL CONFIGURATION #################
##########################################################

# x min coordinate in centimeters
MAP_X_MIN = -10000 * MOVE_STEP
# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


# y min coordinate in centimeters
MAP_Y_MIN = -10000 * MOVE_STEP

# x max coordinate in centimeters
MAP_X_MAX = 10000 * MOVE_STEP

# y max coordinate in centimeters
MAP_Y_MAX = 10000 * MOVE_STEP

# x and y centimeters per pixel in the resized grid occupancy map
MAP_GRID_DX = MOVE_STEP*100
MAP_GRID_DY = MOVE_STEP*100

CAMERA_HEIGHT = None  # This is set runtime depending on the agent mode ('default' for ithor or 'locobot' for robothor)


##########################################################
############# OBJECT DETECTOR CONFIGURATION ##############
##########################################################
OBJ_SCORE_THRSH = 0.3  # Objects with a lower score than the threshold are discarded
IOU_THRSH = 0.8  # If the IoU between two objects is lower than the IoU threshold, the lower score object is discarded
IRRELEVANT_CLASSES = ['floor', 'wall', 'roomdecor']
OBJ_COUNT_THRSH = 2  # minimum number of an object observation to consider it a really existing object

##########################################################
############### PATH PLANNER CONFIGURATION ###############
##########################################################
# Distance from which an object can be manipulated, i.e., all goal positions in path planner are the ones within
# the max distance manipulation from a goal object position.
MAX_DISTANCE_MANIPULATION = 140  # centimeters
MAX_USELESS_GOAL_CELLS = int(150 / MAP_GRID_DX)  # Useless goal cells cover 1.5 meters

DIAGONAL_MOVE = False  # Automatically set to True for object goal navigation


##########################################################
################# PREDICATE CLASSIFIERS ##################
##########################################################
OBJ_DETECTOR_PATH = "Utils/pretrained_models/faster-rcnn_118classes.pkl"
OPEN_CLASSIFIER_PATH = "Utils/pretrained_models/open_predictor.pth"
ON_CLASSIFIER_PATH = "Utils/pretrained_models/on_predictor.pth"
OBJ_CLASSES_PATH = "Utils/pretrained_models/obj_classes_coco.pkl"



##########################################################
################ EVALUATION CONFIGURATION ################
##########################################################
# Whenever a detected object has an IoU higher than the threshold with the ground-truth bbox of an object of the same
# type, then the two objects matches for evaluation purposes.
OBJ_IOU_MATCH_THRSH = 0.5


##########################################################
########## PREDICATE CLASSIFIERS CONFIGURATION ###########
##########################################################
CLOSE_TO_OBJ_DISTANCE = 1.4  # Used to classify the "close_to(object)" predicate
OPEN_CLASSIFIER_THRSH = 0.5  # Used to classify the "open(object)" and "on(object1, object2)" predicates


##########################################################
############### PDDL PLANNER CONFIGURATION ###############
##########################################################
FF_PLANNER = "FF"
PLANNER_TIMELIMIT = 300
PLANNER = FF_PLANNER
PDDL_PROBLEM_PATH = "OGAMUS/Plan/PDDL/facts.pddl"


##########################################################
############## EVENT PLANNER CONFIGURATION ###############
##########################################################
ROTATE_RIGHT = "right"
ROTATE_LEFT = "left"
ROTATE_DIRECTION = ROTATE_LEFT

##########################################################
############ INPUT INFORMATION CONFIGURATION #############
##########################################################

PICKUPABLE_OBJS = ["alarmclock", "aluminumfoil", "apple", "baseballbat", "book", "boots", "basketball",
                   "bottle", "bowl", "box", "bread", "butterknife", "candle", "cd", "cellphone", "peppershaker",
                   "cloth", "creditcard", "cup", "dishsponge", "dumbbell", "egg", "fork", "handtowel",
                   "kettle", "keychain", "knife", "ladle", "laptop", "lettuce", "mug", "newspaper",
                   "pan", "papertowel", "papertowelroll", "pen", "pencil", "papershaker", "pillow", "plate", "plunger",
                   "pot", "potato", "remotecontrol", "saltshaker", "scrubbrush", "soapbar", "soapbottle",
                   "spatula", "spoon", "spraybottle", "statue", "tabletopdecor", "teddybear", "tennisracket",
                   "tissuebox", "toiletpaper", "tomato", "towel", "vase", "watch", "wateringcan", "winebottle"]

RECEPTACLE_OBJS = ["armchair", "bathtub", "bathtubbasin", "bed", "bowl", "box", "cabinet", "coffeemachine",
                   "coffeetable", "countertop", "desk", "diningtable", "drawer", "fridge",  # "dresser", does not exist in floorplan229
                   "garbagecan", "handtowelholder", "laundryhamper", "microwave", "mug", "ottoman", "pan", #  "cup", is too little to generate feasible goals
                   "plate", "pot", "safe", "shelf", "sidetable", "sinkbasin", "sofa", "toaster", # "sink", is not used since there is sinkbasin
                   "toilet", "toiletpaperhanger", "towelholder", "tvstand", "stoveburner"]

OPENABLE_OBJS = ["blinds", "book", "box", "cabinet", "drawer", "fridge", "kettle", "laptop", "microwave",
                 "safe", "showercurtain", "showerdoor", "toilet"]

NOT_CONTAINED_OBJS = ["diningtable", "armchair", "bathtub", "bathtubbasin", "cabinet", "dresser", "fridge",
                      "handtowelholder", "safe", "showercurtain", "showerdoor", "toilet", "sidetable",
                      'countertop', 'floor', 'wall']


##########################################################
################## OTHER CONFIGURATION ###################
##########################################################
RESULTS_DIR = 'Results/{}_steps{}'.format(DATASET, MAX_ITER)
DATASET_DIR = 'Datasets'

# This is set runtime
GOAL_OBJECTS = []

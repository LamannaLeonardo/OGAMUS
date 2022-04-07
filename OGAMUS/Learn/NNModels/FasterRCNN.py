# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import os
import pickle

import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

import Configuration
from Utils import Logger
from Utils.torchvision_utils import draw_bounding_boxes

import numpy as np
import matplotlib
from matplotlib.cm import cmaps_listed
color_palette = matplotlib.cm.get_cmap('viridis', 119).colors

class FasterRCNN:

    def __init__(self):

        # Use the GPU or the CPU, if a GPU is not available
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Set the number of classes, 118 categories plus background class (with label 0)
        num_classes = 119

        # Load an instance segmentation model pre-trained on COCO
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # Get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # Replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Load pretrained model on custom dataset, if exists
        if os.path.exists(Configuration.OBJ_DETECTOR_PATH):
            if not torch.cuda.is_available():
                self.model.load_state_dict(torch.load(Configuration.OBJ_DETECTOR_PATH,
                                                      map_location=torch.device('cpu')))
            else:
                self.model.load_state_dict(torch.load(Configuration.OBJ_DETECTOR_PATH))

        # move model to the right device
        self.model.to(self.device)


        self.pil_to_tensor = transforms.Compose([
            transforms.Resize((224, 224)),  # (224, 224) is the size of all images in the custom training set
            transforms.ToTensor()
        ])

        self.categories = ['Background'] + pickle.load(open("Utils/pretrained_models/obj_classes_coco.pkl", "rb"))


    def resize_boxes(self, boxes, original_size, new_size):
        ratios = [
            torch.tensor(s, dtype=torch.float32, device=boxes.device) /
            torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)

        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)


    def predict(self, rgb_img):

        # Set model in evaluation mode
        self.model.eval()

        # Predict objects in the image
        rgb_img = Image.fromarray(rgb_img, mode="RGB")
        if rgb_img.width != 224 or rgb_img.height != 224:
            rgb_img_resized = rgb_img.resize((224, 224))
        else:
            rgb_img_resized = copy.deepcopy(rgb_img)

        pred = self.model(self.pil_to_tensor(rgb_img_resized).unsqueeze(0).to(self.device))[0]

        # Resize boxes to fit input image size
        rgb_img_tensor = torch.from_numpy(np.array(rgb_img).transpose((2,0,1))) # move channels to first tensor axis
        bbox = self.resize_boxes(pred['boxes'], (224, 224), rgb_img_tensor.shape[-2:])
        pred['boxes'] = bbox

        # Discard predictions with low scores
        idx = [i for i, score in enumerate(pred['scores']) if score > Configuration.OBJ_SCORE_THRSH]
        pred['boxes'] = pred['boxes'][idx]
        pred['labels'] = pred['labels'][idx]
        pred['scores'] = pred['scores'][idx]

        # Parse labels id to category names
        if self.device == torch.device('cpu'):
            color_labels = [color_palette[int(l)] for l in np.array(pred['labels'])]
            pred['labels'] = [self.categories[int(el)] for el in np.array(pred['labels'])]
        else:
            color_labels = [color_palette[int(l)] for l in pred['labels'].cpu().numpy()]
            pred['labels'] = [self.categories[int(el)] for el in pred['labels'].cpu().numpy()]

        color_labels = [matplotlib.colors.rgb2hex(color_labels[i]) for i in range(len(color_labels))]

        # DEBUG
        if Configuration.PRINT_OBJS_PREDICTIONS:
            goal_objs = Configuration.GOAL_OBJECTS
            print_pred_labels = [el for el in pred['labels'] if el.lower().strip() in goal_objs]
            printed_pred_boxes = [pred['boxes'][i].detach().numpy() for i in range(len(pred['boxes']))
                                  if pred['labels'][i].lower().strip() in goal_objs]

            pred_img = transforms.ToPILImage()(draw_bounding_boxes(rgb_img_tensor, torch.FloatTensor(printed_pred_boxes), colors=color_labels, labels=print_pred_labels))
            prev_preds = [img for img in os.listdir(Logger.LOG_DIR_PATH) if img.startswith("pred_")]
            pred_img.save('{}/pred_{}.jpg'.format(Logger.LOG_DIR_PATH, len(prev_preds)),"JPEG")

        # Return predicted bboxes with labels and scores
        pred['boxes'] = pred['boxes'].cpu().detach().numpy()
        pred['labels'] = np.array(pred['labels'])
        pred['scores'] = pred['scores'].cpu().detach().numpy()
        return pred









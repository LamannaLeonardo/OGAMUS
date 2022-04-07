# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import errno
import os
import torch
from PIL import Image
from torchvision import transforms

import Configuration
from OGAMUS.Learn.NNModels.OpenPredicateClassifierNN import OpenPredicateClassifierNN


class OpenPredicateClassifier:

    def __init__(self, input_model_path=None):

        # Initialize neural network model
        self.model = OpenPredicateClassifierNN()

        # Load input model weights
        if input_model_path is not None:

            # Load pretrained model on custom dataset, if exists
            if os.path.exists(input_model_path):

                if not torch.cuda.is_available():
                    self.model.load_state_dict(torch.load(input_model_path, map_location=torch.device('cpu')))
                else:
                    self.model.load_state_dict(torch.load(input_model_path))

            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), input_model_path)

        # Set input data transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # These are RGB mean+std values
                std=[0.229, 0.224, 0.225])   # across a large photo dataset.
        ])


    def predict(self, rgb_img):

        # Set model in evaluation mode
        self.model.eval()

        # Normalize input image
        rgb_img = self.transform(Image.fromarray(rgb_img))

        # Create single sample batch
        rgb_img = rgb_img.unsqueeze(0)

        # Predict
        y_pred = self.model(rgb_img)
        # y_pred_tag = torch.round(torch.sigmoid(y_pred))
        y_pred_tag_sigm = torch.sigmoid(y_pred)
        y_pred_tag = torch.where(y_pred_tag_sigm >= Configuration.OPEN_CLASSIFIER_THRSH, 1, 0)
        y_pred_tag = y_pred_tag.flatten().cpu().detach().numpy().astype(int)

        return y_pred_tag[0]








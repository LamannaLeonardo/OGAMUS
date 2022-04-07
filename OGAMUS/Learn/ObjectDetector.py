# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import Configuration
from OGAMUS.Learn.NNModels.FasterRCNN import FasterRCNN


class ObjectDetector:


    def __init__(self):

        # Use Faster RCNN object detector pretrained on MSCOCO and finetuned on ITHOR custom dataset with 118 classes
        self.model = FasterRCNN()


    def get_visible_objects(self, rgb_img):

        # Get visible objects
        visible_objs = self.model.predict(rgb_img)

        # Filter visible objects according to relevant classes (e.g. floor is not a relevant class)
        removed_objs = []
        for i in range(len(visible_objs['labels'])):
            if visible_objs['labels'][i].lower() in Configuration.IRRELEVANT_CLASSES:
                removed_objs.append(i)

        visible_objs['boxes'] = [visible_objs['boxes'][i] for i in range(len(visible_objs['boxes']))
                                 if i not in removed_objs]
        visible_objs['labels'] = [visible_objs['labels'][i] for i in range(len(visible_objs['labels']))
                                 if i not in removed_objs]
        visible_objs['scores'] = [visible_objs['scores'][i] for i in range(len(visible_objs['scores']))
                                 if i not in removed_objs]

        # Filter visible objects with high IoU over bboxes
        removed_objs = []
        for i in range(len(visible_objs['labels']) - 1):
            if i not in removed_objs:

                for j in range(i+1, len(visible_objs['labels'])):
                    if j not in removed_objs:

                        bbox_i = {'x1': visible_objs['boxes'][i][0], 'y1': visible_objs['boxes'][i][1],
                                  'x2': visible_objs['boxes'][i][2], 'y2': visible_objs['boxes'][i][3]}
                        bbox_j = {'x1': visible_objs['boxes'][j][0], 'y1': visible_objs['boxes'][j][1],
                                  'x2': visible_objs['boxes'][j][2], 'y2': visible_objs['boxes'][j][3]}

                        iou = self.get_iou(bbox_i, bbox_j)

                        if iou > Configuration.IOU_THRSH:
                            if visible_objs['scores'][i] > visible_objs['scores'][j]:
                                removed_objs.append(j)
                            else:
                                removed_objs.append(i)

        visible_objs['boxes'] = [visible_objs['boxes'][i] for i in range(len(visible_objs['boxes']))
                                 if i not in removed_objs]
        visible_objs['labels'] = [visible_objs['labels'][i] for i in range(len(visible_objs['labels']))
                                 if i not in removed_objs]
        visible_objs['scores'] = [visible_objs['scores'][i] for i in range(len(visible_objs['scores']))
                                 if i not in removed_objs]

        return visible_objs


    def get_visible_objects_ground_truth(self, event):

        visible_objects = dict()

        visible_objs_instances = []
        [visible_objs_instances.append(obj['objectId'])
         for obj in event.metadata['objects']
         if obj['objectId'] in event.instance_detections2D
         and obj['objectId'] not in visible_objs_instances
         and obj['objectType'].lower() not in Configuration.IRRELEVANT_CLASSES]

        all_boxes = []
        all_labels = []
        all_scores = []

        for obj in visible_objs_instances:
            object = [o for o in event.metadata['objects'] if o['objectId'] == obj][0]

            bbox = event.instance_detections2D[object['objectId']]  # xmin, ymin, xmax, ymax
            if bbox[2] - bbox[0] > 1 and bbox[3] - bbox[1] > 1:
                all_boxes.append(bbox)
                all_labels.append(object['objectType'].lower())
                all_scores.append(1)


        visible_objects['boxes'] = all_boxes
        visible_objects['labels'] = all_labels
        visible_objects['scores'] = all_scores

        return dict(visible_objects)


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



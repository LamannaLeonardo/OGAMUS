# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import os

import matplotlib.pyplot as plt
from collections import Counter

DIR = 'Results/example_test'

def plotCurve(X, Y, xlabel, ylabel, title, file_name):

    plt.plot(X, Y)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(X)
    plt.title(title)
    plt.savefig(file_name)
    plt.close()



def plotBar(names, values, xlabel, ylabel, title, file_name):

    plt.bar(names, values)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(file_name)
    plt.close()


def ogn_metrics():

    all_goal_success = []
    all_spl = []
    all_distance_to_success = []

    for episode_dir in sorted([el for el in os.listdir('../{}'.format(DIR)) if '.' not in el],
                              key=lambda x: int(x.split('_')[1])):
        episode_metrics = json.load(open(os.path.join('../', DIR, episode_dir, 'metrics.json'), 'r'))

        all_goal_success.append(episode_metrics['goal']['success'])
        all_spl.append(episode_metrics['goal']['spl'])
        all_distance_to_success.append(episode_metrics['goal']['distance_to_success'])

    with open('../ogn_metrics'.format(DIR), "w") as f:
        f.write('Total episodes: {}'.format(len(all_goal_success)))
        f.write('\nAverage success: {}'.format(len([el for el in all_goal_success if el == 1])/len(all_goal_success)))
        f.write('\nAverage spl: {}'.format(sum(all_spl)/len(all_spl)))
        f.write('\nAverage distance to success: {}'.format(sum(all_distance_to_success)/len(all_distance_to_success)))


def generate_plots():

    all_goal_success = []
    all_distance_to_success = []
    all_predicates_precision = []
    all_predicates_recall = []
    all_objects_precision = []
    all_objects_recall = []
    all_global_predicates_precision = []
    all_global_predicates_recall = []
    all_fp_predicates = dict()
    all_fn_predicates = dict()

    for episode_dir in sorted([el for el in os.listdir('../{}'.format(DIR)) if '.' not in el],
                              key=lambda x: int(x.split('_')[1])):
        episode_metrics = json.load(open(os.path.join('../', DIR, episode_dir, 'metrics.json'), 'r'))

        all_goal_success.append(episode_metrics['goal']['success'])
        all_distance_to_success.append(episode_metrics['goal']['distance_to_success'])
        all_objects_recall.append(episode_metrics['objects']['recall'])
        all_objects_precision.append(episode_metrics['objects']['precision'])
        all_predicates_recall.append(episode_metrics['predicates']['recall'])
        all_predicates_precision.append(episode_metrics['predicates']['precision'])
        all_global_predicates_recall.append(episode_metrics['global']['recall'])
        all_global_predicates_precision.append(episode_metrics['global']['precision'])

        fp_predicates = [pred.split('(')[0] for pred in episode_metrics['predicates']['belief']
                         if pred not in episode_metrics['predicates']['gt']]

        fn_predicates = [pred.split('(')[0] for pred in episode_metrics['predicates']['gt']
                         if pred not in episode_metrics['predicates']['belief']]

        fp_predicates = dict(Counter(fp_predicates))
        fn_predicates = dict(Counter(fn_predicates))

        all_fp_predicates = {k: all_fp_predicates.get(k, 0) + fp_predicates.get(k, 0) for k in set(all_fp_predicates) | set(fp_predicates)}
        all_fn_predicates = {k: all_fn_predicates.get(k, 0) + fn_predicates.get(k, 0) for k in set(all_fn_predicates) | set(fn_predicates)}


    plotCurve(list(range(len(all_goal_success))), all_goal_success, 'Episodes', 'Goal success',
              'Goal success', '../{}/goal_success.png'.format(DIR))
    plotCurve(list(range(len(all_distance_to_success))), all_distance_to_success, 'Episodes', 'Distance to success',
              'Distance to success', '../{}/distance_to_success.png'.format(DIR))
    plotCurve(list(range(len(all_objects_recall))), all_objects_recall, 'Episodes', 'Recall',
              'Objects recall', '../{}/objects_recall.png'.format(DIR))
    plotCurve(list(range(len(all_objects_precision))), all_objects_precision, 'Episodes', 'Precision',
              'Objects precision', '../{}/objects_precision.png'.format(DIR))
    plotCurve(list(range(len(all_predicates_recall))), all_predicates_recall, 'Episodes', 'Recall',
              'Predicates recall (only real objects)', '../{}/predicates_recall.png'.format(DIR))
    plotCurve(list(range(len(all_predicates_precision))), all_predicates_precision, 'Episodes', 'Precision',
              'Predicates precision (only real objects)', '../{}/predicates_precision.png'.format(DIR))
    plotCurve(list(range(len(all_global_predicates_recall))), all_global_predicates_recall, 'Episodes', 'Recall',
              'Predicates recall (all objects)', '../{}/global_predicates_recall.png'.format(DIR))
    plotCurve(list(range(len(all_global_predicates_precision))), all_global_predicates_precision, 'Episodes', 'Precision',
              'Predicates precision (all objects)', '../{}/global_predicates_precision.png'.format(DIR))

    plotBar(all_fp_predicates.keys(), all_fp_predicates.values(), 'Predicates', 'False positives',
              'False positives (only real objects)', '../{}/predicates_fp.png'.format(DIR))

    plotBar(all_fn_predicates.keys(), all_fn_predicates.values(), 'Predicates', 'False negatives',
              'False negatives (only real objects)', '../{}/predicates_fn.png'.format(DIR))


    with open('../{}/metrics'.format(DIR), "w") as f:
        f.write('Average success: {}'.format(len([el for el in all_goal_success if el == 1])/len(all_goal_success)))
        f.write('\nAverage distance to success: {}'.format(sum(all_distance_to_success)/len(all_distance_to_success)))
        f.write('\nAverage object precision: {}'.format(sum(all_objects_precision)/len(all_objects_precision)))
        f.write('\nAverage object recall: {}'.format(sum(all_objects_recall)/len(all_objects_recall)))
        f.write('\nAverage predicate precision (on real objects): {}'.format(sum(all_predicates_precision)/len(all_predicates_precision)))
        f.write('\nAverage predicate recall (on real objects): {}'.format(sum(all_predicates_recall)/len(all_predicates_recall)))
        f.write('\nAverage predicate precision (on all objects): {}'.format(sum(all_global_predicates_precision)/len(all_global_predicates_precision)))
        f.write('\nAverage predicate recall (on all objects): {}'.format(sum(all_global_predicates_recall)/len(all_global_predicates_recall)))


if __name__ == "__main__":

    generate_plots()

    # ogn_metrics()

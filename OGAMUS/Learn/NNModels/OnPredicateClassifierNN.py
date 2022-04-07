# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


from torch import nn


class OnPredicateClassifierNN(nn.Module):

    def __init__(self):
        super(OnPredicateClassifierNN, self).__init__()
        self.fc1 = nn.Linear(244, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x

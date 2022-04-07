# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


from collections import defaultdict

class AbstractModel:

    def __init__(self):
        self.states = []
        self.transitions = defaultdict(list)

    def add_transition(self, state_src, action, state_dest):
        self.transitions[state_src.id, action] = state_dest.id

    def add_state(self, state_new):
        self.states.append(state_new)


# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.



class State:

    def __init__(self, state_id, perceptions, visible_objects):
        self.id = state_id
        self.perceptions = perceptions
        self.visible_objects = visible_objects

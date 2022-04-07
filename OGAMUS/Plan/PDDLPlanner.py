# Copyright (c) 2022, Leonardo Lamanna
# All rights reserved.
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


import datetime
import re
import subprocess

from Utils import Logger


class PDDLPlanner:

    def __init__(self):
        pass

    def pddl_plan(self):
        pddl_plan = self.FF()

        # Add final stop action
        if pddl_plan is not None:
            pddl_plan.append("STOP()")

        return pddl_plan


    def FF(self):
        """
        Compute the plan using FastForward planner, the pddl file (i.e. "domain.pddl" and "facts.pddl") must be contained
        into the "PDDL" folder.
        :return: a plan
        """

        # DEBUG
        start = datetime.datetime.now()

        bash_command = "./OGAMUS/Plan/PDDL/Planners/FF/ff -o OGAMUS/Plan/PDDL/domain.pddl -f OGAMUS/Plan/PDDL/facts.pddl"

        process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

        output, error = process.communicate()

        begin_index = -1
        end_index = -1
        result = str(output).split("\\n")

        for el in result:
            if el.__contains__("No plan will solve it"):
                return None
            elif el.__contains__("unknown or empty type"):
                return None
            elif el.__contains__("unknown constant"):
                Logger.write('WARNING: unknown constant in pddl problem file. Cannot compute any plan.')
                return None

        for i in range(len(result)):
            if not result[i].find("step"):
                begin_index = i

            elif not result[i].find("time"):
                end_index = i-2

        plan = [result[i].split(":")[1].replace("\\r", "") for i in range(begin_index, end_index)
                if result[i].split(":")[1].replace("\\r", "").lower().strip() != 'reach-goal']
        syntax_plan = []

        if len(plan) == 0:
            return syntax_plan

        # DEBUG
        end = datetime.datetime.now()
        # print("FF computational time: {}".format(end-start))

        for el in plan:
            tmp = el
            tmp = re.sub("[ ]", ",", tmp.strip())
            tmp = tmp.replace(",", "(", 1)
            tmp = tmp + ")"
            tmp = tmp[:tmp.index('(')].replace("-", "_") + tmp[tmp.index('('):]
            syntax_plan.append(tmp)

        return syntax_plan

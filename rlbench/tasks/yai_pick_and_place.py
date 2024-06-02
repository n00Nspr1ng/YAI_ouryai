from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, ConditionSet, \
    GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary, BoundingBox
from rlbench.const import colors


class YaiPickAndPlace(Task):

    def init_task(self) -> None:
        self.target_blocks = [Shape('y_block'), Shape('a_block'), Shape('i_block')]
        self.success_detectors = [ProximitySensor('y_success0'), ProximitySensor('y_success1'), ProximitySensor('y_success2')]
        self.success_detectors += [ProximitySensor('a_success0'), ProximitySensor('a_success1'), ProximitySensor('a_success2')]
        self.success_detectors += [ProximitySensor('i_success0'), ProximitySensor('i_success1'), ProximitySensor('i_success2')]

        cond_sets = []
        for block in self.target_blocks:
            cond_set = ConditionSet([
                DetectedCondition(block, self.success_detectors[self.target_blocks.index(block) * 3]),
                DetectedCondition(block, self.success_detectors[self.target_blocks.index(block) * 3 + 1]),
                DetectedCondition(block, self.success_detectors[self.target_blocks.index(block) * 3 + 2]),
            ])
            cond_sets.append(cond_set)
        self.register_success_conditions(cond_sets)

        self.spawn_boundary = SpawnBoundary([Shape('workspace')])
        self.yai_plane = Shape('yai_boundary')


    def init_episode(self, index: int) -> List[str]:
       

        self.spawn_boundary.clear()
        self.spawn_boundary.sample(self.yai_plane)


        for block in self.target_blocks:
            self.spawn_boundary.sample(block, min_distance=0.22, min_rotation=(-3.14, 0.0, 0.0), max_rotation=(3.14, 0.0, 0.0))

        # for block in [self.target_block] + self.distractors:
        #     self.boundary.sample(block, min_distance=0.1)

        # return ['pick up the %s block and lift it up to the target' %
        #         block_color_name,
        #         'grasp the %s block to the target' % block_color_name,
        #         'lift the %s block up to the target' % block_color_name]
    
        return ['pick up y, a, i block and place it on the target',]

    def variation_count(self) -> int:
        # return len(colors)
        return 1 # no variation 
    

    # def get_low_dim_state(self) -> np.ndarray:
    #     # One of the few tasks that have a custom low_dim_state function.
    #     return np.concatenate([self.target_block.get_position(), self.success_detector.get_position()], 0)

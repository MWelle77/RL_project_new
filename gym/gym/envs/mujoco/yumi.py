import os
import datetime

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


def body_index(model, body_name):
    return model.body_names.index(body_name)


def body_pos(model, body_name):
    ind = body_index(model, body_name)
    return model.body_pos[ind]

def body_quat(model,body_name):
    ind = body_index(model, body_name)
    return model.body_quat[ind]


class YumiEnvSimple(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.high = np.array([40, 35, 30, 20, 15, 10, 10])
        self.low = -self.high
        self.wt = 0.0
        self.we = 0.0
        root_dir = os.path.dirname(__file__)
        xml_path = os.path.join(root_dir, 'yumi', 'yumi.xml')
        mujoco_env.MujocoEnv.__init__(self, xml_path, 1)
        utils.EzPickle.__init__(self)

        # Manually define this to let a be in [-1, 1]^d
        self.action_space = spaces.Box(low=-np.ones(7) * 2, high=np.ones(7) * 2, dtype=np.float32)
        self.init_params()

    def init_params(self, wt=0.9, x=0.0, y=0.0, z=0.2):
        """
        :param wt: Float in range (0, 1), weight on euclidean loss
        :param x, y, z: Position of goal
        """
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

    def step(self, a):
        a_real = a * self.high / 2
        self.do_simulation(a_real, self.frame_skip)
        reward = self._reward(a_real)
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

    def _reward(self, a):
        arm = np.concatenate([body_pos(self.model, 'gripper_r_finger_l'),body_quat(self.model, 'gripper_r_finger_l')])
        goal =np.concatenate([body_pos(self.model, 'goal'),body_quat(self.model, 'goal')])
        arm2goal = np.linalg.norm(arm - goal)
        return - np.linalg.norm(a)*1000
        #return -arm2goal*100*0 - np.linalg.norm(a)*1000

    def _get_obs(self):
        return np.concatenate([
             self.sim.data.qpos.flat[:7],
             body_pos(self.model, 'gripper_r_finger_l'),
             body_pos(self.model, 'goal')
         ])

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        qpos[9] =  1.0
        #set inital position        
        qpos[0] = 0.4097769755571825 #j_r1

        #qpos[0] = -1.0 #0.4097769755571825 #j_r1

        qpos[1] = -0.8696447789194206 #j_r2
        qpos[2] = -1.3862958447109897 #j_r7
        qpos[3] = 0.7371426330845399 #j_r3
        qpos[4] = 0.38664608272307444 #j_r4
        qpos[5] = 1.2331409655322743 #j_r5
        qpos[6] = 0.27708597056285544 #j_r6

        self.set_state(qpos, qvel*0)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = 2.0
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 180
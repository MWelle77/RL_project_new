import os

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env



class YumiEnvSimple1(mujoco_env.MujocoEnv, utils.EzPickle):
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

    def init_params(self):
        """
        :param wt: Float in range (0, 1), weight on euclidean loss
        """
        self.wt = 0.9
        self.we = 1 - self.wt
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

    def step(self, a):
        a_real = a #* self.high / 2
        self.do_simulation(a_real, self.frame_skip)
        reward = self._reward(a_real)
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

    def _reward(self, a):
        eef = self.get_body_com('gripper_r_finger_l')
        goal = self.get_body_com('goal')
        goal_distance = np.linalg.norm(eef - goal)
        tofar=0
        #if(goal_distance>0.3):
        #    tofar=10000
        # This is the norm of the joint angles
        # The ** 4 is to create a "flat" region around [0, 0, 0, ...]
        q_norm = 0#np.linalg.norm(self.sim.data.qpos.flat[:7]) ** 4 / 100.0

        # TODO in the future
        # f_desired = np.eye(3)
        # f_current = body_frame(self, 'gripper_r_base')

        reward = -(
            self.wt * goal_distance ** 2 *10 +  # Scalars here is to make this part of the reward approx. [0, 1]
            self.we * np.linalg.norm(a) / 40 +
            q_norm+tofar
        )
        return reward

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:7],
            np.clip(self.sim.data.qvel.flat[:7], -10, 10),
            self.get_body_com('gripper_r_finger_l'),
            self.get_body_com('goal')
        ])

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
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
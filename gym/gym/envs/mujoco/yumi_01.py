import os

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


def body_index(model, body_name):
    return model.body_names.index(body_name)


# def body_pos(model, body_name):
#     ind = body_index(model, body_name)
#     return model.body_pos[ind]


# def body_quat(model, body_name):
#     ind = body_index(model, body_name)
#     return model.body_quat[ind]


# def body_frame(env, body_name):
#     """
#     Returns the rotation matrix to convert to the frame of the named body
#     """
#     ind = body_index(env.model, body_name)
#     b = env.data.body_xpos[ind]
#     q = env.data.body_xquat[ind]
#     qr, qi, qj, qk = q
#     s = np.square(q).sum()
#     R = np.array([
#         [1 - 2 * s * (qj ** 2 + qk ** 2), 2 * s * (qi * qj - qk * qr), 2 * s * (qi * qk + qj * qr)],
#         [2 * s * (qi * qj + qk * qr), 1 - 2 * s * (qi ** 2 + qk ** 2), 2 * s * (qj * qk - qi * qr)],
#         [2 * s * (qi * qk - qj * qr), 2 * s * (qj * qk + qi * qr), 1 - 2 * s * (qi ** 2 + qj ** 2)]
#     ])
#     return R


class YumiEnvSimple1(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        #set up evil force
        self._adv_bindex = 0
        self.adv_max_force = 10.0
        self.randf =np.random.uniform(low=-self.adv_max_force, high=self.adv_max_force, size=(6,))*0
        self.time =0

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


        qpos = self.init_qpos
        self._adv_bindex = body_index(self.model,'gripper_r_finger_l')
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

    #evil force
    def _adv_to_xfrc(self, adv_act):
        self.sim.data.xfrc_applied[self._adv_bindex]=np.array([adv_act[0], adv_act[1], adv_act[2], adv_act[3], adv_act[4], adv_act[5]])


    def step(self, a):
        self.time+=self.dt
        # Sine forces
        # =====================================================
        sinusoidal_x = self.adv_max_force * np.sin(5 * np.pi * self.time) + np.random.normal(0, 1, 1)
        sinusoidal_y = 0 + np.random.normal(0, 1, 1)
        sine_forces = np.zeros_like(self.randf)
        sine_forces[:2] = sinusoidal_x, sinusoidal_y
        self.randf = sine_forces
        self._adv_to_xfrc(self.randf)

        # # sawthooth forces
        # # =====================================================
        # sawthooth_x = self.adv_max_force * signal.sawtooth(5 * np.pi * self.time) + np.random.normal(0, 1, 1)
        # sawthooth_y = 0+ np.random.normal(0, 1, 1)
        # sawthooth_forces =np.zeros_like(self.randf)
        # sawthooth_forces[:2] = sawthooth_x, sawthooth_y   
        # self.randf = sawthooth_forces
        # self._adv_to_xfrc(self.randf)


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
        eef_quat = self.data.get_body_xquat('gripper_r_finger_l')
        goal_quat= self.data.get_body_xquat('goal')
        #https://math.stackexchange.com/questions/90081/quaternion-distance
        quat_distance = np.arccos(2*np.inner(eef_quat,goal_quat)**2-1)
        #print(quat_distance)
        wd=2.2
        wq=0.25
        wa=0.05/40
        wjp=0.0001
        wjv=0.02

        # print("R: = " + str(wd * goal_distance) + " , " + str(wq*quat_distance ) + " , " +
        #     str(wa * np.linalg.norm(a)) + " , " + str(wjp*np.square(self.sim.data.qpos.flat[:7]).sum()) + 
        #     " , " + str(0.02*np.square(self.sim.data.qvel.flat[:7]).sum()))


        reward = -(
            wd * goal_distance  + wq*quat_distance +  # Scalars here is to make this part of the reward approx. [0, 1]
            wa * np.linalg.norm(a)  +
            wjp*np.square(self.sim.data.qpos.flat[:7]).sum() +
            wjv*np.square(self.sim.data.qvel.flat[:7]).sum()
        )
        return reward

    def _get_obs(self):
        #print(str(self.sim.data.qpos[0]) + "," +str(self.sim.data.qpos[1]) + "," +str(self.sim.data.qpos[2]) + ","
        #+ str(self.sim.data.qpos[3]) + ","+str(self.sim.data.qpos[4]) + ","+str(self.sim.data.qpos[5]) + "," +str(self.sim.data.qpos[6]) + ",")
        return np.concatenate([
            self.sim.data.qpos.flat[:7],
            np.clip(self.sim.data.qvel.flat[:7], -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        #set inital position        

        qpos[:7] = 0.66,  -1.53,  -1.39,    0.97,  -0.36,   0.83,  -0.73


        self.set_state(qpos, qvel*0)
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.distance = 2.0
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 180

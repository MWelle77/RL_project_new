import os
import datetime

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from scipy import signal


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

         #set up evil force
        self._adv_bindex = 0
        self.adv_max_force = 0.1
        self.randf =np.random.uniform(low=-self.adv_max_force, high=self.adv_max_force, size=(6,))*0
        self.time =0

        root_dir = os.path.dirname(__file__)
        xml_path = os.path.join(root_dir, 'yumi', 'yumi.xml') 
        mujoco_env.MujocoEnv.__init__(self, xml_path, 1)
        utils.EzPickle.__init__(self)
        self.obs_dim =self._get_obs().size
        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        
        self.init_params()

    def init_params(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self._adv_bindex = body_index(self.model,'gripper_r_finger_l')
        self.set_state(qpos, qvel)

    #evil force
    def _adv_to_xfrc(self, adv_act):
        self.sim.data.xfrc_applied[self._adv_bindex]=np.array([adv_act[0], adv_act[1], adv_act[2], adv_act[3], adv_act[4], adv_act[5]])


    def step(self, a):
        self.time+=self.dt

        #evil forces
        # Random forces
        # =====================================================
        self.randf =np.random.uniform(low=-self.adv_max_force, high=self.adv_max_force, size=(6,))
        
        # Sine forces
        # =====================================================
        #sinusoidal_x = magn * np.sin(5 * np.pi * self.time)
        #sinusoidal_y = 0.5*magn * np.sin(2.5 * np.pi * self.time)
        # # for the time being, I;m assuming the forces are exerted only on x and
        # # y axes (smaller freq for y) and there are no torques
        #sine_forces = np.zeros_like(self.randf)
        #sine_forces[:2] = sinusoidal_x, sinusoidal_y
        #self.randf = sine_forces

        # 3. Triangular forces a) from triangular distribution and b) of actual triangular shape, still only on x and y
        # ======================================================
        #triangular_forces = np.zeros_like(self.randf)
        # a) the distribution
        # triangular_forces[:2] = np.random.triangular(-self.adv_max_force, 0,  self.adv_max_force, size = (2,))
        # self.randf = triangular_forces

        # b) the sawtooth function 
        #triangle_x = magn * signal.sawtooth(5 * np.pi * self.time)
        #triangle_y = 0.5*magn * signal.sawtooth(2.5 * np.pi * self.time)
        #triangular_forces =np.zeros(6)
        #triangular_forces[:2] = triangle_x, triangle_y   
        #self.randf=triangular_forces

        #====apply forces==========================
        #self._adv_to_xfrc(self.randf)
        #print("hej")
        self.do_simulation(a,self.frame_skip)
        reward = self.reward(a)
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

    def reward(self, a):
        #reward function Mutsionii 1
        ee3pos = self.sim.data.get_body_xpos('gripper_r_finger_l')
        target3pos = self.sim.data.get_body_xpos('goal')
        d2=(ee3pos[0]-target3pos[0])**2+(ee3pos[1]-target3pos[1])**2+(ee3pos[2]-target3pos[2])**2
        wl=0.001
        alpha =0.1
        wlog=1.0
        wu=0.01
        cost=(wl*d2+wlog*np.log(d2+alpha)+wu*np.linalg.norm(a)) 
        #print(cost)       
        #return -cost
        #reward reacher
        vec = self.sim.data.get_body_xpos('gripper_r_finger_l')-self.sim.data.get_body_xpos('goal')
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()

        return reward_dist + reward_ctrl
        #reward for doing nothing
        #return - np.linalg.norm(a) # movement is bad
        #return -arm2goal*100*0 - np.linalg.norm(a)*1000 # distance to goal and movment get peneltys

    def _get_obs(self):
        return np.concatenate([
             self.sim.data.qpos.flat[:7],
             self.sim.data.get_body_xpos('gripper_r_finger_l'),
             self.sim.data.get_body_xpos('goal')
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
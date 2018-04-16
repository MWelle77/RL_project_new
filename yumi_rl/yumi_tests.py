
import gym
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import tensorflow as tf
import collections

from gym import spaces



#build the enviroment
env = gym.make('Yumi-Simple-v1')
env.reset()
env.render()




#print infos
print("action space: tourg action on the right arm joints [1, 2, 7, 3, 4, 5, 6]")
print(env.action_space)
print("action upper bound")
print(env.action_space.high)
print("action lower bound")
print(env.action_space.low)


print("observationspace:  jointpos [1, 2, 7, 3, 4, 5, 6], jointvel [1, 2, 7, 3, 4, 5, 6], EEpos [x, y, z],EEqats [x y z w] ")
print(env.observation_space)
print("observationspace upper bound")
print(env.observation_space.high)
print("observationspace lower bound")
print(env.observation_space.low)




rewardlist=[]
actionlist0=[]
actionlist1=[]
actionlist2=[]
actionlist3=[]
actionlist4=[]
actionlist5=[]
actionlist6=[]
foreceslist0=[]
foreceslist1=[]
foreceslist2=[]
foreceslist3=[]
foreceslist4=[]
foreceslist5=[]

totalreward=0


for _ in range(1000): # run for 1000 steps
    env.render()
    #get info from the world
    action = np.zeros(7) #do nothing
    #action = env.action_space.sample() # pick a random action
    #print("performing action:")
    #print(action)
    observation, reward, done, info = env.step(action)

    #print("observing")
    #print(observation)
    #print("reward")
    #print(reward)
    rewardlist.append(reward)
    totalreward+=reward

    actionlist0.append(action[0])
    actionlist1.append(action[1])
    actionlist2.append(action[2])
    actionlist3.append(action[3])
    actionlist4.append(action[4])
    actionlist5.append(action[5])
    actionlist6.append(action[6])

    foreceslist0.append(observation[-6])
    foreceslist1.append(observation[-5])
    foreceslist2.append(observation[-4])
    foreceslist3.append(observation[-3])
    foreceslist4.append(observation[-2])
    foreceslist5.append(observation[-1])
    #forecvec = np.array([observation[-6], observation[-5], observation[-4], observation[-3], observation[-2], observation[-1]])
    #foreceslist.append(forecvec)

    
    #env.step(action) # take action

#print(len(actionlist[1]))
print(totalreward)
plt.plot(rewardlist)
plt.title('Reward (distance to goal pose)')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.ylim(-300, 0)
plt.show()

#x=np.linspace(0, 1000, num=100, endpoint=True)
plt.plot(actionlist0, label="f_j_r_1")
plt.plot(actionlist1, label="f_j_r_2")
plt.plot(actionlist2, label="f_j_r_7")
plt.plot(actionlist3, label="f_j_r_3")
plt.plot(actionlist4, label="f_j_r_4")
plt.plot(actionlist5, label="f_j_r_5")
plt.plot(actionlist6, label="f_j_r_6")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.axis([0, 1000, -2, 2])

plt.title('Action (joint forces)')
plt.xlabel('episodes')
plt.ylabel('force')
plt.show()


plt.plot(foreceslist0, label="f_x")
plt.plot(foreceslist1, label="f_y")
plt.plot(foreceslist2, label="f_z")
plt.plot(foreceslist3, label="t_x")
plt.plot(foreceslist4, label="t_y")
plt.plot(foreceslist5, label="t_z")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title('Adversary forces on right hand')
plt.xlabel('episodes')
plt.ylabel('force/torque')
plt.show()

#!/usr/bin/env python3
import argparse
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger

def train(env_id, num_timesteps, seed):
    from baselines.common import set_global_seeds
    from baselines.ppo2_new import ppo2
    from baselines.ppo2_new.policies import MlpPolicy
    import gym
    import tensorflow as tf
    ncpu = 8
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    env = gym.make('Yumi-Simple-v1') 
    #env = bench.Monitor(env, logger.get_dir())
    env.num_envs=1

    set_global_seeds(seed)
    policy = MlpPolicy
    
    ppo2.learn(policy=policy, env=env, nsteps=1000, nminibatches=100,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps, name="./ppo_models/newhope.pkl")

    

def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()

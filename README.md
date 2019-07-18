# gym-dqn
Implementation of the Deep Q Learning Algorithm mentioned in the paper
[Playing Atari with Deep Reinforcement Learning](https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning/).

The code has been used to solve two [OpenAI Gym](https://gym.openai.com/envs/Acrobot-v1/) environments, namely [Cartpole](https://gym.openai.com/envs/CartPole-v0/) and [Acrobot](https://gym.openai.com/envs/Acrobot-v1/) but with can be used to solve other environments as well. 

Configuration for new environments including DQN hyper-parameters, depth and layer sizes of the Q network should be specified in conf.json.

## Cartpole in action


[![](https://i9.ytimg.com/vi/bjBKKu00daM/mq1.jpg?sqp=CJDUv-kF&rs=AOn4CLCjisvcp4HXjY0-ry85X6g76CW5Vw)](https://youtu.be/bjBKKu00daM)


## Learning across episodes


### 1. Cartpole-v0


![](https://drive.google.com/file/d/1AKDBrXEMD4_Dfk-qST6s8Cuh8vbMnRur/view)

Hyper-parameters used :-

    "num-trials" : 50,
    "num-episodes" : 500,
    "replay-start-size" : 1000,
    "replay-max-size" : 5000,
    "mini-batch-size" : 128,
    "eps-init" : 0.5,
    "eps-final" : 0.0005,
    "exploration-time" : 20000,
    "target-network-update-frequency" : 100,
    "update-frequency" : 1,
    "start-no-op" : false,
    "no-op-max" : 30,
    "gamma" : 0.95,
    "skip-state" : false,
    "skip-state-length" : 4,
    "eps-decay" : 0.995,
    "optimizer" : "adam",
    "hidden_layers" : [128,128]


### 2. Acrobot-v1 


![](https://drive.google.com/open?id=12VdWUhSdwkHRGC-ixOf5c15FfHPOK5sr)

Hyper-parameters used :-

    "num-trials" : 50,
    "num-episodes" : 500,
    "replay-start-size" : 200,
    "replay-max-size" : 10000,
    "mini-batch-size" : 128,
    "eps-init" : 1,
    "eps-final" : 0.0001,
    "exploration-time" : 100000,
    "target-network-update-frequency" : 500,
    "update-frequency" : 50,
    "start-no-op" : false,
    "no-op-max" : 30,
    "gamma" : 0.99,
    "skip-state" : false,
    "skip-state-length" : 4,
    "eps-decay" : 0.995,
    "optimizer" : "adam",
    "hidden_layers" : [256,512]











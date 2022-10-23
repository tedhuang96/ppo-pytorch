# ppo-pytorch
PPO in pytorch version.

We run multiple episodes with the same policy, and create an experience replay buffer out of trajectories in these episodes to perform on-policy policy gradient updates using PPO. We clear the replay buffer from the last run before we start another run of multiple episodes. 

## Set up Python environment
Run
```
virtualenv -p /usr/bin/python3 ppoenv
source ppoenv/bin/activate
pip install -r requirements.txt
```
or
```
virtualenv -p /usr/bin/python3 ppoenv
source ppoenv/bin/activate
pip install gym==0.18.0
pip install torch
pip install tqdm
pip install tensorboard
```

## Train and evaluate agent in RL (cartpole).
```
source ppoenv/bin/activate
python train.py
```
Check training progress by running
```
source ppoenv/bin/activate
tensorboard --logdir results/
```
After training is complete, find `[SAVED_LOG]` in `results/` (e.g., `20221023_172239`). To evaluate without visualization, run
```
source ppoenv/bin/activate
python eval.py --log [SAVED_LOG]
```
To evaluate with visualization, run
```
source ppoenv/bin/activate
python eval.py --log [SAVED_LOG] --visualize
```
If you want to evaluate on a checkpoint at a specific episode (e.g., 1000), run
```
source ppoenv/bin/activate
python eval.py --log [SAVED_LOG] --visualize --training_episodes 1000
```

## Credits
Borrowed code from [ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail), [vita-epfl/CrowdNav](https://github.com/vita-epfl/CrowdNav), and [agrimgupta92/sgan](https://github.com/agrimgupta92/sgan).
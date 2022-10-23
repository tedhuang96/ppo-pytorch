import argparse

def arg_parse():
    parser = argparse.ArgumentParser()
    # gym environment hyperparameters
    parser.add_argument('--state_size', type=int, default=4)
    parser.add_argument('--action_size', type=int, default=2)  
    # rollout hyperparameters
    parser.add_argument('--num_episodes_per_run', type=int, default=100)
    # policy hyperparameters
    parser.add_argument('--policy_model', type=str, default='actor-critic', help='Policy options: actor-critic.')
    parser.add_argument('--model_embedding_size', type=int, default=64)
    parser.add_argument('--model_hidden_size', type=int, default=128)
    parser.add_argument('--model_dropout', type=float, default=0.1)
    parser.add_argument('--num_episodes_per_checkpoint', type=int, default=100)
    # memory hyperparameters
    parser.add_argument('--experience_replay', action='store_true', default=False)
    parser.add_argument('--memory_capacity', type=int, default=100000, help='typically equals num_episodes_per_run * step_limit')
    # training hyperparameters
    parser.add_argument('--training_episodes', type=int, default=2000, help='equals num_episodes_per_run * num_runs')
    # ppo parameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--ppo_epoch', type=int, default=4)
    parser.add_argument('--clip_grad', type=float, default=0.5)
    parser.add_argument('--value_loss_coef', type=float, default=1.)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--use_clipped_value_loss', action='store_true')
    # evaluation inputs
    parser.add_argument('--log', default=None)
    parser.add_argument('--visualize', action='store_true')
    return parser.parse_args()
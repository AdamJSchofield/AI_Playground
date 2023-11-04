import argparse

from torch.cuda import is_available as cuda_available

def get_pg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)

    # Hyperparameters
    parser.add_argument('--lr', type=float, default=.001, help='optimizer learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='a smaller gamma favors earlier win')
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128])

    # Training
    parser.add_argument('--max-epoch', type=int, default=10)
    parser.add_argument('--min_step-per-epoch', type=int, default=10000)
    parser.add_argument('--episode-per-test', type=int, default=30)
    parser.add_argument('--episode-per-collect', type=int, default=30)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--buffer-size', type=int, default=50000, help='VectorReplayBuffer size')

    parser.add_argument('--training-envs', type=int, default=1, help='number of parallel training environments')
    parser.add_argument('--test_envs', type=int, default=1, help='number of parallel testing environments')
    parser.add_argument('--win-rate', type=float, default=0.95, help='win rate (if win reward is 1) or mean reward to stop training')
    parser.add_argument('--agent-id', type=int, default=1, help='the learned agent plays as the agent_id-th player. Choices are 0 and 1')
    
    # log/card_combat/pg/policy.pth
    parser.add_argument('--resume-path', type=str, default='', help='the path of agent pth file for resuming from a pre-trained agent')
    parser.add_argument('--opponent-path', type=str, default='', help='the path of opponent agent pth file for resuming from a pre-trained agent')
    
    # For human eyes
    parser.add_argument('--render-rate', type=float, default=0.01, help='delay between rendered frames')
    parser.add_argument('--watch', default=False, action='store_true', help='watch the play of pre-trained models without training')

    # System and util
    parser.add_argument('--device', type=str, default='cuda' if cuda_available() else 'cpu')
    parser.add_argument('--logdir', type=str, default='log')

    return parser
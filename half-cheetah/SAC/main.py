import argparse
import sys

from SAC.train import train

def get_parser(formatter_class=argparse.RawTextHelpFormatter):
    parser = argparse.ArgumentParser(
        description='PyTorch Soft Actor-Critic',
        usage='Use "python soft_actor_critic --help" for more information',
        formatter_class=formatter_class
    )

    # choice of the subparser
    subparsers = parser.add_subparsers(help='Selection of the mode to perform', dest='mode')

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        '--env-name', default="LunarLanderContinuous-v2", type=str, metavar='',
        help='Gym environment to train on (default: %(default)s)'
    )

    parent_parser.add_argument(
        '--hidden-units', nargs='+', type=int, default=[256, 256], metavar='',
        help='List of networks\' hidden units (default: %(default)s)'
    )

    # train parser
    parser_train = subparsers.add_parser(
        "train", parents=[parent_parser],
        help='Train an agent'
    )

    return parser

def main(mode, **kwargs):
    if mode == 'train':
        print('Training SAC with the following arguments:')
        print("kwargs entering training", kwargs)
        train(**kwargs)
    elif mode == 'eval':
        print(f'Evaluating SAC of the run {kwargs["run_name"]} the following arguments:')
        print(kwargs)
        evaluate(**kwargs)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    parser = get_parser()
    arguments = parser.parse_args()
    
    if len(sys.argv) == 1:
        print("getting stuck here")
        parser.print_help(sys.stderr)
        sys.exit(1)
    print("did i make it")
    kwargs = vars(arguments)
    mode = kwargs.pop('mode', 'train')
    exit(main(mode, **kwargs))

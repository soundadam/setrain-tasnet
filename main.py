from train import Train
import argparse
from option import parse

# sweep
# use_grnn=False
# kernel_size=3,5,7
# channel_size = 16,24,32

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='utils/configs/train_gtcrn.yml', help='Path to option YAML file.')
    args = parser.parse_args()

    opt = parse(args.opt, is_train=True)
    Train(opt)
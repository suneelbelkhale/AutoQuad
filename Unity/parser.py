#add in all parser functions here
import argparse

def get_dqn_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-y", "--yaml", help="include a yaml file (relative path from Unity folder)", type=str, default="")
    parser.add_argument("-t", "--type", help="0: training, 1: inference, 2: demonstrations", type=int, default=0)
    parser.add_argument("--load_model", help="load model name specified in yaml file", action="store_true")
    parser.add_argument("--log_prefix", help="load model name specified in yaml file", type=str, default="")
    return parser
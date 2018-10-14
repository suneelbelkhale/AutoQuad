import numpy as np
import matplotlib.pyplot as plt
import argparse

def main(args):
    obs = np.load(args.observations_file)

    for i in range(obs.shape[0]):
        plt.imshow(obs[i].reshape((128,128)), cmap='gray')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('observations_file', help='observations file to visualize in order')
    args = parser.parse_args()
    main(args)
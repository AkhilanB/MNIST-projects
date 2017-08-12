import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import math
import numpy as np

def display(images):
    size = math.ceil(len(images)**0.5)
    _, axes = plt.subplots(size, size)
    index = 0
    stop = False
    for row, col in itertools.product(range(size),range(size)):
        axes[row, col].imshow(np.clip(images[index],0,1), cmap=cm.gray)
        axes[row, col].axis('off')
        index += 1
        if index == len(images):
            break
    plt.show()

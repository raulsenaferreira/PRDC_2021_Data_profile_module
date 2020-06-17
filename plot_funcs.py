import numpy as np
from math import sqrt, floor
import matplotlib.pyplot as plt
from PIL import Image


def plot_images(data, labels, num_row, num_col):

    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num_row*num_col):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(np.squeeze(data[i]), cmap='gray')
        ax.set_title('{}'.format(labels[i]))
        ax.set_axis_off()
    plt.tight_layout(pad=3.0)
    plt.show()
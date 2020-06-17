from plot_funcs import plot_images
from datasets import Dataset

#reading datasets and its variants and plotting some images
dataset = Dataset('mnist')
x_train, y_train, x_test, y_test = dataset.load_dataset_variation('cvt')
plot_images(x_train[40:80], y_train[40:80], num_row=4, num_col=10)
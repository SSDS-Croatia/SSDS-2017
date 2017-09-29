import os, sys, gzip, math, urllib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Dataset:

    def __init__(self, data, labels=None):
        self.data = data 
        if type(labels) == None:
            self.supervised = False 
        else:
            self.supervised = True
            self.labels = labels 

        self.n = len(data)

        self.batches_complete = 0
        self.position_in_epoch = 0 

    def next_batch(self, batch_size, return_labels=False):
        new_epoch = False
        if self.position_in_epoch + batch_size >= self.n:
            self.position_in_epoch = 0
            self.batches_complete += 1
            new_epoch = True

        batch = self.data[self.position_in_epoch:self.position_in_epoch + batch_size]

        if self.supervised and return_labels:
            batch_labels = self.labels[self.position_in_epoch, self.position_in_epoch + batch_size]      
            batch = (batch, batch_labels)
        self.position_in_epoch += batch_size

        return new_epoch, batch        

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def plot_single(sample, epoch=0):
    plt.axis('off')
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')


def download_mnist(data_folder, dataset):
    """
    Download and extract database
    :param database_name: Database name
    """

    image_files = ['train-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte.gz']
    label_files = ['train-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    url = 'http://yann.lecun.com/exdb/mnist/'
    dataset_folder = os.path.join(data_folder, dataset)


    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
        for filename in image_files + label_files:
            filepath = os.path.join(dataset_folder, filename)
            filepath, _ = urllib.request.urlretrieve(url + filename, filepath)
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    else:
        print('Found {} Data'.format(dataset))

    return dataset_folder

def extract_data(filename, num_data, head_size, data_size):
    with gzip.open(filename) as bytestream:
        bytestream.read(head_size)
        buf = bytestream.read(data_size * num_data)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
    return data

def load_mnist(dataset_folder):
    data = extract_data(dataset_folder + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(dataset_folder + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(dataset_folder + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(dataset_folder + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec

def images_square_grid(images, mode):
    """
    Save images as a square grid
    :param images: Images to be used for the grid
    :param mode: The mode to use for images
    :return: Image of images in a square grid
    """
    # Get maximum size for square grid of images
    save_size = math.floor(np.sqrt(images.shape[0]))

    # Scale to 0-255
    images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)

    # Put images in a square arrangement
    images_in_square = np.reshape(
            images[:save_size*save_size],
            (save_size, save_size, images.shape[1], images.shape[2], images.shape[3]))
    if mode == 'L':
        images_in_square = np.squeeze(images_in_square, 4)

    # Combine images to grid image
    new_im = Image.new(mode, (images.shape[1] * save_size, images.shape[2] * save_size))
    for col_i, col_images in enumerate(images_in_square):
        for image_i, image in enumerate(col_images):
            im = Image.fromarray(image, mode)
            new_im.paste(im, (col_i * images.shape[1], image_i * images.shape[2]))

    return new_im

def get_sample_images(data, dataset='mnist', n=25):
    """
    Get a sample of n images from a dataset, able to be displayed with matplotlib
    :param data_dir: Root directory of the dataset
    :param dataset: 
    """
    # Display options
    if dataset == 'mnist':
        mode = 'L'
    else:
        mode = 'RGB'

    return data[:n], mode

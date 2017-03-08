import dicom
from glob import glob
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize

from keras.models import Model

from lunaModel import get_unet
from lunaIterator import segment_lungs

data_path = '../../../data/'
weights_path = data_path+'luna16/results/'
scan_path = data_path+'stage1/'
labels_file = data_path+'stage1_labels.csv'
test_file = data_path+'stage1_sample_submission.csv'
results_path = data_path+'results/'

img_width = 512
img_height = 512


class dsbIterator(object):

    def __init__(self, files, batch_size=2, shuffle=False, seed=None):
        self.files = files
        self.n = len(files)
        self.batch_size = batch_size
        self.batch_index = 0
        self.total_batches_seen = 0
        self.index_generator = self._gen_index(self.n, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _gen_index(self, n, batch_size=32, shuffle=False, seed=None):

        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield(index_array[current_index: current_index + current_batch_size],
                     current_index, current_batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):

        index_array, current_index, current_batch_size = next(self.index_generator)

        lungs = np.ndarray([current_batch_size, 1, img_width, img_height])
        for i in range(0, current_batch_size):

            # get image data and file
            dcm = dicom.read_file(self.files[index_array[i]])
            img = dcm.pixel_array

            # segment lungs
            lungs[i, 0] = segment_lungs(img)

        return lungs


def segment_train(model_path):

	df = pd.read_csv(labels_file)
	df.sort_values(by='id')
	model = get_unet()
	model.load_weights(model_path)

	for i in tqdm(range(len(df))):
	    uid = df.loc[i, 'id']
	    scans = glob(scan_path+uid+'/*.dcm')

        # TODO: sort by z locations

	    dsb_gen = dsbIterator(scans)
	    nod = model.predict_generator(dsb_gen, len(scans))
	    np.save(data_path+'segmented/train/'+uid+'.npy', nod.astype('uint8'))


def segment_test(model_path):

    df = pd.read_csv(test_file)
    df.sort_values(by='id')
    model = get_unet()
    model.load_weights(model_path)

    for i in tqdm(range(len(df))):
        uid = df.loc[i, 'id']
        scans = glob(scan_path+uid+'/*.dcm')

        # TODO: sort by z locations

        dsb_gen = dsbIterator(scans)
        nod = model.predict_generator(dsb_gen, len(scans))
        np.save(data_path+'segmented/test/'+uid+'.npy', nod.astype('uint8'))


if __name__ == '__main__':
 #    print('Segmenting train files...')
	# segment_train(weights_path+'weights_1e-6_01_-0.608.hdf5')
    print('Segmenting test files...')
    segment_test(weights_path+'weights_1e-6_01_-0.009.hdf5')

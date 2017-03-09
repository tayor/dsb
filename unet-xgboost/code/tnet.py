from __future__ import division
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
#import cv2
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.cluster import KMeans
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier as RF
from skimage import morphology
from skimage import measure
from skimage.transform import resize
import dicom
from glob import glob
from tqdm import tqdm
import os
#import xgboost as xgb
#import SimpleITK as sitk

data_path = '../../../data/'
weights_path = data_path+'luna16/results/'
scan_path = data_path+'stage1/'
labels_file = data_path+'stage1_labels.csv'
test_file = data_path+'stage1_sample_submission.csv'
results_path = data_path+'results/'
scan_path_luna = data_path+'luna16/scans/'
csv_path = data_path+'luna16/csvfiles/annotations.csv'
smooth = 1.
img_height = 512
img_width = 512

#lunaIterator
def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing

def worldToVoxelCoord(worldCoord, origin, spacing):
    strechedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = strechedVoxelCoord / spacing
    return voxelCoord.astype(int)

def make_mask(dims, diameter, width, height):
    img = np.zeros((width, height), np.uint8)
    cv2.circle(img, dims, int(diameter), (1,1,1), -1)
    return img

def segment_lungs(img):

    # normalize
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std

    # remove overflow and underflow, set to mean
    middle = img[100:400,100:400]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    img[img==max]=mean
    img[img==min]=mean

    # cluster and threshold
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)

    # smooth and label regtions
    eroded = morphology.erosion(thresh_img,np.ones([4,4]))
    dilation = morphology.dilation(eroded,np.ones([10,10]))
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)

    # make good mask from good regions, numbers from tutorial
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<480 and B[3]-B[1]<480 and B[0]>40 and B[2]<477:
            good_labels.append(prop.label)
    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0

    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10]))

    # multiply by mask
    img = mask*img

    # renormalizing the masked image
    new_mean = np.mean(img[mask>0])
    new_std = np.std(img[mask>0])
    #
    #  Pushing the background color up to the lower end
    #  of the pixel range for the lungs
    #
    old_min = np.min(img)       # background color
    img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
    img = img-new_mean
    img = img/new_std

    return img

class lunaIterator(object):

    def __init__(self, path, df, batch_size=32, shuffle=False, seed=None):
        self.path = path
        self.df = df
        self.n = len(df)
        self.batch_size = batch_size
        self.batch_index = 0
        self.total_batches_seen = 0
        self.index_generator = self._gen_index(self.n, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _gen_index(self, n , batch_size=32, shuffle=False, seed=None):

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
        masks = np.ndarray([current_batch_size, 1, img_width, img_height])
        for i in range(0, current_batch_size):
            # get image data and file
            ann = self.df.iloc[index_array[i]]
            filename = self.path+ann['seriesuid']+'.mhd'
            image, origin, spacing = load_itk_image(filename)

            # get voxelCoords
            worldCoord = np.asarray([ann['coordZ'], ann['coordY'], ann['coordX']])
            voxelCoord = worldToVoxelCoord(worldCoord, origin, spacing)

            # segment lungs and make mask
            lungs[i, 0] = segment_lungs(image[voxelCoord[0]])
            masks[i, 0] = make_mask((voxelCoord[2], voxelCoord[1]), ann['diameter_mm']+5,
               img_width, img_height)

        return lungs, masks

#lunaModel
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((1,img_height, img_width))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    conv3 = BatchNormalization(axis=1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    conv4 = BatchNormalization(axis=1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    conv5 = BatchNormalization(axis=1)(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)
    conv6 = BatchNormalization(axis=1)(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)
    conv7 = BatchNormalization(axis=1)(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)
    conv8 = BatchNormalization(axis=1)(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)
    conv9 = BatchNormalization(axis=1)(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1.0e-6), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def train_model(csv_path, scan_path, results_path, batch_size=4, num_epoch=8, load_weights=False):
    # all nodules
    df = pd.read_csv(csv_path)
    num_ids = len(df)
    num_train = int(num_ids * 0.75)
    num_val = num_ids - num_train
    df_train = df[:num_train]
    df_val = df[num_train:]

    train_gen = lunaIterator(scan_path, df_train, batch_size=batch_size, shuffle=True)
    val_gen = lunaIterator(scan_path, df_val, batch_size=batch_size, shuffle=False)

    model = get_unet()

    if load_weights:
    	model.load_weights(results_path+'weights_1e-6_01_-0.608.hdf5')

    model_checkpoint = ModelCheckpoint(results_path+'weights_1e-6_{epoch:02d}_{val_loss:.3f}.hdf5',
                                         monitor='val_loss', save_best_only=False, save_weights_only=True)

    model.fit_generator(train_gen, num_train, nb_epoch=num_epoch, validation_data=val_gen,
                    nb_val_samples=num_val, callbacks=[model_checkpoint])

#segmentDSB
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
#extract features
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def load_scan(scan_path):
    slices = [dicom.read_file(scan_path + '/' + s) for s in os.listdir(scan_path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))

    slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    return slices


def zloc_min_max(slices):
    zlocs = [s.ImagePositionPatient[2] for s in slices]
    return min(zlocs), max(zlocs)


def get_regions(seg_slice):
    thr = np.where(seg_slice > np.mean(seg_slice),-1,1)
    label_image = measure.label(thr)
    labels = label_image.astype(int)
    regions = measure.regionprops(labels)
    return regions


def get_features(uid, seg_path):
    dcm_slices = load_scan(scan_path+uid)
    dcm_images = get_pixels_hu(dcm_slices)
    seg_slices = np.load(seg_path+uid+'.npy')
    nslices = seg_slices.shape[0]

    # crude hueristic to filter some bad segmentaitons
    # do not allow any nodes to be larger than 10% of the pixels to eliminate background regions
    maxAllowedArea = 0.10 * 512 * 512
    minAllowedArea = 16

    #metrics
    totalArea = 0.
    avgArea = 0.
    maxArea = 0.
    avgEcc = 0.
    avgEquivlentDiameter = 0.
    stdEquivlentDiameter = 0.
    weightedX = 0.
    weightedY = 0.
    numNodes = 0.
    numNodesperSlice = 0.

    areas = []
    eqDiameters = []
    zlocs = []
    max_hu = []
    avg_hu = []

    z_min, z_max = zloc_min_max(dcm_slices)
    quad_len = (z_max-z_min)/4
    q1 = z_min + quad_len
    q2 = z_min + 2*quad_len
    q3 = z_min + 3*quad_len

    q0_cnt = 0
    q1_cnt = 0
    q2_cnt = 0
    q3_cnt = 0


    for slicen in range(nslices):
        regions = get_regions(seg_slices[slicen,0,:,:])
        for region in regions:
            if region.area > maxAllowedArea or region.area < minAllowedArea:
                continue
            totalArea += region.area
            areas.append(region.area)
            avgEcc += region.eccentricity
            eqDiameters.append(region.equivalent_diameter)
            weightedX += region.centroid[0]*region.area
            weightedY += region.centroid[1]*region.area
            numNodes += 1

            zlocs.append(dcm_slices[slicen].ImagePositionPatient[2])

            seg = dcm_images[slicen]
            min_x, min_y, max_x, max_y = region.bbox
            seg = seg[min_x:max_x, min_y:max_y]

            max_hu.append(np.amax(seg))
            avg_hu.append(np.average(seg))

    weightedX = weightedX / totalArea
    weightedY = weightedY / totalArea
    avgArea = totalArea / numNodes
    avgEcc = avgEcc / numNodes
    avgEquivlentDiameter = np.average(eqDiameters)
    stdEquivlentDiameter = np.std(eqDiameters)
    maxArea = max(areas)
    numNodesperSlice = numNodes*1. / nslices

    # softest nodule found by absolute
    min_max_hu = min(max_hu)
    avg_max_hu = np.average(max_hu)

    # softest average
    min_avg_hu = min(avg_hu)

    # total average
    avg_avg_hu = np.average(avg_hu)

    for loc in zlocs:
        if loc <= q1:
            q0_cnt += 1
        elif loc <= q2:
            q1_cnt += 1
        elif loc <= q2:
            q2_cnt += 1
        else:
            q3_cnt += 1


    return np.array([avgArea,maxArea,avgEcc,avgEquivlentDiameter,\
                     stdEquivlentDiameter, weightedX, weightedY, numNodes, numNodesperSlice, \
                     q0_cnt, q1_cnt, q2_cnt, q3_cnt, min_max_hu, avg_max_hu, min_avg_hu, avg_avg_hu])


def createTrainFeatureDataset(nodfiles=None):
    if nodfiles == None:
        noddir = data_path+'segmented/train/'
        nodfiles = glob(noddir +"*.npy")

    df = pd.read_csv(labels_file)

    numfeatures = 17
    feature_array = np.zeros((len(nodfiles),numfeatures))
    truth_metric = np.zeros((len(nodfiles)))

    for i  in tqdm(range(len(nodfiles))):
        base = os.path.basename(nodfiles[i])
        patID = os.path.splitext(base)[0]
        truth_metric[i] = df[df['id'] == patID].iloc[0]['cancer']
        feature_array[i] = get_features(patID, noddir)

    np.save(results_path+'dataY.npy', truth_metric)
    np.save(results_path+'dataX.npy', feature_array)


def createTestFeatureDataset(nodfiles=None):
    if nodfiles == None:
        noddir = data_path+'segmented/test/'
        nodfiles = glob(noddir +"*.npy")

    df = pd.read_csv(test_file)

    numfeatures = 17
    feature_array = np.zeros((len(nodfiles),numfeatures))
    ids = np.ndarray(len(nodfiles), dtype=object)

    for i  in tqdm(range(len(nodfiles))):
        base = os.path.basename(nodfiles[i])
        patID = os.path.splitext(base)[0]
        ids[i] = patID
        feature_array[i] = get_features(patID, noddir)

    np.save(results_path+'testX.npy', feature_array)
    np.save(results_path+'testId.npy', ids)

#Classify features
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def logregobj(y_true, y_pred):
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    grad = y_pred - y_true
    hess = y_pred * (1.0-y_pred)
    return grad, hess


def train_classifier():
    X = np.load(results_path+'dataX.npy')
    Y = np.load(results_path+'dataY.npy')

    kf = KFold(Y, n_folds=3)
    y_pred = Y * 0
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]
        clf = RF(n_estimators=100, n_jobs=3)
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)
    print(classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"]))

    y_pred[y_pred > 0.95] = 0.75
    y_pred[y_pred < 0.05] = 0.25
    print("logloss",logloss(Y, y_pred))

    # All Cancer
    print("Predicting all positive")
    y_pred = np.ones(Y.shape)
    print(classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"]))
    print("logloss",logloss(Y, y_pred))

    # No Cancer
    print("Predicting all negative")
    y_pred = Y*0
    print(classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"]))
    print("logloss",logloss(Y, y_pred))

    # try XGBoost
    print ("XGBoost")
    kf = KFold(Y, n_folds=3)
    y_pred = Y * 0
    for train, test in kf:
        X_train, X_test, y_train, y_test = X[train,:], X[test,:], Y[train], Y[test]
        clf = xgb.XGBClassifier(learning_rate =0.1, n_estimators=250, max_depth=9, min_child_weight=1, gamma=0.1,
                                 subsample=0.85, colsample_bytree=0.75, objective= 'binary:logistic', nthread=4,
                                  scale_pos_weight=1, seed=27, reg_alpha=0.01)
        # param = {'num_class': 2}
        # clf.set_params(**param)
        clf.fit(X_train, y_train)
        y_pred[test] = clf.predict(X_test)
    # print classification_report(Y, y_pred, target_names=["No Cancer", "Cancer"])
    params = {'objective': 'multi:softmax',
              'num_class': 2}

    dtrain = xgb.DMatrix(X, Y)
    clf = xgb.train(params, dtrain)
    y_pred = clf.predict(xgb.DMatrix(X))

    y_pred[y_pred > 0.95] = 0.95
    y_pred[y_pred < 0.05] = 0.05

    print(y_pred[0:100])
    print("logloss",logloss(Y, y_pred))

    print("Predicting all 0.25")
    y_pred = np.ones(Y.shape)*0.25
    print("logloss",logloss(Y, y_pred))


def predict_test():

    X = np.load(results_path+'dataX.npy')
    Y = np.load(results_path+'dataY.npy')
    X_test = np.load(results_path+'testX.npy')
    X_ids = np.load(results_path+'testId.npy')

    params = {'objective': 'multi:softmax',
              'num_class': 2}

    dtrain = xgb.DMatrix(X, Y)
    clf = xgb.train(params, dtrain)

    y_pred = clf.predict(xgb.DMatrix(X_test))

    y_pred[y_pred > 0.90] = 0.90
    y_pred[y_pred < 0.05] = 0.05

    subm = np.stack([X_ids, y_pred], axis=1)

    subm_file_name = results_path+'subm3.csv'
    np.savetxt(subm_file_name, subm, fmt='%s,%.5f', header='id,cancer', comments='')
    print('Saved predictions in {}'.format(subm_file_name))

if __name__ == '__main__':
    '''
    print('training unet on luna16 dataset..')
    train_model(csv_path, scan_path_luna, weights_path, num_epoch=5, load_weights=False)
    print('Segmenting train files..')
    segment_train(weights_path+'weights_1e-6_01_-0.009.hdf5')
    print('Segmenting test files..')
    segment_test(weights_path+'weights_1e-6_01_-0.009.hdf5')
    '''
    print('Creating train features...')
    createTrainFeatureDataset()
    print('Creating test data features')
    createTestFeatureDataset()
    print('Training classifier on features..')
    train_classifier()
    print('Predicting on test set..')
    predict_test()

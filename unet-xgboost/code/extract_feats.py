# usage: python classify_nodes.py nodes.npy 

import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import dicom

from skimage import measure

data_path = '/home/ubuntu/fs/data/dsb17/'
labels_file = data_path+'stage1_labels.csv'
test_file = data_path+'stage1_sample_submission.csv'
results_path = data_path+'results/'
scan_path = data_path+'stage1/'


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
        feature_array[i] = get_features(patID)
    
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
        feature_array[i] = get_features(patID)
    
    np.save(results_path+'testX.npy', feature_array)
    np.save(results_path+'testId.npy', ids)


if __name__ == "__main__":   
    # print('Creating train features...') 
    # createTrainFeatureDataset(data_path+'segmented/train/')
    print('Creating test data features')
    createTestFeatureDataset(data_path+'segmented/test/')

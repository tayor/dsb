import keras
import theano
import numpy as np
import pandas as pd
import load_data
import SimpleITK
np.random.seed(42) #Make sure results can be reproduced
from keras.models import Sequential #Feed forward?
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Convolution3D, MaxPooling3D
from keras.utils import np_utils

labels = pd.read_csv('stage1_labels.csv') #load the CSV containing the labels
patient_id = '0a0c32c9e08cc2ea76a71649de56be6d' #Our selected patient ID
patient1 = load_data.get_lungs_arr(patient_id) #Use preprocessing code to obtain 64x64x64 3D image for our patient


X_train = patient1.reshape(1,1,64,64,64)
'''Can be interpret in this way (number of samples, number of channels (1 for b/w 3 for colour), depth, width height'''


y_train = np_utils.to_categorical(labels[labels['id']==patient_id]['cancer'].values, 2) # Change labels to array. E.g. instead of having 0 or 1 we have [1,0] or [0,1] where the index of '1' is the true label


model = Sequential() # Feed forward 1 by 1 hence sequential. Other architectures such as LSTM would be different

model.add(Convolution3D(8,10,10,10, activation='relu',input_shape=(1,64,64,64),dim_ordering='th'))
'''Add layers to model. In this case our first layer is a 3D convolution layer. The first 4 numbers represents (number of filters, filter d,w,h). The activation is the function used to make the network non-linear. Since this is the first layer of the model, we will need an input_shape parameter too. dim_ordering is just the parameter telling the model that we use Theano convention for the dimesion ordering i.e. channel d,w,h'''

model.add(Convolution3D(8,10,10,10,activation='relu',dim_ordering='th'))
# print model.output_shape
# >>(None, 8, 46, 46, 46)

model.add(MaxPooling3D(pool_size=(2,2,2))) # Reduce number of parameters
# print model.output_shape
# >>(None, 8, 23,23,23)

model.add(Convolution3D(8,5,5,5,activation='relu',dim_ordering='th'))
# print model.output_shape
# >>(None, 8, 19, 19, 19)

model.add(MaxPooling3D(pool_size=(2,2,2))) # Reduce number of parameters
# print model.output_shape
# >>(None, 8, 9, 9, 9) 

model.add(Dropout(0.25)) # Regularised the model to prevent overfite

model.add(Flatten()) #Must flattern to pass into dense
# print model.output_shape
# >>5832

model.add(Dense(128, activation='relu')) # 
# print model.output_shape
# >>128

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax')) # 10 ouput matches y
# print model.output_shape
# >>2


model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])


model.fit(X_train,y_train)

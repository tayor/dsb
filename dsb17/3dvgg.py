model = Sequential()
model.add(ZeroPadding3D((1,1,1),input_shape=(1,128,128,128)))
model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
model.add(ZeroPadding3D((1,1,1)))
model.add(Convolution3D(64, 3, 3, 3, activation='relu'))
model.add(MaxPooling3D((2,2,2), strides=(2,2,2), dim_ordering="th"))

model.add(ZeroPadding3D((1,1,1)))
model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
model.add(ZeroPadding3D((1,1,1)))
model.add(Convolution3D(128, 3, 3, 3, activation='relu'))
model.add(MaxPooling3D((2,2,2), strides=(2,2,2), dim_ordering="th"))

model.add(ZeroPadding3D((1,1,1)))
model.add(Convolution3D(256, 3, 3, 3, activation='relu'))
model.add(ZeroPadding3D((1,1,1)))
model.add(Convolution3D(256, 3, 3, 3, activation='relu'))
model.add(ZeroPadding3D((1,1,1)))
model.add(Convolution3D(256, 3, 3, 3, activation='relu'))
model.add(MaxPooling3D((2,2,2), strides=(2,2,2), dim_ordering="th"))

model.add(ZeroPadding3D((1,1,1)))
model.add(Convolution3D(512, 3, 3, 3, activation='relu'))
model.add(ZeroPadding3D((1,1,1)))
model.add(Convolution3D(512, 3, 3, 3, activation='relu'))
model.add(ZeroPadding3D((1,1,1)))
model.add(Convolution3D(512, 3, 3, 3, activation='relu'))
model.add(MaxPooling3D((2,2,2), strides=(2,2,2), dim_ordering="th"))

model.add(ZeroPadding3D((1,1,1)))
model.add(Convolution3D(512, 3, 3, 3, activation='relu'))
model.add(ZeroPadding3D((1,1,1)))
model.add(Convolution3D(512, 3, 3, 3, activation='relu'))
model.add(ZeroPadding3D((1,1,1)))
model.add(Convolution3D(512, 3, 3, 3, activation='relu'))
model.add(MaxPooling3D((2,2,2), strides=(2,2,2), dim_ordering="th"))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=l_rate, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy')

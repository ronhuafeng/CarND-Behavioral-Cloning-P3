import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

from keras.models import Model
from keras.layers import Flatten, Input, Dense, Conv2D, MaxPooling2D
from keras.layers import Activation, BatchNormalization, Dropout, Cropping2D
from keras.layers.merge import concatenate
from keras import optimizers

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2, shuffle=True)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
batch_size = 64
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320  # Trimmed image format

inputs = Input(shape=(row, col, ch))

x = Cropping2D(cropping=((50, 20), (0, 0)))(inputs)

x = Conv2D(6, (5, 5), padding='valid')(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
x = Activation(activation='elu')(x)
x = BatchNormalization()(x)

x = Conv2D(24, (5, 5), padding='valid')(inputs)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
x = Activation(activation='elu')(x)
x = BatchNormalization()(x)

x = Conv2D(64, (3, 3), padding='valid')(inputs)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
x = Activation(activation='elu')(x)
x = BatchNormalization()(x)

x = Flatten()(x)

x = Dense(units=200)(x)
x = Activation(activation='elu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(units=60)(x)
x = Activation(activation='elu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(units=20)(x)
x = Activation(activation='softmax')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

logits = Dense(1)(x)

adam_opt = optimizers.adam(lr=0.001)

model = Model(inputs=inputs, outputs=logits)
model.compile(optimizer=adam_opt, loss='mse', metrics=['accuracy'])

model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size, 
            validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=3)

model.save('model.h5', overwrite=True)


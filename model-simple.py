import csv
import cv2
import imageio
import numpy as np
import matplotlib as plt
from sklearn.utils import shuffle

from keras.models import Model, load_model
from keras.layers import Flatten, Input, Dense, Conv2D, MaxPooling2D
from keras.layers import Activation, BatchNormalization, Dropout, Cropping2D, Lambda
from keras.layers.merge import concatenate
from keras import optimizers

from keras.applications.inception_v3 import InceptionV3

# data_paths = ['./data-1', './data-2', 'data-3-reversed'] 
data_paths = ['./data-1', 'data-3-reversed'] 
data_paths = ['./data-3-reversed'] 

total_samples = []
total_steerings = []
for path in data_paths:
    samples = []
    steerings = []
    samples_left = []
    steerings_left = []
    samples_right = []
    steerings_right = []
    correction = 0.2
    with open(path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            samples.append(path + '/IMG/' + line[0].split('/')[-1])
            steerings.append(float(line[3]))
            samples_left.append(path + '/IMG/' + line[1].split('/')[-1])
            steerings_left.append(float(line[3]) + correction)
            samples_right.append(path + '/IMG/' + line[2].split('/')[-1])
            steerings_right.append(float(line[3]) - correction)
            
    total_samples.extend(samples[0:-100])
    total_steerings.extend(steerings[0:-100])
    total_samples.extend(samples_left[0:-100])
    total_steerings.extend(steerings_left[0:-100])
    total_samples.extend(samples_right[0:-100])
    total_steerings.extend(steerings_right[0:-100])

from sklearn.model_selection import train_test_split
shuffle(total_samples, total_steerings)

# total_samples = total_samples[0:10000]
# total_steerings = total_steerings[0:10000]
train_samples, validation_samples, train_steerings, valid_steerings = train_test_split(total_samples, total_steerings, test_size=0.2, shuffle=True)

def generator(samples, steerings, batch_size=32):
    num_samples = len(samples)
    batch_size = batch_size # since we flip the image to augment data

    while 1: # Loop forever so the generator never terminates
        shuffle(samples, steerings)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_steerings = steerings[offset:offset+batch_size]

            images = []
            angles = []
            for i, batch_sample in enumerate(batch_samples):
                name = batch_sample
                center_angle = steerings[i]
                center_image = imageio.imread(name)
                images.append(center_image)
                angles.append(center_angle)
                # images.append(np.fliplr(center_image))
                # angles.append(-center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
batch_size = 64
train_generator = generator(train_samples, train_steerings, batch_size=batch_size)
validation_generator = generator(validation_samples, valid_steerings, batch_size=batch_size)

ch, row, col = 3, 160, 320

inputs = Input(shape=(row, col, ch))

normalized_inputs = Cropping2D(cropping=((70,25), (0,0)))(inputs)
normalized_inputs = Lambda(lambda i: i / 255.0 - 0.5)(normalized_inputs)

x = Conv2D(6, (5, 5), padding='valid', activation='relu')(normalized_inputs)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
# x = BatchNormalization()(x)

x = Conv2D(24, (5, 5), padding='valid', activation='relu')(inputs)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
x = Activation(activation='elu')(x)
x = BatchNormalization()(x)

# x = Conv2D(64, (3, 3), padding='valid')(inputs)
# x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
# x = Activation(activation='elu')(x)
# x = BatchNormalization()(x)

x = Flatten()(x)

x = Dense(units=84)(x)
x = Activation(activation='elu')(x)
# x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(units=43)(x)
# x = Activation(activation='elu')(x)
# x = BatchNormalization()(x)
x = Dropout(0.5)(x)

logits = Dense(1)(x)                        

adam_opt = optimizers.adam(lr=0.001)

model = Model(inputs=inputs, outputs=logits)
# model.compile(optimizer=adam_opt, loss='mse', metrics=['accuracy'])
model.compile(optimizer=adam_opt, loss='mse')

history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size, 
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples), nb_epoch=3, shuffle=True)

model.save('model.h5', overwrite=True)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
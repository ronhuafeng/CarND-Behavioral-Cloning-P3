import csv
import imageio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from keras.models import Model, load_model
from keras.layers import Flatten, Input, Dense, Conv2D, MaxPooling2D
from keras.layers import Activation, BatchNormalization, Dropout, Cropping2D, Lambda
from keras.layers.merge import concatenate
from keras import optimizers

from keras import regularizers

from keras.backend import tf as ktf

from keras.applications.inception_v3 import InceptionV3

data_paths = ['data-1', 'data-3-reversed', 'data-4', 'data-5-reversed'] 
data_paths = ['data', 'data-1', 'data-4'] 
data_paths = ['../data/data-6'] 

total_samples = []
total_steerings = []
L_AUG = True
R_AUG = True

for path in data_paths:
    samples = []
    steerings = []
    samples_left = []
    steerings_left = []
    samples_right = []
    steerings_right = []
    correction = 0.1
    with open(path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for line in reader:
            samples.append(path + '/IMG/' + line[0].split('/')[-1])
            steerings.append(float(line[3]))
            if L_AUG == True:
                samples_left.append(path + '/IMG/' + line[1].split('/')[-1])
                steerings_left.append(float(line[3]) + correction)
            if R_AUG == True:
                samples_right.append(path + '/IMG/' + line[2].split('/')[-1])
                steerings_right.append(float(line[3]) - correction)
            
    total_samples.extend(samples[0:-100])
    total_steerings.extend(steerings[0:-100])
    if L_AUG == True:
        total_samples.extend(samples_left[0:-100])
        total_steerings.extend(steerings_left[0:-100])
    if R_AUG == True:
        total_samples.extend(samples_right[0:-100])
        total_steerings.extend(steerings_right[0:-100])

from sklearn.model_selection import train_test_split
shuffle(total_samples, total_steerings)

train_samples, validation_samples, train_steerings, valid_steerings = train_test_split(total_samples, total_steerings, test_size=0.2, shuffle=True)

def generator(samples, steerings, batch_size=32):
    num_samples = len(samples)

    while 1:
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
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

batch_size = 32
train_generator = generator(train_samples, train_steerings, batch_size=batch_size)
validation_generator = generator(validation_samples, valid_steerings, batch_size=batch_size*2)

inputs = Input(shape=(160, 320, 3))

x = Cropping2D(cropping=((60,24), (0,0)))(inputs)
# x = Lambda(lambda img: ktf.image.resize_images(img, (38, 160)))(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(6, (5, 5), padding='valid', activation='relu')(x)
x = MaxPooling2D()(x)
x = BatchNormalization()(x)

x = Conv2D(24, (5, 5), padding='valid', activation='relu')(x)
x = MaxPooling2D()(x)
x = BatchNormalization()(x)

x = Flatten()(x)

x = Dense(units=200)(x)
x = Activation(activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.7)(x)

x = Dense(units=84)(x)
x = Activation(activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

# logits = Dense(1, kernel_regularizer=regularizers.l2(0.01))(x)                        
logits = Dense(1)(x)                        

adam_opt = optimizers.adam(lr=0.001)

model = Model(inputs=inputs, outputs=logits)
model.compile(optimizer=adam_opt, loss='mse')

history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size, 
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples), nb_epoch=3, shuffle=True)

model.save('model.h5', overwrite=True)

print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
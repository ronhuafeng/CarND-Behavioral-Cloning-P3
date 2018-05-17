import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Flatten, Input, Dense, Conv2D, MaxPooling2D
from keras.layers import Activation, BatchNormalization, Dropout, Cropping2D, Lambda
from keras.layers.merge import concatenate
from keras import optimizers, losses
from keras import regularizers
from keras.backend import tf as ktf
from keras.applications.inception_v3 import InceptionV3

from model import extract_data, sample_generator

data_paths = ['data-1', 'data-3-reversed', 'data-4', 'data-5-reversed'] 

data_paths_full = [
    '../data/data-1', 
    '../data/data-4', 
    '../data/data-5-reversed', 
    '../data/data-6', 
    '../data/data-7', 
    '../data/data-6-7-red-corner-1', 
    '../data/data-6-7-after-red-corner-2', 
    '../data/data-6-7-red-corner-connect-3',
    '../data/data-8-red1'
    ] 

# data_paths = ['data', 'data-1', 'data-4'] 
data_paths = ['../data/data-6', '../data/data-7']

# data_paths = ['../data/data-6-7-red-corner-1', '../data/data-6-7-after-red-corner-2', '../data/data-6-7-red-corner-connect-3', '../data/data-7'] 
# data_paths = data_paths_full

L_AUG = True
R_AUG = True
Flip_AUG = True
Transfer_Learn = False # used for transfer learning
batch_size = 32

def nvidia_model() -> Model:
    inputs = Input(shape=(160, 320, 3))

    x = Cropping2D(cropping=((60,24), (0,0)))(inputs)
    x = BatchNormalization()(x)

    x = Conv2D(24, (5, 5), padding='valid', strides=(2, 2), activation='relu')(x)
    x = Conv2D(36, (5, 5), padding='valid', strides=(2, 2), activation='relu')(x)
    x = Conv2D(48, (3, 3), padding='valid', strides=(2, 2), activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='valid', strides=(2, 2), activation='relu')(x)

    x = Flatten()(x)

    x = Dense(units=1164)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(units=100)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(units=50)(x)
    x = Activation(activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(units=10)(x)
    x = Activation(activation='relu')(x)
                    
    logits = Dense(1)(x)
    model = Model(inputs=inputs, outputs=logits)
    return model

if Transfer_Learn is True:
    model = load_model('model-no23.h5', compile=False)

    for i, layer in enumerate(model.layers):
        if i < 11: # 8 is two fully connected layer
            layer.trainable = False
        else:
            layer.trainable = True
        print(i, layer.name)

    adam_opt = optimizers.adam(lr=0.002)
    model.compile(optimizer=adam_opt, loss=losses.mean_squared_error)   
else:
    model = nvidia_model()
    adam_opt = optimizers.adam(lr=0.001)
    model.compile(optimizer=adam_opt, loss=losses.mean_squared_error)


# Extract training&validation dataset 

total_samples, total_steerings = extract_data(data_paths, L_AUG=L_AUG, R_AUG=R_AUG, correction=0.15)
shuffle(total_samples, total_steerings)

train_samples, validation_samples, train_steerings, valid_steerings = train_test_split(total_samples, total_steerings, test_size=0.1, shuffle=True)

train_generator = sample_generator(train_samples, train_steerings, batch_size=batch_size, Flip_AUG=Flip_AUG)
validation_generator = sample_generator(validation_samples, valid_steerings, batch_size=batch_size*4, Flip_AUG=Flip_AUG)

# Train model

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
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Model, load_model
from keras.layers import Flatten, Input, Dense, Conv2D, MaxPooling2D
from keras.layers import Activation, BatchNormalization, Dropout, Cropping2D, Lambda, GaussianNoise
from keras.layers.merge import concatenate
from keras import optimizers, losses
from keras import regularizers
from keras.backend import tf as ktf
from keras.applications.inception_v3 import InceptionV3
from model import extract_data, sample_generator, default_direction_filter

BATCH_SIZE = 256
LR_Angle_Correction=0.1

# construct model struction based on the nvidia model
def nvidia_model() -> Model:
    inputs = Input(shape=(160, 320, 3))

    x = Cropping2D(cropping=((60, 24), (0, 0)))(inputs)
    x = BatchNormalization()(x)
    x = GaussianNoise(0.01)(x)

    x = Conv2D(24, (5, 5), padding='valid',
               strides=(2, 2), activation='elu')(x)
    x = Conv2D(36, (5, 5), padding='valid',
               strides=(2, 2), activation='elu')(x)
    x = Conv2D(48, (3, 3), padding='valid',
               strides=(2, 2), activation='elu')(x)
    x = Conv2D(64, (3, 3), padding='valid',
               strides=(2, 2), activation='elu')(x)

    x = Flatten()(x)

    x = Dense(units=1164)(x)
    x = Activation(activation='elu')(x)
    x = Dropout(0.5)(x)

    x = Dense(units=100)(x)
    x = Activation(activation='elu')(x)
    x = Dropout(0.5)(x)

    x = Dense(units=50)(x)
    x = Activation(activation='elu')(x)
    x = Dropout(0.5)(x)

    x = Dense(units=10)(x)
    x = Activation(activation='elu')(x)

    logits = Dense(1)(x)
    model = Model(inputs=inputs, outputs=logits)
    return model

# add a traing work to a works list
def add_work(data_paths, L_AUG=False, R_AUG=False, Flip_AUG=True, filter_function=default_direction_filter, epochs=5, bypass=False, works=[]):
    works.append({
        'data_paths': data_paths, 
        'L_AUG': L_AUG, 
        'R_AUG': R_AUG, 
        'Flip_AUG': Flip_AUG, 
        'Filter_Function': filter_function,
        'epochs': epochs,
        'bypass': bypass})
    return works

# process a works list based on a existing model
def process_works(works, base_model):
    model = base_model

    for i_w, work in enumerate(works):
        # If work is set to be bypassed, it means this work needs not to be trained.
        # Mainly because the work is trained before and already generates a model file.
        if work['bypass'] is True:
            continue

        if i_w > 0:
            for i, layer in enumerate(model.layers):
                if i < 15:  # 15 train 3 fc layers; 18 train 2 fc layers
                    layer.trainable = True # set to True if all layers are re-trained.
                else:
                    layer.trainable = True
                print(i, layer.name)
            # exit(0)

            adam_opt = optimizers.adam(lr=0.0005)
            model.compile(optimizer=adam_opt, loss=losses.mean_squared_error)
        else:
            adam_opt = optimizers.adam(lr=0.001)
            model.compile(optimizer=adam_opt, loss=losses.mean_squared_error)

        # Extract training&validation dataset

        total_samples, total_steerings = extract_data(
            data_paths=work['data_paths'],
            L_AUG=work['L_AUG'], 
            R_AUG=work['R_AUG'], 
            correction=LR_Angle_Correction,
            directionFilter=work['Filter_Function'])

        shuffle(total_samples, total_steerings)

        train_samples, validation_samples, train_steerings, valid_steerings = train_test_split(
            total_samples, total_steerings, test_size=0.1, shuffle=True)

        train_generator = sample_generator(
            train_samples, train_steerings, 
            batch_size=BATCH_SIZE, 
            Flip_AUG=work['Flip_AUG'])
        validation_generator = sample_generator(
            validation_samples, valid_steerings, 
            batch_size=BATCH_SIZE, 
            Flip_AUG=work['Flip_AUG'])

        # Train model
        history_object = model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_samples)/BATCH_SIZE,
            validation_data=validation_generator,
            validation_steps=len(validation_samples)/BATCH_SIZE,
            nb_epoch=work['epochs'],
            shuffle=True)

        model.save('model-work-' + str(i_w) + '.h5', overwrite=True)

    print(history_object.history.keys())
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

# only consider steering angles that moves the car from right border to middle road
def filterRightBorder(ang):
    return (ang < -0.03), ang

# only consider steering angles that moves the car from left border to middle road
def filterLeftBorder(ang):
    return (ang > 0.000), ang

# only consider steering angles that actually changes the direction of the car
def filterAboveZeroAngle(ang):
    return (abs(ang) > 0.005), ang

works = add_work(['../data/data-6', '../data/data-7'], bypass=False, epochs=7)
works = add_work(['../data/data-8-red1'], bypass=False, filter_function=filterRightBorder, epochs=10)
works = add_work(['../data/data-10-right-border'], epochs=10, bypass=False, filter_function=filterRightBorder)
works = add_work(['../data/data-9-left-border'], epochs=10, bypass=False, filter_function=filterLeftBorder)
works = add_work(['../data/data-10-right-border'], epochs=5, bypass=False, filter_function=filterRightBorder)
works = add_work(['../data/data-9-left-border'], epochs=5, bypass=False, filter_function=filterLeftBorder)
works = add_work(['../data/data-10-right-border'], epochs=3, bypass=False, filter_function=filterRightBorder)

# base_model_name = 'model-work-3.h5'
# model = load_model(base_model_name, compile=False)
model = nvidia_model()

# train the model using `nvidia_model()`
process_works(works, model)
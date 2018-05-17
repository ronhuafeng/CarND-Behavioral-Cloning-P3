import csv
import numpy as np
from sklearn.utils import shuffle
import imageio

def extract_data(data_paths, L_AUG=True, R_AUG=True, correction=0.2):
    all_samples = []
    all_steerings = []
    for path in data_paths:
        samples = []
        steerings = []
        samples_left = []
        steerings_left = []
        samples_right = []
        steerings_right = []
        with open(path + '/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for line in reader:
                samples.append(path + '/IMG/' + line[0].split('/')[-1])
                steerings.append(float(line[3]))
                if L_AUG is True:
                    samples_left.append(path + '/IMG/' + line[1].split('/')[-1])
                    steerings_left.append(float(line[3]) + correction)
                if R_AUG is True:
                    samples_right.append(path + '/IMG/' + line[2].split('/')[-1])
                    steerings_right.append(float(line[3]) - correction)
                
        all_samples.extend(samples[0:-100])
        all_steerings.extend(steerings[0:-100])
        if L_AUG == True:
            all_samples.extend(samples_left[0:-100])
            all_steerings.extend(steerings_left[0:-100])
        if R_AUG == True:
            all_samples.extend(samples_right[0:-100])
            all_steerings.extend(steerings_right[0:-100])
    return all_samples, all_steerings

def sample_generator(samples, steerings, batch_size=32, Flip_AUG=True):
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
                if Flip_AUG is True:
                    images.append(np.fliplr(center_image))
                    angles.append(-center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
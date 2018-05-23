import csv
import numpy as np
from sklearn.utils import shuffle
import imageio

# accept all steering angles
def default_direction_filter(ang):
    return True, ang

# extract datasets from a list of `data_paths`
# L_AUG means if needs to apply left-camera images augmention
# R_AUG means if needs to apply right-camera images augmention
# corretion is the angle to be added or subtracted when applying augmention
# directionFilter is the function to filter data based on the steering angle
def extract_data(data_paths, L_AUG=True, R_AUG=True, correction=0.2, directionFilter=default_direction_filter):
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
                ang = float(line[3])
                is_add, ang_add = directionFilter(ang)
                if is_add:
                    samples.append(path + '/IMG/' + line[0].split('/')[-1])
                    steerings.append(ang_add)
                    if L_AUG is True:
                        samples_left.append(path + '/IMG/' + line[1].split('/')[-1])
                        steerings_left.append(ang_add + correction)
                    if R_AUG is True:
                        samples_right.append(path + '/IMG/' + line[2].split('/')[-1])
                        steerings_right.append(ang_add - correction)
                
        all_samples.extend(samples[0:-100])
        all_steerings.extend(steerings[0:-100])
        if L_AUG == True:
            all_samples.extend(samples_left[0:-100])
            all_steerings.extend(steerings_left[0:-100])
        if R_AUG == True:
            all_samples.extend(samples_right[0:-100])
            all_steerings.extend(steerings_right[0:-100])

    return all_samples, all_steerings

# sampel_generator generates training datasets
# samples and steering are input datasets
# Flip_AUG is whether to flip images to augment datasets
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
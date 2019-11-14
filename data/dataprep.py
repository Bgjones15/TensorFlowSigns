# Different approaches for image dimensionality
# One: take the average dimensions of all images and shrink/stretch them to fit
# Two: take the max dimension and fill in blankspace with a 0 pixel value
#
# If data is bad we can go back and pick ones with a minimum dimension, as those will probably
# be of better resolution

import os
import argparse
import math
import random
import shutil
from PIL import Image

def count_occluded(csv_path):
    return 0

def count_not_main_road(csv_path):
    return 0

def count_occluded_and_not_main_road(csv_path):
    return 0

def count_total(csv_path):
    return 0

def category_totals(csv_path):
    return 0

# Creates a new annotations csv that does not contain any images
# that were occluded or not on the main road
def create_filtered_csv(categories, exclude_occluded, exclude_not_main_road, orig_path, dest_path):
    categories = [c.lower() for c in categories]
    new_csv = open(dest_path, 'w+')
    orig_csv = open(orig_path, 'r')
    new_csv.write(orig_csv.readline()) # copy header over and ignore in iteration
    orig_csv_lines = orig_csv.readlines()
    orig_csv.close()

    # Don't worry about the previous file.. that's really only relevant when cropping which we can handle with a counter
    for line in orig_csv_lines:
        fields = line.split(";")
        category = fields[1].lower()
        occluded = fields[6].split(",")[0]
        not_main_road = fields[6].split(",")[1]

        if exclude_occluded and occluded == "1":
            continue

        if exclude_not_main_road and not_main_road == "1":
            continue

        if category in categories:
            new_csv.write(line)

    new_csv.close()

def avg_dimension(csv_lines):
    total_width = 0.0
    total_height = 0.0
    for line in csv_lines:
        fields = line.split(";")

        upper_left_x = fields[2]
        upper_left_y = fields[3]
        lower_right_x = fields[4]
        lower_right_y = fields[5]

        width = int(lower_right_x) - int(upper_left_x)
        height = int(lower_right_y) - int(upper_left_y)
        total_width += width
        total_height += height

    return (math.ceil(total_width / len(csv_lines)), math.ceil(total_height / len(csv_lines)))


dirname = os.path.dirname(__file__)
'''
# Exclude examples that are either not in the categories we want, occluded, or not on a main road
desired_categories = ['addedLane', 'keepRight', 'laneEnds', 'merge', 'pedestrianCrossing', 'school', 'signalAhead', 'stop', 'yield']
create_filtered_csv(desired_categories, True, True, os.path.join(dirname, 'allAnnotations.csv'), os.path.join(dirname, 'filteredAnnotations.csv'))

# Crop images, resize, and sort into folder by category
csv = open(os.path.join(dirname, 'filteredAnnotations.csv'), 'r')
header = csv.readline()
csv_lines = csv.readlines()
csv_lines.sort()
csv.close()

# Create new annotations csv that references the cropped, resized, and sorted images
if not os.path.exists(os.path.join(dirname, 'prepped')):
    os.mkdir('prepped')
savePath = os.path.join(dirname, 'prepped')
prepped_csv = open(os.path.join(savePath, 'preppedAnnotations.csv'), 'w+')
prepped_csv.write(header)

img = Image.new('RGB', (1,1))
prev_count = 2
prev_file_path = ''
normalized_dimension = avg_dimension(csv_lines)
for line in csv_lines:
    fields = line.split(";")
    relative_file_path = fields[0]
    category = fields[1]

    # Crop the sign from the image
    width = int(fields[4])-int(fields[2])
    height = int(fields[5])-int(fields[3])
    box = [int(fields[2])-width/100, int(fields[3])-height/100, int(fields[4])+width/100, int(fields[5])+height/100]

    img = Image.open(os.path.join(dirname, relative_file_path))
    img = img.crop(box)

    # Resize image (normalize dimension)
    img = img.resize(normalized_dimension)

    # Convert colored images into grayscale  (normalize pixels)
    img = img.convert('L')

    # Save the image to the appropriate location
    filename = os.path.basename(relative_file_path)
    # A single annotated frame may have multiple signs -- this accounts for that by
    # appending a counter to the end of the cropped sign image file name
    if relative_file_path == prev_file_path:
        filename_root = '.'.join(filename.split(".")[:-1])
        filename_ext = filename.split(".")[-1]
        filename = filename_root + "_" + str(prev_count) + "." + filename_ext
        prev_count += 1
    else:
        prev_count = 2
    prev_file_path = relative_file_path

    # Save the image to its category folder
    if not os.path.exists(os.path.join(savePath, category)):
        os.mkdir(os.path.join(savePath, category))
    dest_file_path = os.path.join(savePath, category, filename)
    img.save(dest_file_path)

    # Write new data to prepped csv to account for multiple signs in one frame
    prepped_line = os.path.join(category, filename) + ";" + category + "\n"
    prepped_csv.write(prepped_line)

prepped_csv.close()
img.close()'''

training_dir = os.path.join(dirname, "training")
validation_dir = os.path.join(dirname, "validation")
testing_dir = os.path.join(dirname, "testing")
if not os.path.exists(training_dir):
    os.mkdir(training_dir)
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
if not os.path.exists(testing_dir):
    os.mkdir(testing_dir)

category_dirs = os.listdir(os.path.join(dirname, "prepped"))
category_dirs = [d for d in category_dirs if os.path.isdir(os.path.join(dirname, "prepped", d))]
for d in category_dirs:
    examples = os.listdir(os.path.join(dirname, "prepped", d))
    examples = [e for e in examples if e.endswith(".png")]
    print(d + ' (total): ' + str(len(examples)))

    # Copy training (60% of total data)
    indices = random.sample(range(len(examples)), math.ceil(0.60*len(examples)))
    training = [examples[i] for i in indices]
    print(d + ' (training): ' + str(len(training)))
    for t in training:
        source = os.path.join(dirname, "prepped", d, t)
        if not os.path.exists(os.path.join(training_dir, d)):
            os.mkdir(os.path.join(training_dir, d))
        dest = os.path.join(training_dir, d, t)
        shutil.copyfile(source, dest)
        examples.remove(t)

    # Copy validation (50% of the remaing 40% = 20% of total)
    indices = random.sample(range(len(examples)), math.ceil(0.50*len(examples)))
    validation = [examples[i] for i in indices]
    print(d + ' (validation): ' + str(len(validation)))
    for v in validation:
        source = os.path.join(dirname, "prepped", d, v)
        if not os.path.exists(os.path.join(validation_dir, d)):
            os.mkdir(os.path.join(validation_dir, d))
        dest = os.path.join(validation_dir, d, v)
        shutil.copyfile(source, dest)
        examples.remove(v)

    # Copy testing (100% of the remaining 20% = 20% of total)
    testing = list(examples)
    print(d + ' (testing): ' + str(len(testing)))
    for t in testing:
        source = os.path.join(dirname, "prepped", d, t)
        if not os.path.exists(os.path.join(testing_dir, d)):
            os.mkdir(os.path.join(testing_dir, d))
        dest = os.path.join(testing_dir, d, t)
        shutil.copyfile(source, dest)
        examples.remove(t)

print('## Total file counts')
for c in os.listdir(training_dir):
    print(c + ' (training): ' + str(len(os.listdir(os.path.join(training_dir, c)))))
    print(c + ' (validation): ' + str(len(os.listdir(os.path.join(validation_dir, c)))))
    print(c + ' (testing): ' + str(len(os.listdir(os.path.join(testing_dir, c)))))

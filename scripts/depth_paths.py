import json, os
from PIL import Image
import numpy as np

#Incremental variance and mean calculation

# for a new value newValue, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far
def update(existingAggregate, newValue):
    newValue = np.array(newValue)
    (count, mean, M2) = existingAggregate
    count += np.prod(np.size(newValue))
    delta = newValue - mean
    mean += np.sum(delta) / count
    M2 += np.sum(np.multiply(newValue, newValue))

    return (count, mean, M2)

# retrieve the mean, variance and sample variance from an aggregate
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1))
    if count < 2:
        return float('nan')
    else:
        return (mean, variance, sampleVariance)

if __name__ == "__main__":

    with open('datasets/sunspot/annotations/instances.json', 'r') as f:
        sunrgbd = json.load(f)

    existingAggregate = (0, 0, 0)
    paths = {}
    mean = []

    i =0
    for im in sunrgbd['images']:
        idx = im['id']
        file_name = im['file_name']
        depth_dir = os.path.join(file_name.split('image')[0], 'depth_bfx/')
        depth_file = os.listdir(os.path.join('datasets/SUNRGBD/images', depth_dir))
        paths[idx] = os.path.join(depth_dir, depth_file[0])

        depth = Image.open(os.path.join('datasets/SUNRGBD/images', paths[idx]))
        existingAggregate = update(existingAggregate, depth)
        if i % 100 == 0:
            print(finalize(existingAggregate))
        i = i +1

    final = finalize(existingAggregate)
    print("Mean depth: {}".format(final[0]))
    print("Std depth: {}".format(np.sqrt(final[1])))

    with open('datasets/sunspot/annotations/depth.json', 'w') as f:
        json.dump(paths, f)

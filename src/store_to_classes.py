import numpy as np
import tensorflow as tf
from os.path import exists

from parse_binary import samples_from_binary

# Check which devices visible
from tensorflow.python.client import device_lib
print("Devices available:", device_lib.list_local_devices())

# Folders for storage/retrival
data_directory = '../data/'
binaries_folder = data_directory + 'binaries/'
records_folder  = data_directory + 'class_records/'
dataset_folder  = data_directory + 'dataset/'

class_names = [] # 345 classes
with open(data_directory + 'classnames.csv', 'r') as cln_file:
    for line in cln_file:
        class_names += [line[:-1]]

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def store_to_tfrecord(class_name):
    print("Storing class {} to tfrecord".format(class_name))
    source_name = binaries_folder + class_name + '.bin'
    target_name = records_folder + class_name + '.tfrecords'
    if not exists(source_name):
        print(source_name, "not found")
        return
    
    samples = samples_from_binary(class_name, with_imgs=False)
    with tf.python_io.TFRecordWriter(target_name) as writer:
        while True:
            try:
                sketch = next(samples)
            except StopIteration:
                break

            sketch_flat = np.reshape(sketch, [-1])
            label = bytes(class_name, 'utf-8')

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'sketch': _int64_feature(sketch_flat), # [num_strokes*3]  
                        'label': _bytes_feature([label])  
                    }))
            writer.write(example.SerializeToString())
            
        
# In parallel write the tfrecords
from joblib import Parallel, delayed
Parallel(n_jobs=10)(map(delayed(store_to_tfrecord), class_names))

# In sequence
# for cln in class_names:
#     store_to_tfrecord(cln)
import numpy as np
import tensorflow as tf

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=""

# Check which devices visible
from tensorflow.python.client import device_lib
print("Devices available:", device_lib.list_local_devices())

# Folders for storage/retrival
main_directory  = '../'
data_directory  = main_directory + 'data/'
binaries_folder = data_directory + 'binaries/'
records_folder  = data_directory + 'class_records/'
dataset_folder  = data_directory + 'dataset/'
train_folder    = data_directory + 'train/'
eval_folder     = data_directory + 'eval/'
test_folder     = data_directory + 'test/'

class_names = [] # 345 classes
with open(data_directory + 'classnames.csv', 'r') as cln_file:
    for line in cln_file:
        class_names += [line[:-1]]

# Create necessary folders
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(eval_folder):
    os.makedirs(eval_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# Helper fcts for parsing and storing data
def _parse_tfexample_fn(example_proto):
    tfrecord_features = tf.parse_single_example(
        example_proto,
        features={
            'sketch': tf.VarLenFeature(dtype=tf.int64),
            'label': tf.FixedLenFeature([], dtype=tf.string)
        }, name='features')

    # Recreate strokes
    sketch = tf.sparse_tensor_to_dense(tfrecord_features['sketch'])
    sketch = tf.reshape(sketch, shape=[-1,3])
    label = tfrecord_features['label']
    
    return sketch, label

def _get_data(file_pattern):
    dataset = tf.data.Dataset.list_files(file_pattern=file_pattern)
    dataset = dataset.interleave( 
        lambda x: tf.data.TFRecordDataset(x).shuffle(200000), # shuffle data
        cycle_length=345, # Interleave from all classes
        block_length=1)   # 1 at a time
    dataset = dataset.map( # 
        _parse_tfexample_fn,
        num_parallel_calls=1 # No speed up found, as image creation is bottleneck
    )
    dataset = dataset.prefetch(10)
    sketch_var, label_var =  dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        while True:
            try: 
                sketch_out, label_out = sess.run([sketch_var, label_var])
                yield sketch_out, label_out
            except tf.errors.OutOfRangeError:
                break;
# Store part
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
            
# With help of generator, store amount of samples to a file
def _store(target_name, samples, counts=None):
    with tf.python_io.TFRecordWriter(target_name) as writer:
        while (counts == None) or (counts > 0):
            try:
                sketch, label = next(samples)
            except StopIteration:
                break

            sketch_flat = np.reshape(sketch, [-1])
#            label = bytes(class_name, 'utf-8')

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'sketch': _int64_feature(sketch_flat), # [num_strokes*3]  
                        'label': _bytes_feature([label])  
                    }))
            writer.write(example.SerializeToString())
            
            if counts != None:
                counts -= 1

def store_in_splits(class_name):
    print("Storing class {} to tfrecord".format(class_name))
    source_name = records_folder + class_name + '.tfrecords'
    train_target = train_folder + class_name + '.tfrecords'
    test_target  = test_folder  + class_name + '.tfrecords'
    eval_target  = eval_folder  + class_name + '.tfrecords'
    
    if not os.path.exists(source_name):
        print(source_name, "not found")
        return
    
    samples = _get_data(source_name)
    _store(eval_target, samples, 5000)     # Validation data
    _store(test_target, samples, 5000)     # test data
    _store(train_target, samples) 	   # Rest goes to train


# Execute in parallel
from joblib import Parallel, delayed
Parallel(n_jobs=10)(map(delayed(store_in_splits), class_names))


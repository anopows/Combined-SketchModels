import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
import cv2 as cv
import glob

# Folders for storage/retrival
main_directory  = '../'
tensorboard_directory = main_directory + 'tb_graphs/'
checkpoints_directory = main_directory + 'checkpts/'
data_directory  = main_directory + 'data/'
binaries_folder = data_directory + 'binaries/'
records_folder  = data_directory + 'class_records/'
dataset_folder  = data_directory + 'dataset/'
train_folder    = data_directory + 'train/'
eval_folder     = data_directory + 'eval/'
test_folder     = data_directory + 'test/'
train_folder_small    = data_directory + 'train_small/'
eval_folder_small     = data_directory + 'eval_small/'
test_folder_small     = data_directory + 'test_small/'

def _populate_folders(main_directory):
    tensorboard_directory = main_directory + 'tb_graphs/'
    checkpoints_directory = main_directory + 'checkpts/'
    data_directory        = main_directory + 'data/'
    binaries_folder       = data_directory + 'binaries/'
    records_folder        = data_directory + 'class_records/'
    dataset_folder        = data_directory + 'dataset/'
    train_folder          = data_directory + 'train/'
    eval_folder           = data_directory + 'eval/'
    test_folder           = data_directory + 'test/'
    train_folder_small    = data_directory + 'train_small/'
    eval_folder_small     = data_directory + 'eval_small/'
    test_folder_small     = data_directory + 'test_small/'
    
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
    length = tf.shape(sketch)[0]
    label = tfrecord_features['label']
    
    return length, sketch, label


def _create_img(length, sketch, imgmode=None, imgperc=None, 
                imgsize = (255,255), output_size = (224,224), border = (5,5), thickness=3):
    """Create an image from given lines

    Arguments:
    length:  number of non-padded pts
    sketch:  array of pts (x,y,end), where end denotes end of a stroke
    imgmode: None (whole image), 'atperc' (until imgperc% of image), 'snapshots' (at snapshots)
    imgsize: output size of an image (default (255,255))

    Assume point coordinates are between 0 and 255
    """
    if imgmode in [None, 'snapshots', 'middle_last']:
        sketch = sketch[:length]
    elif imgmode == 'atperc':
        assert imgperc >= 0 and imgperc <= 1
        length = int(imgperc*length - 1)
        sketch = sketch[:length]
        sketch[-1,2] = 1 # last point is end
    else:
        raise Exception("'{}' image mode not implemented".format(imgmode))
        
    # white image
    imgsizex, imgsizey = (imgsize[0] + 2*border[0], imgsize[1] + 2*border[1])
    img = 255*np.ones((imgsizey,imgsizex), np.uint8) # for opencv first y-coord, then x-coord

    split_indices = sketch[:,2] == 1 # note where we have 1s
    split_indices = np.where(split_indices)[0][:-1] # convert to indices, drop last one(end)
    split_indices += 1 # split one after end token
    sketch = sketch[:,:2] + np.array(border)
    lines = np.split(sketch[:,:2], split_indices)
    
    if imgmode == 'snapshots':
        results = []
        for line in lines:
            cv.polylines(img, [line], False, (0,0,0), thickness, 32)
            # Scale to output image size, also smooths it
            img_transformed = cv.resize(img, output_size, interpolation = cv.INTER_CUBIC)
            results.append(img_transformed)
        imgs = np.stack(results)
        return imgs.reshape((-1, output_size[0]*output_size[1]))
    elif imgmode == 'middle_last':
        imgs = np.empty((2, output_size[0] * output_size[1]), dtype=np.uint8)
        count_els = 0
        over_half = False
        for line in lines: 
            num_els = line.shape[0]
            if (not over_half) and (count_els + num_els >= length // 2): # if half element occurs
                over_half = True
                num_to_take = (length // 2) - count_els
                img_half = img
                cv.polylines(img_half, [line[:num_to_take]], False, (0,0,0), thickness, 32)
                # Scale to output image size, also smooths it
                img_half = cv.resize(img_half, output_size, interpolation = cv.INTER_CUBIC)
                imgs[0] = img_half.reshape((224*224))
                
            cv.polylines(img, [line], False, (0,0,0), thickness, 32)
            count_els += num_els
        img = cv.resize(img, output_size, interpolation = cv.INTER_CUBIC)
        imgs[1] = img.reshape((224*224))
        return imgs
    else:
        for line in lines:
            cv.polylines(img, [line], False, (0,0,0), thickness, 32)
            
        # Scale to output image size, also smooths it
        img = cv.resize(img, output_size, interpolation = cv.INTER_CUBIC)
        return img.reshape(-1)

def _filter_triplet_fn(length, sketch, label):
    # label[0], label[1] must be same
    same_01 = math_ops.equal(label[0], label[1])
    # label[2], label[3] must be same
    same_23 = math_ops.equal(label[2], label[3])
    # But first 2 should be different than second 2
    different =  math_ops.not_equal(label[0], label[2])
    
    same = math_ops.logical_and(same_01, same_23)
    return math_ops.logical_and(same, different)

def _filter_threesplit_fn(length, sketch, label):
    num_per_class = tf.shape(label)[0]/3
    num_per_class = tf.cast(num_per_class, tf.int32)
    
    typeA = label[:num_per_class]
    typeB = label[num_per_class:2*num_per_class]
    typeC = label[2*num_per_class:]
    
    # All classes are really same
    sameA = tf.equal(typeA, typeA[0])
    sameB = tf.equal(typeB, typeB[0])
    sameC = tf.equal(typeC, typeC[0])
    same = tf.concat([sameA, sameB, sameC], axis=0)
    same = tf.reduce_all(same)
    
    # But between are different
    diff1 = tf.not_equal(typeA[0], typeB[0])
    diff2 = tf.not_equal(typeB[0], typeC[0])
    diff3 = tf.not_equal(typeA[0], typeC[0])
    diff = tf.logical_and(diff1, diff2)
    diff = tf.logical_and(diff,  diff3)
    
    return tf.logical_and(same, diff)

def _filter_max_snaphots_fn(max_avg_snapshots, batch_size):
    # Create filter based on batch size and snapshot limit information
    def _filter_max_snaphots(length, sketch, label):
        num_snapshots = tf.reduce_sum(sketch[:,:,2], axis=None)
        avg_snapshots  = num_snapshots / batch_size
        return avg_snapshots < max_avg_snapshots

    return _filter_max_snaphots

def _only_first3(length, sketch, label):
    return length[:3], sketch[:3], label[:3]

def _img_fn(imgmode=None, imgperc=None):
    # Parameterized image creation function: Create full img/ at certain % img/ snapshot imgs 
    create_img_fn = lambda length, sketch: _create_img(length, sketch, imgmode=imgmode, imgperc=imgperc)
    return lambda length, sketch, label: \
                (length, sketch, tf.py_func(create_img_fn, [length, sketch], tf.uint8, stateful=False), label)

def _get_int_labels(labels, small):
    with tf.variable_scope('label_mapping', reuse=tf.AUTO_REUSE):
        if not small:
            class_names = [] # 345 classes
            with open(data_directory + 'classnames.csv', 'r') as cln_file:
                for line in cln_file:
                    class_names += [line[:-1]]
        else:
            class_names = [] # 5 fruits, 5 vehicles
            with open(data_directory + 'classnames_small.csv', 'r') as cln_file:
                for line in cln_file:
                    class_names += [line[:-1]]
        # label name -> id
        mapping_strings = tf.get_variable('class_names', shape=[len(class_names)], dtype=tf.string, 
                                          initializer=tf.constant_initializer(class_names),
                                          trainable=False
                          )
        table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings)
        ids  = table.lookup(labels)
        return ids

def train_data(batch_size=30, num_prefetch=2, int_labels=False,
               epochs=None, mode=None, imgmode=None, imgperc=None, small=False, max_avg_snapshots=None,
               seed=None, base_folder=None):   
    """ Return batches of training data
    
    Returns:
        length
        sketch
        image
        label
    """
    if small: 
        file_pattern = train_folder_small + '*.tfrecords'
    else:
        file_pattern = train_folder + '*.tfrecords'
    return get_data(file_pattern, batch_size=batch_size, int_labels=int_labels, small=small, 
                    num_prefetch=num_prefetch, epochs=epochs, mode=mode, imgmode=imgmode, imgperc=imgperc, 
                    max_avg_snapshots=max_avg_snapshots,
                    seed=seed, base_folder=base_folder)

def eval_data(batch_size=30, num_prefetch=2, int_labels=False, 
              epochs=1, mode=None, imgmode=None, imgperc=None, small=False, max_avg_snapshots=None,
              seed=42, base_folder=None):
    """ Return batches of validation data
    
    Returns:
        length
        sketch
        image
        label
    """
    if small: 
        file_pattern = eval_folder_small + '*.tfrecords'
    else:
        file_pattern = eval_folder + '*.tfrecords'
    return get_data(file_pattern, batch_size=batch_size, int_labels=int_labels, small=small, 
                    num_prefetch=num_prefetch, epochs=epochs, mode=mode, imgmode=imgmode, imgperc=imgperc, 
                    max_avg_snapshots=max_avg_snapshots,
                    seed=seed, base_folder=base_folder)
                   
def test_data(batch_size=30, num_prefetch=2, int_labels=False, 
              epochs=1, mode=None, imgmode=None, imgperc=None, small=False, max_avg_snapshots=None,
              seed=42, base_folder=None):
    """ Return batches of test data
    
    Returns:
        length
        sketch
        image
        label
    """
    if small: 
        file_pattern = test_folder_small + '*.tfrecords'
    else:
        file_pattern = test_folder + '*.tfrecords'
    return get_data(file_pattern, batch_size=batch_size, int_labels=int_labels, small=small,
                    num_prefetch=num_prefetch, epochs=epochs, mode=mode, imgmode=imgmode, imgperc=imgperc, 
                    max_avg_snapshots=max_avg_snapshots,
                    seed=seed, base_folder=base_folder)
    

def get_data(file_pattern=records_folder + '*.tfrecords', 
             batch_size=30, 
             num_prefetch=2, 
             int_labels=False,
             small=True,
             epochs=None,
             mode=None,
             imgmode=None,
             imgperc=None,
             max_avg_snapshots=None,
             seed=None,
             base_folder=None):
    """
    mode: 
        - None:         batch of [length, sketch, image, label] samples
        - 'triplets':   batch of 3 samples [length, sketch, image, label] where only first two are of same label
        - 'threesplit': batch of [length, sketch, image, label] samples. Inside three splits of same classes. 
                         Precondition: bs%3==0
    imgmode:
        - None:         Return last image
        - 'atperc'      Return image at specified percentage of sketch. Specified by imgperc argument
        - 'snapshots'   Return images at snapshots
        - 'middle_last' Return images at 50% and 100% progress
    """
    if base_folder: _populate_folders(base_folder)

    with tf.variable_scope('input_pipepline'):
        
#         # TF 1.9 dependent
#         num_datasets = len(glob.glob(file_pattern))
#         assert num_datasets > 0

#         choices  = tf.data.Dataset.range(num_datasets).repeat().shuffle(5)
#         datasets = [tf.data.TFRecordDataset(tfrecord_file).shuffle(10000).repeat(epochs) for tfrecord_file in glob.glob(file_pattern)]
#         dataset = tf.contrib.data.choose_from_datasets(datasets, choices)
        
        # For TF <= 1.8
        dataset = tf.data.Dataset.list_files(file_pattern=file_pattern, shuffle=False) # TF 1.9: shuffle: True but with seed
        num_datasets = len(glob.glob(file_pattern))
        assert num_datasets > 0

        block_length = 1
        if mode=='triplets': block_length = 2 # group AABB --> AAB
        elif mode=='threesplit': 
            assert(batch_size%3 == 0)
            block_length = batch_size//3
        elif mode:
            raise Exception("{} input feeding mode is not implemented".format(mode))

        # Have num_datasets tfrecords/classes and take 2 entries from each class at a time.
        # Also shuffle datasets before so no same 2 values are taken in later epoch
        dataset = dataset.interleave( 
            lambda x: tf.data.TFRecordDataset(x).repeat(epochs).shuffle(5000, seed=seed), # 
            cycle_length=num_datasets, # Interleave from all classes
            block_length=block_length)   # same classes at a time
        dataset = dataset.map( # 
            _parse_tfexample_fn,
            num_parallel_calls=1 # No speed up found, as image creation is bottleneck
        )

        # dataset now queue of [length, sketch, label]

        if mode=='triplets':
            # always group by 2 elements of same class
            dataset = dataset.padded_batch(batch_size=2, padded_shapes=([], [None,3], [])) 
            dataset = dataset.shuffle(300, seed=seed) # shuffle classes around, so not always next to each other
            # Group by 4. First class != second class. And throw 4th element away. so have classes i,i,j
            dataset = dataset.apply(tf.contrib.data.unbatch())
            dataset = dataset.padded_batch(batch_size=4, padded_shapes=([], [None,3], []))
            # Filter elements, so for a batch have classes iijj for i!=j
            dataset = dataset.filter(_filter_triplet_fn)
            # Throw away 4th element
            dataset = dataset.map(_only_first3, num_parallel_calls=1)
        elif mode=='threesplit':
            # Keep classes together for shuffling
            dataset = dataset.padded_batch(batch_size=block_length, padded_shapes=([], [None,3],[])) 
            dataset = dataset.shuffle(300, seed=seed) # which classes mixed with which
            dataset = dataset.apply(tf.contrib.data.unbatch())
            dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=([], [None,3], []))
            dataset = dataset.filter(_filter_threesplit_fn)

        dataset = dataset.shuffle(5000, seed=seed)

        # Filter out batches with too many snapshots: Bc of possible blow up of GPU memory
        if max_avg_snapshots:
            if not mode is None: raise Exception("Limiting snapshots per batch only implemented for simple batches")
            # Want to consider statistics per batch
            dataset = dataset.padded_batch(batch_size, ([], [None, 3], []))
            # Per batch over all sequences
            dataset = dataset.filter(_filter_max_snaphots_fn(max_avg_snapshots, batch_size)) 
            dataset = dataset.apply(tf.contrib.data.unbatch())

        # create images based on sketch data, unbatch so can create image one by one
        if mode:
            dataset = dataset.apply(tf.contrib.data.unbatch())
        dataset = dataset.map(_img_fn(imgmode=imgmode, imgperc=imgperc), num_parallel_calls=4)

        imgshape = [224*224]
        if imgmode == 'snapshots': # Have multiple images to return
            imgshape = [None] + imgshape 
        elif imgmode == 'middle_last':
            imgshape = [2] + imgshape
        # batch again
        if mode=='triplets':
            if imgmode in ['snapshots', 'middle_last']:
                dataset = dataset.padded_batch(3, ([], [None, 3], imgshape, []))
            else:
                dataset = dataset.batch(3) # can use normal batch because, already padded last time
                
            # Have batches of 3 of (length, sketch, label)
            dataset = dataset.padded_batch(batch_size, ([3], [3, None, 3], [3] + imgshape, [3])) 
        elif mode=='threesplit':
            if imgmode in ['snapshots', 'middle_last']:
                dataset = dataset.padded_batch(batch_size, ([], [None, 3], imgshape, []))
            else:
                dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.padded_batch(batch_size, ([],  [None, 3],    imgshape,   []))


        dataset = dataset.prefetch(num_prefetch)

        *args, labels = dataset.make_one_shot_iterator().get_next()

        if int_labels:
            labels = _get_int_labels(labels, small=small)
        return args + [labels] # (length, sketch, image, label)

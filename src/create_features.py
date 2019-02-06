import numpy as np
import tensorflow as tf
import os

from utility import in_jupyter, limit_cpu
from input_pipeline import train_data, eval_data
from model_fns import ConvNet
from model_fns import classification


# Folders for storage/retrival
main_directory  = '../'
checkpoints_directory = main_directory + 'checkpts/'
features_directory    = main_directory + 'features/'

# Feature choices
num_samples = 100000
num_eval    =   4000
batch_size  =     20

def write_to_file():
    lengths, sketches, images, labels                     = train_data(batch_size=batch_size, epochs=1, mode=None, small=True)
    eval_lengths, eval_sketches, eval_images, eval_labels = eval_data (batch_size=batch_size, epochs=1, mode=None, small=True)

    if not os.path.exists(features_directory):
        os.makedirs(features_directory)
    
    nplengths  = np.empty((0),         dtype=np.int32)
#      npsketches = np.zeros((0,224*224), dtype=np.uint8)
    sketches_list = []
    nplabels   = np.empty((0),         dtype=bytes)
    npimages   = np.zeros((0,224*224), dtype=np.uint8)
    
    npvlengths  = np.empty((0),         dtype=np.int32)
#     npvsketches = np.zeros((0,224*224), dtype=np.uint8)
    vsketches_list = []
    npvlabels   = np.empty((0),         dtype=bytes)
    npvimages   = np.zeros((0,224*224), dtype=np.uint8)
    
    with tf.Session() as sess: 
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
          
        num_runs = (num_samples // batch_size) + 1
        for i in range(num_runs):
            if (i+1) % 5 == 0: print("Step {} from {}".format(i+1, num_runs))
            num_to_take = batch_size
            if i+1 == num_runs: num_to_take = num_samples - i*batch_size
            if num_to_take == 0: break
            
                
            lengths_out, sketches_out, labels_out, images_out = \
                sess.run([lengths, sketches, labels,images])
            nplengths  = np.append(nplengths,  lengths_out[:num_to_take],  axis=0)
            sketches_list.append(sketches_out[:num_to_take])
            nplabels   = np.append(nplabels,   labels_out[:num_to_take],   axis=0)
            npimages   = np.append(npimages,   images_out[:num_to_take],   axis=0)
            
        # create one big np array. Pad based on largest sketch with length maxlen
        maxlen = max([sk.shape[1] for sk in sketches_list])
        npsketches = np.zeros(shape=(len(nplengths), maxlen, 3)) # num samples x biggest sketch lenght x 3
        for i,sk in enumerate(sketches_list):
            num   = sk.shape[0]
            sklen = sk.shape[1]
            npsketches[i*batch_size:i*batch_size+num, :sklen] = sk
                      
        np.save(features_directory + 'lengths',  nplengths)
        np.save(features_directory + 'sketches', npsketches)
        np.save(features_directory + 'labels',   nplabels)
        np.save(features_directory + 'images',   npimages)
        
        num_runs_eval = (num_eval // batch_size) + 1
        for i in range(num_runs_eval):
            if (i+1) % 5 == 0: print("Step {} from {}".format(i+1, num_runs_eval))
            num_to_take = batch_size
            if i+1 == num_runs_eval: num_to_take = num_eval - i*batch_size
            if num_to_take == 0: break
                
            vlengths_out, vsketches_out, vlabels_out, vimages_out = \
                sess.run([eval_lengths, eval_sketches, eval_labels, eval_images])
            npvlengths  = np.append(npvlengths,  vlengths_out[:num_to_take],  axis=0)
            vsketches_list.append(vsketches_out[:num_to_take])
            npvlabels   = np.append(npvlabels,   vlabels_out[:num_to_take],   axis=0)
            npvimages   = np.append(npvimages,   vimages_out[:num_to_take],   axis=0)
            
        # create one big np array. Pad based on largest sketch with length maxlen
        maxlen = max([sk.shape[1] for sk in vsketches_list])
        npvsketches = np.zeros(shape=(len(npvlengths), maxlen, 3)) # num samples x biggest sketch lenght x 3
        for i,sk in enumerate(vsketches_list):
            num   = sk.shape[0]
            sklen = sk.shape[1]
            npvsketches[i*batch_size:i*batch_size+num, :sklen] = sk

        np.save(features_directory + 'eval_lengths',  npvlengths)
        np.save(features_directory + 'eval_sketches', npvsketches)
        np.save(features_directory + 'eval_labels',   npvlabels)
        np.save(features_directory + 'eval_images',   npvimages)  

def _cnn_model(images):
    flags_cnn_model_choice  = 'resnet'      # Choice of Convolutional Model (resnet/inception/vgg)
    flags_feed_mode         = 'threesplit'  # How to feed the model (threesplit/triplets/None)
    flags_loss_mode         = 'hinge'       # Loss: hinge/exp
    flags_units_CNN         = 256           # CNN  state size
    flags_batch_size        = 30            # Batch size
    flags_annealing         = 0             # Half learn rate every x steps. No annealing: 0 (default)
    flags_learning_rate     = 0.0001        # Learning rate for CNN weights
    flags_hinge_loss_alpha  = 1             # Distance for hinge loss hyper parameter
    loss_mode_params  = {'alpha': flags_hinge_loss_alpha}
    
    # Further parameters
    sgd_switch = None # After x steps, switch to SGD optimization with decay
    sgd_learn_rate   = 1e-3
    
    #########
    # Model #
    #########
    
    cnn_model = ConvNet(model=flags_cnn_model_choice, feed_mode=flags_feed_mode, embedding_size=flags_units_CNN,
                        loss_mode=flags_loss_mode, loss_mode_params=loss_mode_params)
    # Classifier part
    logits      = cnn_model.logits(images,      training=False, feed_mode=None, stop_gradient=True) # Training false for BNorm
    
    return cnn_model.model_name, logits

def _lstm_model(sketch, length):
    # Parse command line arguments
    flags_hiddensize = 256 # LSTM state size
    flags_lstm_stack = 2   # How many LSTMs to stack at every layer
    flags_normalize = 'diff'  # Calculate standard deviations for positions(pos)/differences(diff)
    flags_bnorm_before  = False # Use batch normalization before classification layers
    flags_bnorm_middle = False # Use batch normalization in between classification layers
    flags_bnorm_eval_update = False # Update statistics in validating
    flags_used_steps = 'snapshots'  # Use all/allAndPad/last/snapshots hidden states of the LSTM for training
    flags_prefix = 'LSTMv5' # Model name prefix
    
    # Model choices
    batch_size     = tf.shape(sketch)[0]
    embedding_size = 64
    hidden_size    = flags_hiddensize
    clip_gradient  = 1
    stacked_LSTMs  = flags_lstm_stack
    num_steps      = 50000
    differences    = True
    normalize      = flags_normalize
    bnorm_before   = flags_bnorm_before
    bnorm_middle   = flags_bnorm_middle
    
    model_name     = '{}_hs{}x{}'.format(flags_prefix, stacked_LSTMs, hidden_size)
    if normalize:         model_name += '_' + flags_normalize + 'Normalized'
    if bnorm_before:      model_name += '_bn-before'
    if bnorm_middle:      model_name += '_bn-middle'
    if flags_bnorm_eval_update :     model_name += '_bnEvalUpdate'
    model_name += '_usedSteps_' + flags_used_steps
    
    ##################
    # Pre-processing #
    ##################
    
    def normalize_positions(sketch):
        # De-mean and calculate how many std devs away
        mean_pos = tf.constant([113.6, 107.9])
        std_pos  = tf.constant([ 69.2,  63.4])
        sketch_vals = (sketch[:, :, 0:2] - mean_pos) / std_pos
        sketch_vals = tf.where(tf.not_equal(sketch[:,:,:2],0), # Only keep non-padded values
                               sketch_vals,
                               tf.zeros(tf.shape(sketch_vals))
                      )
        return tf.concat([sketch_vals, sketch[:,:,2:]], axis=2)
    
    def normalize_movement(sketch_diff, lengths):
        mean    = tf.constant([ 0.04, -0.67])
        std_dev = tf.constant([34.53, 26.54])
        
        vals = (sketch_diff[:,:,:2] - mean) / std_dev
        vals = mask_sketch(vals, lengths)
        return tf.concat([vals, sketch_diff[:,:,2:]], axis=2)
        
    def mask_sketch(sketches, lengths):
        def repl(x, max_len):
            ones  = tf.ones(x, dtype=tf.int32)
            zeros = tf.zeros(max_len-x, dtype=tf.int32)
            return tf.concat([ones,zeros], axis=0)
    
        max_len = tf.shape(sketches)[1]
        mask = tf.map_fn(lambda x: repl(x, max_len), lengths)
        mask_float = tf.cast(mask, tf.float32)
        return tf.expand_dims(mask_float,2) * sketches
    
    def calc_diffs(sketches, lengths):
        sketches = tf.cast(sketches, tf.float32)
        batch_size  = tf.shape(sketches)[0]
        sketch_vals = sketches[:, :, 0:2]
    
        # add (122.5,122.5) pts as first pt. To preserve starting point when calculating diffs
        first_rows  = 122.5 * tf.zeros(tf.stack([batch_size, 1, 2])) 
        sketch_vals = tf.concat([first_rows, sketch_vals], axis=1) 
        # Difference calculation
        sketch_vals = sketch_vals[:, 1:, 0:2] - sketch_vals[:, 0:-1, 0:2]
        # Throw away the last row of difference when it should be padding
        sketch_vals = mask_sketch(sketch_vals, lengths)
        return tf.concat([sketch_vals, sketches[:,:,2:]], axis=2)
    
    with tf.variable_scope('preprocessing', reuse=tf.AUTO_REUSE):
        sketch      = tf.cast(sketch,      tf.float32)
    
        if normalize == 'pos':
            sketch      = normalize_positions(sketch)
    
        if differences:
            sketch      = calc_diffs(sketch, length)
    
            if normalize == 'dist': 
                sketch      = normalize_movement(sketch, length)
    
    
    ##############
    # LSTM Model #
    ##############
    
    with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):   
        rnn_cell   = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(hidden_size) for _ in range(stacked_LSTMs)])
        init_state = rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    
        lstm_out, _ = tf.nn.dynamic_rnn(rnn_cell, sketch,
                                        initial_state=init_state,
                                        dtype=tf.float32,
                                        scope='lstm_dynamic'
                              )
    
    
    # Take the last logit of sketch
    indices = tf.stack([tf.range(batch_size), length-1], axis=1)    
    logits = tf.gather_nd(lstm_out, indices)
    
    return model_name, logits

def _combined_model(images, sketch, length):
    # Parse command line arguments, model parameters
    if in_jupyter(): 
        tf.app.flags.DEFINE_string('f', '', 'kernel')
    flags_units_LSTM = 256 # LSTM state size
    flags_units_CNN = 256 # CNN  state size
    flags_es_lstm = 0 # O: take LSTM output. Otherwise: MatMul to desired embedding_size
    flags_lstm_stack = 2 # How many LSTMs to stack at every layer
    flags_batch_size = 20 # Batch size
    flags_num_steps = 50000 # Steps of training
    flags_middleRepr = False # Use middle representations of CNN
    flags_trainCNN = True # Training CNN weights or keeping them fixed
    flags_pretrainedCNN = True # Use a pretrained CNN weights
    
    # Most likely fixed model choices
    flags_normalize = 'diff' # Calculate standard deviations for positions(pos)/differences(diff)
    flags_bnorm_before =  False # Use batch normalization before classification layers
    flags_bnorm_middle =  False # Use batch normalization in between classification layers
    flags_bnorm_eval_update =  False # Update statistics in validating
    flags_used_steps =    'snapshots' # Use all/allAndPad/last/snapshots hidden states of the LSTM for training
    flags_prefix =        'CombinedModel' # Model name prefix
    
    # Fixed Model choices
    embedding_size = 64
    clip_gradient  = 1
    differences    = True

    model_name     = '{}_unitsLSTM{}x{}_unitsCNN{}_bs{}'.format(
        flags_prefix, flags_lstm_stack, flags_units_LSTM, flags_units_CNN, flags_batch_size)
    
    model_name += '_usedSteps-' + flags_used_steps
    if flags_pretrainedCNN: model_name += '_pretrainedCNN'
    if flags_trainCNN:      model_name += '_trainCNN'
    if flags_es_lstm:       model_name += '_es-lstm' + str(flags_es_lstm)
    if flags_middleRepr:    model_name += '_middleRepr'

    def normalize_positions(sketch):
        # De-mean and calculate how many std devs away
        mean_pos = tf.constant([113.6, 107.9])
        std_pos  = tf.constant([ 69.2,  63.4])
        sketch_vals = (sketch[:, :, 0:2] - mean_pos) / std_pos
        sketch_vals = tf.where(tf.not_equal(sketch[:,:,:2],0), # Only keep non-padded values
                            sketch_vals,
                            tf.zeros(tf.shape(sketch_vals))
                    )
        return tf.concat([sketch_vals, sketch[:,:,2:]], axis=2)

    def normalize_movement(sketch_diff, lengths):
        mean    = tf.constant([ 0.04, -0.67])
        std_dev = tf.constant([34.53, 26.54])
        
        vals = (sketch_diff[:,:,:2] - mean) / std_dev
        vals = mask_sketch(vals, lengths)
        return tf.concat([vals, sketch_diff[:,:,2:]], axis=2)
        
    def mask_sketch(sketches, lengths):
        def repl(x, max_len):
            ones  = tf.ones(x, dtype=tf.int32)
            zeros = tf.zeros(max_len-x, dtype=tf.int32)
            return tf.concat([ones,zeros], axis=0)
    
        max_len = tf.shape(sketches)[1]
        mask = tf.map_fn(lambda x: repl(x, max_len), lengths)
        mask_float = tf.cast(mask, tf.float32)
        return tf.expand_dims(mask_float,2) * sketches

    def calc_diffs(sketches, lengths):
        sketches = tf.cast(sketches, tf.float32)
        batch_size  = tf.shape(sketches)[0]
        sketch_vals = sketches[:, :, 0:2]

        # add (122.5,122.5) pts as first pt. To preserve starting point when calculating diffs
        first_rows  = 122.5 * tf.ones(tf.stack([batch_size, 1, 2])) 
        sketch_vals = tf.concat([first_rows, sketch_vals], axis=1) 
        # Difference calculation
        sketch_vals = sketch_vals[:, 1:, 0:2] - sketch_vals[:, 0:-1, 0:2]
        # Throw away the last row of difference when it should be padding
        sketch_vals = mask_sketch(sketch_vals, lengths)
        return tf.concat([sketch_vals, sketches[:,:,2:]], axis=2)

    with tf.variable_scope('preprocessing', reuse=tf.AUTO_REUSE):
        sketch      = tf.cast(sketch,      tf.float32)

        if flags_normalize == 'pos':
            sketch      = normalize_positions(sketch)

        if differences:
            sketch      = calc_diffs(sketch, length)

            if flags_normalize == 'dist': 
                sketch      = normalize_movement(sketch, length)


    ##############
    # LSTM Model #
    ##############

    with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):   
        rnn_cell   = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(flags_units_LSTM) for _ in range(flags_lstm_stack)])
        init_state = rnn_cell.zero_state(batch_size=flags_batch_size, dtype=tf.float32)

        lstm_out, _ = tf.nn.dynamic_rnn(rnn_cell, sketch,
                                        initial_state=init_state,
                                        dtype=tf.float32,
                                        scope='lstm_dynamic'
                            )
        
        if flags_es_lstm:
            lstm_out      = tf.layers.dense(lstm_out, flags_es_lstm)


    ########################################
    # Adapt LSTM states/labels to snapshot #
    ########################################

    # Calculate indices where snapshot lie
    last_indices = tf.stack([tf.range(flags_batch_size), length-1], axis=1)
    lstm_logits  = tf.gather_nd(lstm_out, last_indices)

    #############
    # CNN Model #
    #############

    cnn_model  = ConvNet(middleRepr=flags_middleRepr, embedding_size=flags_units_CNN)
    cnn_logits = cnn_model.logits(images, stop_gradient=not flags_trainCNN)

    return model_name, tf.concat([lstm_logits, cnn_logits], axis=1)

def store_features(mode='combined'):
    if mode not in ['combined', 'lstm', 'cnn']: raise Exception

    # nplabels = np.load(features_directory + 'labels.npy')
    # npvlabels = np.load(features_directory + 'eval_labels.npy')
    # if mode in ['combined', 'cnn']:
    npimages = np.load(features_directory + 'images.npy')
    npvimages = np.load(features_directory + 'eval_images.npy')
    assert len(npimages) == num_samples
    # if mode in ['combined', 'lstm']:
    nplengths  = np.load(features_directory + 'lengths.npy')
    npsketches = np.load(features_directory + 'sketches.npy')
    npvlengths  = np.load(features_directory + 'eval_lengths.npy')
    npvsketches = np.load(features_directory + 'eval_sketches.npy')
    assert len(npsketches) == num_samples

    image_placeholder  = tf.placeholder(dtype=tf.uint8, shape=[None, 224*224])
    sketch_placeholder = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
    length_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])

    feed_dict = \
        lambda start_index, end_index: \
            {
                image_placeholder:  npimages[start_index:end_index],
                sketch_placeholder: npsketches[start_index:end_index],
                length_placeholder: nplengths[start_index:end_index]
            }
    eval_feed_dict = \
        lambda start_index, end_index: \
            {
                image_placeholder:  npvimages[start_index:end_index],
                sketch_placeholder: npvsketches[start_index:end_index],
                length_placeholder: npvlengths[start_index:end_index]
            }

    if mode == 'cnn':
        model_name, logits = _cnn_model(image_placeholder)
        model_name_out = 'cnn_model'
        batch_size = 20
    elif mode == 'lstm':
        model_name, logits = _lstm_model(sketch_placeholder, length_placeholder)
        model_name_out = 'lstm_model'
        batch_size = 30
    elif mode == 'combined':
        model_name, logits = _combined_model(image_placeholder, sketch_placeholder, length_placeholder)
        model_name_out = 'combined_model'
        batch_size = 20

    saver = tf.train.Saver()
    # Check validity of restoring
    with tf.Session() as sess: 
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        checkpoint_dir = checkpoints_directory + model_name + '/'
        # Recover previous work
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_dir + 'checkpoint'))
        if not ckpt: 
            print("ckpt:", ckpt)
            print("checkpoint_dir:", checkpoint_dir)
            print("path name:",  os.path.dirname(checkpoint_dir + 'checkpoint'))
            raise Exception("No values were restored from model {}".format(checkpoint_dir))
        ckpt_path = ckpt.model_checkpoint_path
            
        if ckpt_path:
            saver.restore(sess, ckpt_path)
            print("Values were restored")
        else: 
            raise Exception("No values were restored from model {}".format(checkpoint_dir))
            
            
        ## Save training logits  
        npfeats  = np.zeros((0,256),      dtype=np.float32)
        num_runs = (num_samples // batch_size) + 1
        for i in range(num_runs):
            num_to_take = batch_size
            if (i+1)*batch_size > num_samples:
                num_to_take = num_samples - i*batch_size
            if num_to_take <= 0: break
            
            start_index = i*batch_size
            end_index   = start_index + num_to_take
            
            if (i+1) % 5 == 0: print("Step {:4} from {:4}. (Index: {:4} to {:4})".format(i+1, num_runs, start_index, end_index))
            logits_out = sess.run(logits, feed_dict=feed_dict(start_index, end_index))
            if len(logits_out): npfeats = np.append(npfeats, logits_out, axis=0)

        np.save(features_directory + 'features_' + model_name_out, npfeats)

        ## Save validation logits
        assert len(npvimages) == num_eval
        npfeats_eval  = np.zeros((0,256),      dtype=np.float32)
        num_runs_eval = (num_eval // batch_size) + 1
        for i in range(num_runs_eval):
            num_to_take = batch_size
            if (i+1)*batch_size > num_eval:
                num_to_take = num_eval - i*batch_size
            if num_to_take <= 0: break

            start_index = i*batch_size
            end_index   = start_index + num_to_take
            
            print("Validation. Step {:4} from {:4}. (Index: {:4} to {:4})".format(i+1, num_runs_eval, start_index, end_index))
            logits_out = sess.run(logits, feed_dict=eval_feed_dict(start_index, end_index))
            if len(logits_out): npfeats_eval = np.append(npfeats_eval, logits_out, axis=0)

        np.save(features_directory + 'eval_features_' + model_name_out, npfeats_eval)

def main():
    if in_jupyter():
        get_ipython().system('echo "GPU Device in use: \'$CUDA_VISIBLE_DEVICES\'"')
    else:
        os.system('echo "GPU Device in use: \'$CUDA_VISIBLE_DEVICES\'"')

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    if in_jupyter(): 
        tf.app.flags.DEFINE_string('f', '', 'kernel')
    flags.DEFINE_boolean('write_samples',  False,      "If a selection of features should be written to files in npy format")
    flags.DEFINE_string ('store_features', 'combined', "Store features based on stored samples. Models: combined, cnn, lstm")

    if FLAGS.write_samples:
        write_to_file()
    if FLAGS.store_features:
        store_features(mode=FLAGS.store_features)
    
    
if __name__ == '__main__':
    main()
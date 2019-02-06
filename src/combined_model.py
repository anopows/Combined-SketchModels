import numpy as np
import tensorflow as tf
import os

from utility import in_jupyter, limit_cpu, get_logger
from input_pipeline import train_data, eval_data
from model_fns import classification, ConvNet
from loss_fns  import classification_loss
from loss_fns  import classification_summary

if in_jupyter():
    get_ipython().system('echo "GPU Device in use: \'$CUDA_VISIBLE_DEVICES\'"')
else:
    os.system('echo "GPU Device in use: \'$CUDA_VISIBLE_DEVICES\'"')

flags = tf.app.flags
FLAGS = flags.FLAGS

# Parse command line arguments, model parameters
if in_jupyter(): 
    tf.app.flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_boolean('small',            True,          "Use small dataset")
flags.DEFINE_integer('units_LSTM',      256,     "LSTM state size")
flags.DEFINE_integer('units_CNN',       256,     "CNN  state size")
flags.DEFINE_integer('es_lstm',           0,     "O: take LSTM output. Otherwise: MatMul to desired embedding_size")
flags.DEFINE_integer('lstm_stack',        2,     "How many LSTMs to stack at every layer")
flags.DEFINE_integer('batch_size',       20,     "Batch size")
flags.DEFINE_integer('num_steps',     50000,     "Steps of training")
flags.DEFINE_boolean('middleRepr',    False,     "Use middle representations of CNN")
flags.DEFINE_boolean('trainCNN',      True,      "Training CNN weights or keeping them fixed")
flags.DEFINE_boolean('pretrainedCNN', False,      "Use a pretrained CNN weights")
flags.DEFINE_float  ('lrCNN',             0,     "If set, use different learning rate for CNN")

# Most likely fixed model choices
flags.DEFINE_string( 'normalize',     'diff',   "Calculate standard deviations for positions(pos)/differences(diff)")
flags.DEFINE_boolean('bnorm_before',  False,     "Use batch normalization before classification layers")
flags.DEFINE_boolean('bnorm_middle',  False,    "Use batch normalization in between classification layers")
flags.DEFINE_boolean('bnorm_eval_update',  False,    "Update statistics in validating")
flags.DEFINE_string( 'used_steps',    'snapshots',     "Use all/allAndPad/last/snapshots hidden states of the LSTM for training")
flags.DEFINE_string( 'prefix',        'CombinedModel', "Model name prefix")

# Fixed Model choices
embedding_size = 64
clip_gradient  = 1
differences    = True

model_name     = '{}_unitsLSTM{}x{}_unitsCNN{}_bs{}'.format(
    FLAGS.prefix, FLAGS.lstm_stack, FLAGS.units_LSTM, FLAGS.units_CNN, FLAGS.batch_size)

model_name += '_usedSteps-' + FLAGS.used_steps
if FLAGS.pretrainedCNN: model_name += '_pretrainedCNN'
if FLAGS.trainCNN:      model_name += '_trainCNN'
if FLAGS.es_lstm:       model_name += '_es-lstm' + str(FLAGS.es_lstm)
if FLAGS.middleRepr:    model_name += '_middleRepr'
if FLAGS.lrCNN:         model_name += '_lrCNN' + str(FLAGS.lrCNN)
if not FLAGS.small:     model_name += '_fullModel'

# Loging/Checkpts options
log_in_tb     = True
log_in_file   = True
save_checkpts = True
restore       = False

# Folders for storage/retrival
main_directory  = '../'
tensorboard_directory = main_directory + 'tb_graphs/'
checkpoints_directory = main_directory + 'checkpts/'
logging_directory     = main_directory + 'logs/'

if FLAGS.used_steps != 'snapshots':
    raise Exception("'{}' not implemented".format(FLAGS.used_steps))
    
#########
# Input #
#########

num_classes = 10 if FLAGS.small else 345
length, sketch, images, labels  = \
    train_data(batch_size=FLAGS.batch_size, mode=None, 
               imgmode='snapshots', max_avg_snapshots=7, small=FLAGS.small, int_labels=True)
eval_length, eval_sketch, eval_images, eval_labels = \
    eval_data (batch_size=FLAGS.batch_size, mode=None,
               imgmode='middle_last', max_avg_snapshots=7, small=FLAGS.small, int_labels=True, epochs=None)


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
    first_rows  = 122.5 * tf.ones(tf.stack([batch_size, 1, 2])) 
    sketch_vals = tf.concat([first_rows, sketch_vals], axis=1) 
    # Difference calculation
    sketch_vals = sketch_vals[:, 1:, 0:2] - sketch_vals[:, 0:-1, 0:2]
    # Throw away the last row of difference when it should be padding
    sketch_vals = mask_sketch(sketch_vals, lengths)
    return tf.concat([sketch_vals, sketches[:,:,2:]], axis=2)

with tf.variable_scope('preprocessing', reuse=tf.AUTO_REUSE):
    sketch      = tf.cast(sketch,      tf.float32)
    eval_sketch = tf.cast(eval_sketch, tf.float32)

    if FLAGS.normalize == 'pos':
        sketch      = normalize_positions(sketch)
        eval_sketch = normalize_positions(eval_sketch)

    if differences:
        sketch      = calc_diffs(sketch, length)
        eval_sketch = calc_diffs(eval_sketch, eval_length)

        if FLAGS.normalize == 'dist': 
            sketch      = normalize_movement(sketch, length)
            eval_sketch = normalize_movement(eval_sketch, eval_length)


##############
# LSTM Model #
##############

with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):   
    rnn_cell   = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(FLAGS.units_LSTM) for _ in range(FLAGS.lstm_stack)])
    init_state = rnn_cell.zero_state(batch_size=FLAGS.batch_size, dtype=tf.float32)

    lstm_out, _ = tf.nn.dynamic_rnn(rnn_cell, sketch,
                                    initial_state=init_state,
                                    dtype=tf.float32,
                                    scope='lstm_dynamic'
                          )
    
    eval_lstm_out, _ = tf.nn.dynamic_rnn(rnn_cell, eval_sketch,
                                                      initial_state=init_state,
                                                      dtype=tf.float32,
                                                      scope='lstm_dynamic'
                                    )

    if FLAGS.es_lstm:
        lstm_out      = tf.layers.dense(lstm_out, FLAGS.es_lstm)
        eval_lstm_out = tf.layers.dense(eval_lstm_out, FLAGS.es_lstm)


########################################
# Adapt LSTM states/labels to snapshot #
########################################

# Calculate indices where snapshot lie
snapshot_indices      = tf.where(sketch[:,:,2])      # end of a stroke: indices for snapshot
eval_snapshot_indices = tf.where(eval_sketch[:,:,2]) 
# Apply for labels
labels_snapshots      = tf.gather_nd(labels, snapshot_indices[:,:1]) # only take indices denoting sample position
eval_labels_snapshots = tf.gather_nd(eval_labels, eval_snapshot_indices[:,:1])

# Apply for lstm states
lstm_logits      = tf.gather_nd(lstm_out, snapshot_indices) 

# For the summary save the states relating to end/middle logits
def gather_middle_last(values, lengths):
    lengths_half = (lengths-1)//2
    lengths_end  = lengths-1
    indices_middle = tf.stack([tf.range(tf.shape(lengths)[0]), lengths_half], axis=1)
    indices_end    = tf.stack([tf.range(tf.shape(lengths)[0]), lengths_end],  axis=1)
    return tf.gather_nd(values, indices_middle), tf.gather_nd(values, indices_end)

eval_lstm_middle, eval_lstm_last = gather_middle_last(eval_lstm_out, eval_length)

#############
# CNN Model #
#############

# Pick snaphots images
def snapshot_images(images, sketch):
    images = tf.reshape(images,  [FLAGS.batch_size,-1,224*224])
    # find out how many snapshots per sketch
    length = tf.reduce_sum(sketch[:,:,2], axis=1)
    length = tf.reshape(length, [FLAGS.batch_size])
    length = tf.cast(length, tf.int32)
    # unstack every sample of batch
    imglist = tf.unstack(images)
    lenlist = tf.unstack(length)
    # gather all non-padded snapshot images
    slices = []
    for img,l in zip(imglist, lenlist):
        slices.append(img[:l])
    # stack them together again
    return tf.concat(slices, axis=0), length

imgs_snapshots, _      = snapshot_images(images, sketch)

# Calculate CNN representation
cnn_model  = ConvNet(middleRepr=FLAGS.middleRepr, embedding_size=FLAGS.units_CNN, scope='cnn')
cnn_logits = cnn_model.logits(imgs_snapshots, stop_gradient=not FLAGS.trainCNN)

# For summary evaluate middle and last snapshots
print(eval_images)
eval_cnn_middle = cnn_model.logits(eval_images[:,0], stop_gradient=True)
eval_cnn_last   = cnn_model.logits(eval_images[:,1], stop_gradient=True)

#############################
# Combine CNN & LSTM Logits #
#############################

# Combine with CNN logits
logits_combined      = tf.concat([lstm_logits, cnn_logits], axis=1)
# For summary
eval_middle          = tf.concat([eval_lstm_middle, eval_cnn_middle], axis=1)
eval_last            = tf.concat([eval_lstm_last,   eval_cnn_last],   axis=1)

##################
# Classification #
##################

def classification_fn(logits, training=True): 
    if training:
        training_flag  = True
        trainable_flag = True
    else:
        training_flag  = FLAGS.bnorm_eval_update
        trainable_flag = False
    
    cl_logits = classification(logits, layer_sizes=[64,num_classes],
           bnorm_before=FLAGS.bnorm_before, bnorm_middle=FLAGS.bnorm_middle, # if: apply after first dense layer
           training=training_flag, trainable=trainable_flag, name='classifier')
    return cl_logits


with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
    classifier_logits       = classification_fn(logits_combined, training=True)
    eval_logits_middle      = classification_fn(eval_middle,     training=False)
    eval_logits_last        = classification_fn(eval_last  ,     training=False)


########
# Loss #
########

if FLAGS.lrCNN: # Different optimizer for the classifiers
    loss_lstm, _, global_step, train_op_lstm = \
        classification_loss(
            classifier_logits,
            labels_snapshots,
            scope_to_train='lstm',
            training=True,
            clip_gradient=clip_gradient,
            batch_normalization=FLAGS.bnorm_before or FLAGS.bnorm_middle
        )
    loss_cnn, _, _, train_op_cnn = \
        classification_loss(
            classifier_logits,
            labels_snapshots,
            scope_to_train='cnn', 
            training=True,
            clip_gradient=clip_gradient,
            batch_normalization=FLAGS.bnorm_before or FLAGS.bnorm_middle
        )

    avg_loss = tf.reduce_mean(tf.concat([loss_lstm, loss_cnn], axis=0))
    train_op = tf.group([train_op_lstm, train_op_cnn])
else:
    loss, avg_loss, global_step, train_op = \
        classification_loss(
            classifier_logits, labels_snapshots, scope_to_train=None, 
            training=True, clip_gradient=clip_gradient,
            batch_normalization=FLAGS.bnorm_before or FLAGS.bnorm_middle
        )


#############
# Summaries #
#############
    
# validation from last logits
summary_op = tf.summary.scalar('combined_classification_loss', avg_loss)
eval_summary_op, eval_summary_vars     = classification_summary(eval_logits_last,  eval_labels, 'combined_validation_classification_summaries')

# Validation from middle logits
middle_summary_op, middle_summary_vars = classification_summary(eval_logits_middle, eval_labels, 'combined_validation_classification_summaries/middlepoint')


############
# Training #
############

if save_checkpts or restore: saver = tf.train.Saver(max_to_keep=10)
    
checkpoint_dir = checkpoints_directory + model_name + '/'
tensorboard_dir = tensorboard_directory + model_name + '/'

if save_checkpts and not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if log_in_tb and not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
if log_in_file:
    if not os.path.exists(logging_directory):
        os.makedirs(logging_directory)
    logger = get_logger(model_name, logging_directory)
    logging = logger.info
else:
    logging = print
    
logging("Current model: \n\t{}".format(model_name))

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess: 
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    
    # Recover previous work
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_dir + 'checkpoint'))
    if restore and ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        logging("Previously trained values were restored")
    elif restore and FLAGS.pretrainedCNN:
        cnn_model.restore(sess)
        logging("CNN model restored")
    else: 
        logging("No values were restored. Starting with new weights")

    sess.graph.finalize()

    if log_in_tb: writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
    starti = global_step.eval()
    for i in range(starti, FLAGS.num_steps):
        # Regular training
        if (i+1)%20 == 0 or (i-starti)<20: 
            loss_batch, _, summary = sess.run([avg_loss, train_op, summary_op])
            if log_in_tb:
                writer.add_summary(summary, global_step=i)
        else:
            sess.run(train_op)
            
        # Evaluate every x steps
        if (i+1)%40 == 0 or (i-starti)<20: 
            eval_summary, eval_accuracy_out, middle_summary = \
                sess.run([eval_summary_op, 
                          eval_summary_vars['classification_accuracy'], 
                          middle_summary_op]
                        )
            eval_text = " Step {} loss in batch: {:.3f}. Validation accuracy {:.3f}".format(
                i+1, loss_batch, eval_accuracy_out)
            logging(eval_text)
                
            if log_in_tb: 
                writer.add_summary(eval_summary, global_step=i)
                writer.add_summary(middle_summary, global_step=i)
                writer.flush() # Update tensorboard

        # Save all 5000 steps
        if save_checkpts and (i+1)%5000 == 0:
            saver.save(sess, checkpoint_dir + 'checkpoint', global_step.eval())
    
    # Final evaluation
    sum_acc_last     = 0.0
    sum_acc_middle   = 0.0
    num_eval_batches = 1000
    for i in range(num_eval_batches):
        eval_accuracy_last, eval_accuracy_middle =             sess.run([eval_summary_vars['classification_accuracy'], 
                     middle_summary_vars['classification_accuracy']]
            )
        sum_acc_last    += eval_accuracy_last
        sum_acc_middle  += eval_accuracy_middle

        if i%100 == 0: 
            text_middle  = " After {:4d} steps, the average accuracy of half finished sketches: {:.3f}\n".format(i+1, sum_acc_middle/(i+1))
            text_last    = " After {:4d} steps, the average accuracy for the last token:        {:.3f}\n".format(i+1, sum_acc_last/(i+1))
            logging(text_middle + text_last + "========\n")
            
    text_middle  = "The average accuracy of half finished sketches: {:.3f}\n".format(sum_acc_middle/num_eval_batches)
    text_last    = "The average accuracy for the last toke:         {:.3f}\n".format(sum_acc_last/num_eval_batches)
    logging("\n\nFinal accuracy:\n=========\n" + text_middle + text_last)

num_vars = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
logging("Number of trainable variables:" + str(num_vars))

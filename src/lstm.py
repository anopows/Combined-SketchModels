import numpy as np
import tensorflow as tf
import os

from utility import *
from input_pipeline import train_data, eval_data
from model_fns import classification
from loss_fns  import classification_loss
from loss_fns  import classification_summary

if in_jupyter():
    get_ipython().system('echo "GPU Device in use: \'$CUDA_VISIBLE_DEVICES\'"')
else:
    os.system('echo "GPU Device in use: \'$CUDA_VISIBLE_DEVICES\'"')

flags = tf.app.flags
FLAGS = flags.FLAGS

# Parse command line arguments
if in_jupyter(): 
    tf.app.flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_boolean('small',            True,          "Use small dataset")
flags.DEFINE_integer('hiddensize',       256,   "LSTM state size")
flags.DEFINE_integer('lstm_stack',         2,   "How many LSTMs to stack at every layer")
flags.DEFINE_string( 'normalize',     'diff',   "Calculate standard deviations for positions(pos)/differences(diff)")
flags.DEFINE_boolean('bnorm_before',  False,     "Use batch normalization before classification layers")
flags.DEFINE_boolean('bnorm_middle',  False,    "Use batch normalization in between classification layers")
flags.DEFINE_boolean('bnorm_eval_update',  False,    "Update statistics in validating")
flags.DEFINE_string( 'used_steps',    'snapshots',     "Use all/allAndPad/last/snapshots hidden states of the LSTM for training")
flags.DEFINE_string( 'prefix',        'LSTMv5', "Model name prefix")

# Model choices
batch_size     = 30
embedding_size = 64
hidden_size    = FLAGS.hiddensize
clip_gradient  = 1
stacked_LSTMs  = FLAGS.lstm_stack
num_steps      = 50000
differences    = True
normalize      = FLAGS.normalize
bnorm_before   = FLAGS.bnorm_before
bnorm_middle   = FLAGS.bnorm_middle

model_name     = '{}_hs{}x{}'.format(FLAGS.prefix, stacked_LSTMs, hidden_size)
if normalize:         model_name += '_' + FLAGS.normalize + 'Normalized'
if bnorm_before:      model_name += '_bn-before'
if bnorm_middle:      model_name += '_bn-middle'
if FLAGS.bnorm_eval_update :     model_name += '_bnEvalUpdate'
model_name += '_usedSteps_' + FLAGS.used_steps
if not FLAGS.small:   model_name += '_fullModel'

print("Current model:\n\t", model_name)

# Loging/Checkpts options
log_in_tb     = False
log_in_file   = False
save_checkpts = False
restore       = True

# Folders for storage/retrival
main_directory  = '../'
tensorboard_directory = main_directory + 'tb_graphs/'
checkpoints_directory = main_directory + 'checkpts/'
logging_directory     = main_directory + 'logs/'


#########
# Input #
#########

num_classes = 10 if FLAGS.small else 345
# Triplet part
length, sketch, _, labels  = train_data(batch_size=batch_size, mode=None, small=FLAGS.small, int_labels=True)
eval_length, eval_sketch, _, eval_labels = eval_data (batch_size=batch_size, mode=None, 
                                                      small=FLAGS.small, int_labels=True, epochs=None)


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
    eval_sketch = tf.cast(eval_sketch, tf.float32)

    if normalize == 'pos':
        sketch      = normalize_positions(sketch)
        eval_sketch = normalize_positions(eval_sketch)

    if differences:
        sketch      = calc_diffs(sketch, length)
        eval_sketch = calc_diffs(eval_sketch, eval_length)

        if normalize == 'dist': 
            sketch      = normalize_movement(sketch, length)
            eval_sketch = normalize_movement(eval_sketch, eval_length)



##############
# LSTM Model #
##############

with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):   
    rnn_cell   = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(hidden_size) for _ in range(stacked_LSTMs)])
    print(rnn_cell)
    init_state = rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

    lstm_out, state_out = tf.nn.dynamic_rnn(rnn_cell, sketch,
                                    initial_state=init_state,
                                    dtype=tf.float32,
                                    scope='lstm_dynamic'
                          )
    
    eval_lstm_out, eval_state_out = tf.nn.dynamic_rnn(rnn_cell, eval_sketch,
                                                      initial_state=init_state,
                                                      dtype=tf.float32,
                                                      scope='lstm_dynamic'
                                    )



##################
# Classification #
##################

classification_fn =     lambda logits : classification(logits, layer_sizes=[64,num_classes], 
                                   bnorm_before=bnorm_before, bnorm_middle=bnorm_middle, # if: apply after first dense layer
                                   training=True, trainable=True, name='classifier')

# Needs to be separate to not update batch norm params
eval_classification_fn =     lambda logits : classification(logits, layer_sizes=[64,num_classes], 
                                   bnorm_before=bnorm_before, bnorm_middle=bnorm_middle, # if: apply after first dense layer
                                   training=FLAGS.bnorm_eval_update, trainable=False, name='classifier')


with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
    classifier_logits  = classification_fn(tf.reshape(lstm_out, [-1, hidden_size]))
    classifier_logits  = tf.reshape(classifier_logits, [batch_size, -1, num_classes])
    
    eval_classifier_logits  = eval_classification_fn(tf.reshape(eval_lstm_out, [-1, hidden_size]))
    eval_classifier_logits  = tf.reshape(eval_classifier_logits, [batch_size, -1, num_classes])


########
# Loss #
########

with tf.variable_scope('expand_labels', reuse=tf.AUTO_REUSE):
    labels = tf.reshape(labels, [batch_size])
    labels = tf.expand_dims(labels, [1])

    multiples = tf.stack([1, tf.shape(classifier_logits)[1]])
    labels = tf.tile(labels, multiples)
    
    eval_labels = tf.reshape(eval_labels, [batch_size])
    eval_labels = tf.expand_dims(eval_labels, [1])

    eval_multiples = tf.stack([1, tf.shape(eval_classifier_logits)[1]])
    eval_labels = tf.tile(eval_labels, eval_multiples)

if FLAGS.used_steps == 'allAndPad':
    loss, avg_loss, global_step, train_op =         classification_loss(classifier_logits, labels, scope_to_train=None, 
                            training=True, clip_gradient=clip_gradient, batch_normalization=bnorm_before or bnorm_middle)
else: # create mask for the loss
    if FLAGS.used_steps == 'all':
        mask = tf.sequence_mask(length, dtype=tf.float32)
    elif FLAGS.used_steps == 'last':
        zeros = tf.zeros_like(classifier_logits[:, :-1,0])
        ones  = tf.ones_like (classifier_logits[:, :1, 0])
        mask = tf.concat([zeros,ones], axis=1)
    elif FLAGS.used_steps == 'snapshots':
        mask = tf.cast(tf.equal(sketch[:,:,2], 1), tf.float32)
    else:
        raise Exception("used step '{}' not known".format(FLAGS.used_steps))

    loss, avg_loss, global_step, train_op =         classification_loss(classifier_logits, labels, scope_to_train=None, 
                            training=True, mask=mask, clip_gradient=clip_gradient, batch_normalization=bnorm_before or bnorm_middle)



#############
# Summaries #
#############
    
summary_op, summary_vars =     classification_summary(classifier_logits[:,-1], labels[:,-1], 'lstm_classification_summaries', 
                           classifier_logits=classifier_logits[:,-1],
                           loss=loss[:,-1])
eval_summary_op, eval_summary_vars =     classification_summary(eval_classifier_logits[:,-1], eval_labels[:,-1], 'lstm_validation_classification_summaries')


ecl_allsteps = tf.reshape(eval_classifier_logits, shape = [-1,num_classes])
elb_allsteps = tf.reshape(eval_labels, shape = [-1])
eval_summary_op_allsteps, eval_summary_vars =     classification_summary(ecl_allsteps, elb_allsteps, 'lstm_validation_classification_summaries/all_steps')

def gather_at_logits(logits, lengths, progress=1.0):
    len_at = tf.multiply(tf.cast(lengths, tf.float32),progress) - 1
    len_at = tf.cast(len_at, tf.int32)
    indices  = tf.stack([tf.range(tf.shape(lengths)[0]), len_at], axis=1)
    return tf.gather_nd(logits, indices)

# Validation from last logits
last_logits = gather_at_logits(eval_classifier_logits, eval_length, progress=1)
last_summary_op, last_summary_vars =     classification_summary(last_logits, eval_labels[:,-1], 'lstm_validation_classification_summaries/lastpoint')

# Validation from middle logits
middle_logits = gather_at_logits(eval_classifier_logits, eval_length, progress=0.5)
middle_summary_op, middle_summary_vars =     classification_summary(middle_logits, eval_labels[:,-1], 'lstm_validation_classification_summaries/middlepoint')


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
    log_file = open(logging_directory + model_name + '.log', 'w')
    
with tf.Session() as sess: 
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    sess.run(tf.tables_initializer())
    sess.graph.finalize()
    
    # Recover previous work
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_dir + 'checkpoint'))
    if restore and ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Values were restored")
    else: 
        print("No values were restored. Starting with new weights")
        
    if log_in_tb: writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
    starti = global_step.eval()
    for i in range(starti, num_steps):
        # Regular training
        if (i+1)%20 == 0 or (i-starti)<20: 
            loss_batch, _, summary = sess.run([avg_loss, train_op, summary_op])
            if log_in_tb:
                writer.add_summary(summary, global_step=i)
        else:
            sess.run(train_op)
            
        # Evaluate every x steps
        if (i+1)%40 == 0 or (i-starti)<20: 
            eval_summary, eval_accuracy_out, eval_summary_allsteps, middle_summary, last_summary =                 sess.run([eval_summary_op, 
                          eval_summary_vars['classification_accuracy'], 
                          eval_summary_op_allsteps, middle_summary_op, last_summary_op]
                        )
            eval_text = " Step {} loss in batch: {:.3f}. Validation accuracy {:.3f}".format(
                i+1, loss_batch, eval_accuracy_out)
            print(eval_text)
            if log_in_file: 
                print(eval_text, file=log_file)
                
            if log_in_tb: 
                writer.add_summary(eval_summary, global_step=i)
                writer.add_summary(eval_summary_allsteps, global_step=i)
                writer.add_summary(middle_summary, global_step=i)
                writer.add_summary(last_summary, global_step=i)
                writer.flush() # Update tensorboard

        # Save all 5000 steps
        if save_checkpts and (i+1)%5000 == 0:
            saver.save(sess, checkpoint_dir + 'checkpoint', global_step.eval())
    
    # Final evaluation
    sum_acc_lastpad  = 0.0
    sum_acc_middle   = 0.0
    sum_acc_last     = 0.0
    num_eval_batches = 1000
    for i in range(num_eval_batches):
        eval_accuracy_lastpad, eval_accuracy_middle, eval_accuracy_last = \
            sess.run([eval_summary_vars['classification_accuracy'], 
                     middle_summary_vars['classification_accuracy'],
                     last_summary_vars['classification_accuracy']]
            )
        sum_acc_lastpad += eval_accuracy_lastpad
        sum_acc_middle  += eval_accuracy_middle
        sum_acc_last    += eval_accuracy_last

        if i%100 == 0: 
            text_middle  = " After {:4d} steps, the average accuracy of half finished sketches: {:.3f}\n".format(i+1, sum_acc_middle/(i+1))
            text_last    = " After {:4d} steps, the average accuracy for the last toke:         {:.3f}\n".format(i+1, sum_acc_last/(i+1))
            text_lastpad = " After {:4d} steps, the average accuracy for last padded token:     {:.3f}\n".format(i+1, sum_acc_lastpad/(i+1))
            print(text_middle, text_last, text_lastpad, "========\n", sep="")
            if log_in_file:
                print(text_middle, text_last, text_lastpad, "========\n", file=log_file)
            
    text_middle  = "The average accuracy of half finished sketches: {:.3f}\n".format(sum_acc_middle/num_eval_batches)
    text_last    = "The average accuracy for the last toke:         {:.3f}\n".format(sum_acc_last/num_eval_batches)
    text_lastpad = "The average accuracy for last padded token:     {:.3f}\n".format(sum_acc_lastpad/num_eval_batches)
    print("\n\nFinal accuracy:\n=========\n", text_middle, text_last, text_lastpad)
    if log_in_file:
        print("\n\nFinal accuracy:\n=========\n", text_middle, text_last, text_lastpad, file=log_file)


num_vars = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
print("Number of trainable variables:", num_vars)
if log_in_file:
    print("Number of trainable variables:", num_vars, file=log_file)
    log_file.close()

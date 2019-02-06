import numpy as np
import tensorflow as tf
import os

from utility import in_jupyter, limit_cpu, get_logger
from input_pipeline import train_data, eval_data
from loss_fns  import triplet_loss_train, triplet_loss_eval
from model_fns import classification, ConvNet
from loss_fns  import classification_loss
from loss_fns  import triplet_summary, classification_summary

if in_jupyter():
    get_ipython().system('echo "GPU Device in use: \'$CUDA_VISIBLE_DEVICES\'"')
else:
    os.system('echo "GPU Device in use: \'$CUDA_VISIBLE_DEVICES\'"')

# Model choices
flags = tf.app.flags
FLAGS = flags.FLAGS

if in_jupyter(): 
    tf.app.flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_boolean('small',            True,          "Use small dataset")
flags.DEFINE_string ('cnn_model_choice', 'resnet',      "Choice of Convolutional Model (resnet/inception/vgg)")
flags.DEFINE_string ('feed_mode',        'threesplit',  "How to feed the model (threesplit/triplets/None)")
flags.DEFINE_string ('loss_mode',        'hinge',       "Loss: hinge/exp")
flags.DEFINE_integer('units_CNN',                 256,  "CNN  state size")
flags.DEFINE_integer('batch_size',                 30,  "Batch size")
flags.DEFINE_integer('annealing',                   0,  "Half learn rate every x steps. No annealing: 0 (default)")
flags.DEFINE_float  ('learning_rate',          0.0001,  "Learning rate for CNN weights")
flags.DEFINE_integer('hinge_loss_alpha',            1,  "Distance for hinge loss hyper parameter")
loss_mode_params  = {'alpha': FLAGS.hinge_loss_alpha}

# Further parameters
sgd_switch = None # After x steps, switch to SGD optimization with decay
sgd_learn_rate   = 1e-3

# Loging/Checkpts options
log_in_tb     = True
log_in_file   = True
save_checkpts = True
restore       = True
restore_ckpt  = None # 15000

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
_, _, images, labels           = train_data(batch_size=FLAGS.batch_size, mode=FLAGS.feed_mode, 
                                            small=FLAGS.small, int_labels=True)
_, _, eval_images, eval_labels = eval_data (batch_size=FLAGS.batch_size, mode='triplets', 
                                            small=FLAGS.small, int_labels=True, epochs=None)

# Classifier part
_, _, images2, labels2           = train_data(batch_size=FLAGS.batch_size, small=FLAGS.small, int_labels=True)
_, _, eval_images2, eval_labels2 = eval_data( batch_size=FLAGS.batch_size, small=FLAGS.small, epochs=None, int_labels=True)

#########
# Model #
#########

cnn_model = ConvNet(model=FLAGS.cnn_model_choice, feed_mode=FLAGS.feed_mode, embedding_size=FLAGS.units_CNN,
                    loss_mode=FLAGS.loss_mode, loss_mode_params=loss_mode_params, small=FLAGS.small)
# Triplet part
logits1, logits2, logits3                = cnn_model.logits(images,      training=True,  feed_mode=FLAGS.feed_mode)
eval_logits1, eval_logits2, eval_logits3 = cnn_model.logits(eval_images, training=False, feed_mode=FLAGS.feed_mode)

# Classifier part
logits_2      = cnn_model.logits(images2,      training=False, feed_mode=None, stop_gradient=True) # Training false for BNorm
eval_logits_2 = cnn_model.logits(eval_images2, training=False, feed_mode=None, stop_gradient=True)

classifier_logits      = classification(logits_2,      layer_sizes=[64,num_classes], name='classifier')
eval_classifier_logits = classification(eval_logits_2, layer_sizes=[64,num_classes], name='classifier') # shared weights

# def l2_normalize(vals):
#     sqr_sum = tf.reduce_sum(vals**2, axis=1, keepdims=True)
#     tf.assert_equal(tf.shape(sqr_sum)[0], 30)
#     return vals/sqr_sum
# 
# logits1 = l2_normalize(logits1)
# logits2 = l2_normalize(logits2)
# logits2 = l2_normalize(logits2)
# eval_logits1 = l2_normalize(eval_logits1)
# eval_logits2 = l2_normalize(eval_logits2)
# eval_logits2 = l2_normalize(eval_logits2)
# logits_2 = l2_normalize(logits_2)
# eval_logits2 = l2_normalize(eval_logits2)
# classifier_logits = l2_normalize(classifier_logits)
# eval_classifier_logits =l2_normalize(eval_classifier_logits)

##################
# Losses & Train #
##################

# Triplets
loss, avg_loss, global_step, train_op, *args = \
    triplet_loss_train(logits1, logits2, logits3, mode=FLAGS.loss_mode, **loss_mode_params, 
                       learning_rate=FLAGS.learning_rate, annealing=FLAGS.annealing,
                       clip_gradient=1,
                       sgd_switch=sgd_switch, sgd_learn_rate=sgd_learn_rate)
if sgd_switch:
    train_op_sgd = args[0] # Case we switch to SGD: Second train_op

eval_loss, eval_avg_loss = \
    triplet_loss_eval(eval_logits1, eval_logits2, eval_logits3, mode=FLAGS.loss_mode, **loss_mode_params)

# Classifier
classifier_loss, classifier_avg_loss, classifier_global_step, classifier_train_op = \
    classification_loss(classifier_logits, labels2, scope_to_train='classifier/', training=True)

#############
# Summaries #
#############

# Triplets
# summary_op, s_vars = triplet_summary(loss, avg_loss,  'summaries',
#     loss_name=FLAGS.loss_mode, hinge_alpha=FLAGS.hinge_loss_alpha) # debug logits:, logits1=logits1, logits2=logits2, logits3=logits3)
# eval_summary_op, s_eval_vars = triplet_summary(eval_loss, eval_avg_loss, 'validation_summaries',
#     loss_name=FLAGS.loss_mode, hinge_alpha=FLAGS.hinge_loss_alpha)

summary_op, s_vars = triplet_summary(logits1, logits2, logits3,
    loss_mode=FLAGS.loss_mode, hinge_alpha=FLAGS.hinge_loss_alpha, 
    scope_name = 'summaries') # debug logits:, logits1=logits1, logits2=logits2, logits3=logits3)
eval_summary_op, s_eval_vars = triplet_summary(eval_logits1, eval_logits2, eval_logits3,
    loss_name=FLAGS.loss_mode, hinge_alpha=FLAGS.hinge_loss_alpha,
    scope_name = 'validation_summaries')

# Classifier
eval_classification_summary_op, scl_eval_vars = \
    classification_summary(eval_classifier_logits, eval_labels2, 'validation_classification_summaries')


############
# Training #
############

if save_checkpts or restore: saver = tf.train.Saver(max_to_keep=10)
    
checkpoint_dir  = checkpoints_directory + cnn_model.model_name + '/'
tensorboard_dir = tensorboard_directory + cnn_model.model_name + '/'

if save_checkpts and not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if log_in_tb and not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)
if log_in_file:
    if not os.path.exists(logging_directory):
        os.makedirs(logging_directory)
    logger = get_logger(cnn_model.model_name, logging_directory)
    logging = logger.info
else:
    logging = print
    
with tf.Session() as sess: 
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    sess.run(tf.tables_initializer())

    # Classification accuracy placeholder    
    eval_acc_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
    eval_acc_summary = tf.summary.scalar("validation_classification_accuracy_batches", eval_acc_placeholder)
    
    # For resetting classifier
    classifier_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'classifier/')
    init_classifier = tf.variables_initializer(classifier_weights + [classifier_global_step])
    
    sess.graph.finalize()
    
    # Recover previous work
    if restore:
        ckpt_path = None
        # Get path
        if restore_ckpt: 
            ckpt_path = os.path.dirname(checkpoint_dir + 'checkpoint') + '/checkpoint-' + str(restore_ckpt)
        else:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_dir + 'checkpoint'))
            if ckpt: ckpt_path = ckpt.model_checkpoint_path
        
        # Restore checkpoint
        if ckpt_path and os.path.exists(ckpt_path + '.index'):
            saver.restore(sess, ckpt_path)
            logging("Values were restored")
        else:
            logging("No values were restored. Starting with new weights")
    else: 
        logging("No values were restored. Starting with new weights")
            
    
    if log_in_tb: writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
    starti = global_step.eval()
    
    for i in range(starti, 50000):
        # Switch training from ADAM to SGD after some steps, if desired
        cur_train_op = train_op if (not sgd_switch) or i<sgd_switch else train_op_sgd
        
        # Regular training, run with summary every x steps
        if log_in_tb and ( (i+1)%20 == 0 or (i-starti)<20 ): 
            avg_loss_out, _, summary = sess.run([avg_loss, cur_train_op, summary_op])
            writer.add_summary(summary, global_step=global_step.eval())
        else:
            sess.run(cur_train_op)

        # Evaluate every x steps
        if (i+1)%40 == 0 or (i-starti)<20: 
            eval_summary, eval_accuracy_out = \
                sess.run([eval_summary_op, s_eval_vars['accuracy']])
            logging(" Step {:5d}. Loss {:10.8f} | Validation accuracy {:.4f}".format(
                i+1, avg_loss_out, eval_accuracy_out))
            if log_in_tb: 
                writer.add_summary(eval_summary, global_step=global_step.eval())
                writer.flush() # Update tensorboard
                
        # Save all x steps
        if save_checkpts and (i+1)%5000 == 0:
            saver.save(sess, checkpoint_dir + 'checkpoint', global_step.eval())
        
        # All 1000 steps run classifier for 200 steps
        if (i+1)%1000 == 0:
            logging("\n===\nRunning classification\n===")
            # reset
            sess.run(init_classifier)
                        
            # training
            for _ in range(300):
                sess.run(classifier_train_op)
            # validation
            num_steps = 50
            sum_acc = 0
            for j in range(num_steps):
                sum_acc += sess.run(scl_eval_vars['classification_accuracy'])
            eval_acc_out = sess.run(eval_acc_summary, feed_dict={eval_acc_placeholder: sum_acc/num_steps})
            logging("Average Validation classification accuracy {:.3f}".format(sum_acc/num_steps))
            if log_in_tb:
                writer.add_summary(eval_acc_out, sess.run(global_step))
            logging("\n===\nContinuing\n===")

    # Final classification run
    # reset
    sess.run(init_classifier)
    num_steps_training =  300                
    num_steps_eval     = 5000
    logging("\n\n===\nRunning final classification. Training for {} steps\n===".format(
        num_steps_training
    ))
    # training
    for _ in range(num_steps_training):
        sess.run(classifier_train_op)
    # validation
    sum_acc = 0
    for j in range(num_steps_eval):
        cur_acc  = sess.run(scl_eval_vars['classification_accuracy'])
        sum_acc += cur_acc
    logging("Final Average Validation classification accuracy after validation for {} steps: {:.3f}".format(
        num_steps_eval, sum_acc/num_steps_eval))

num_vars = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
logging("Number of trainable variables: {}".format(num_vars))

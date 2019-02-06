import tensorflow as tf

def triplet_loss_eval(logits1, logits2, logits3, name='eval_loss', mode='exp', **kwargs):
    """Calculates triplet loss
    
    Args:
        logits1: Batch of anchor logits
        logits2: Batch of positive logits
        logits3: Batch of negative loigts
        name: Scope name. Default: 'eval_loss'
        mode: Either 'exp' or 'hinge'. Defaults to 'exp'
    Returns:
        Tensor: Batch of losses
        float: Average loss
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # squared norm of distance of positve and negative pair
        dist_same  = tf.square(logits1 - logits2)
        dist_same  = tf.reduce_sum(dist_same, -1) 
        dist_other = tf.square(logits1 - logits3)
        dist_other = tf.reduce_sum(dist_other, -1)

        if mode == 'exp':
            dist_same  = tf.sqrt(dist_same)
            dist_other = tf.sqrt(dist_other)
            loss = tf.nn.softmax([dist_same, dist_other], axis=0)[0]
            loss = loss**2
            avg_loss = tf.reduce_mean(loss)
        elif mode == 'hinge':
            if 'alpha' in kwargs: hinge_alpha = kwargs['alpha']
            else: hinge_alpha = 0.5
            # print("Hinge loss with alpha", alpha_l2)

            loss = dist_same - dist_other + hinge_alpha
            loss = tf.nn.relu(loss) # difference better than alpha, don't count
            avg_loss = tf.reduce_mean(loss)
        else: 
            raise Exception('Loss mode not implemented')
            
        return loss, avg_loss

from utility import tf_print
def triplet_loss_train(logits1, logits2, logits3, name='loss', mode='exp', clip_gradient=None, learning_rate=1e-4, **kwargs):
    """Calculates triplet loss
    
    Args:
        logits1: Batch of anchor logits
        logits2: Batch of positive logits
        logits3: Batch of negative logits
        name: Scope name. Default: 'eval_loss'
        mode: Either 'exp' or 'hinge'. Defaults to 'exp'
    Returns:
        Tensor:    Batch of losses
        float:     Average loss
        Variable:  global_step
        Operation: train_op
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        # squared norm of distance of positve and negative pair
        dist_same  = tf.square(logits1 - logits2)
        dist_same  = tf.reduce_sum(dist_same, -1) 
        # dist_same  = tf_print(dist_same, transform= lambda x: x[:5], 
        #                       message="Distances to same per batch")
        dist_other = tf.square(logits1 - logits3)
        dist_other = tf.reduce_sum(dist_other, -1)
        # dist_other = tf_print(dist_other, transform = lambda x: x[:5],
        #                       message="Distances to other per batch")

        if mode == 'exp':
            dist_same  = tf.sqrt(dist_same)
            dist_other = tf.sqrt(dist_other)
            loss = tf.nn.softmax([dist_same, dist_other], axis=0)[0]
            loss = loss**2
            avg_loss = tf.reduce_mean(loss)
        elif mode == 'hinge':
            if 'alpha' in kwargs: alpha_l2 = kwargs['alpha']
            else: alpha_l2 = 0.5
            # print("Hinge loss with alpha", alpha_l2)

            loss = dist_same - dist_other + alpha_l2
            loss = tf.nn.relu(loss) # difference better than alpha, don't count
            avg_loss = tf.reduce_mean(loss)
        else: 
            raise Exception('Loss mode not implemented')
            
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        if 'annealing' in kwargs and kwargs['annealing']: # Training with annealing learn rate
            exp_lr = tf.train.exponential_decay(
                learning_rate, global_step, kwargs['anneal_after'], 0.5, staircase=True
            )
            # Overwrite optimizer with an annealed one
            optimizer = tf.train.AdamOptimizer(learning_rate=exp_lr)

        elif 'sgd_switch' in kwargs and kwargs['sgd_switch']: # Switch training to SGD after eg 15k steps
            sgd_switch      = kwargs['sgd_switch']
            sgd_learn_rate  = kwargs['sgd_learn_rate']  if 'sgd_learn_rate'  in kwargs else 0.1
            sgd_decay_every = kwargs['sgd_decay_every'] if 'sgd_decay_every' in kwargs else 5000
            sgd_decay_rate  = kwargs['sgd_decay_rate']  if 'sgd_decay_rate'  in kwargs else 0.5
            
            global_sgd_step = global_step - sgd_switch
            exp_lr = tf.train.exponential_decay(
                sgd_learn_rate, 
                global_sgd_step, 
                sgd_decay_every, 
                sgd_decay_rate, 
                staircase=True
            )
            optimizer_sgd = tf.train.GradientDescentOptimizer(learning_rate=exp_lr)
            train_op_sgd = optimizer_sgd.minimize(loss, global_step=global_step)

        gradients = optimizer.compute_gradients(avg_loss)
        if clip_gradient:
            gradients = [(tf.clip_by_value(grad, -1*clip_gradient, clip_gradient), var) 
                         for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)

        if 'sgd_switch' in kwargs and kwargs['sgd_switch']:
            return loss, avg_loss, global_step, train_op, train_op_sgd
        return loss, avg_loss, global_step, train_op

def classification_loss(logits, labels, name='classifier_loss', scope_to_train='classifier/', training=True,
                        learn_rate=1e-3, batch_normalization=False, clip_gradient=None, mask=None,
                        additional_loss=None,
                        return_gradients=False):
    """Classification from logits and sparse labels
    
    Args:
        logits:         Batch of logits to classify
        labels:         Batch of labels
        name:           Current scope
        scope_to_train: What weights should be trained
        training:       If yes: returning also global_step & train_op
        clip_gradient
        lengths:        If wanting to apply classification loss, only to subset of values
        clip_value
    Returns:
        loss
        avg_loss
        (if training)global_step
        (if training)train_op
    """
    
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        
        if additional_loss is not None:
            assert loss.get_shape().as_list() == additional_loss.get_shape().as_list(), "Same shapes for comparison"
            loss = (loss + additional_loss)/2

        if mask is not None:
            loss = loss * mask
            avg_loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        else:
            avg_loss = tf.reduce_mean(loss)

        if not training:
            return loss, avg_loss
        
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='classification_global_step')
        optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)

        # only apply gradients to classifier weights
        if scope_to_train:
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_to_train)
        else:
            train_vars = None
        
        gradients = optimizer.compute_gradients(avg_loss, var_list=train_vars)
        if clip_gradient:
            gradients = [(tf.clip_by_value(grad, -1*clip_gradient, clip_gradient), var) 
                         for grad, var in gradients if grad is not None]
        

        if batch_normalization: # train_op depdendent on updating batch normalization params
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            assert len(update_ops) == 2 # Mean and variance
            with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(
                    grads_and_vars=gradients, global_step=global_step
                )
        else:
            train_op = optimizer.apply_gradients(
                    grads_and_vars=gradients, global_step=global_step
            )
        
        if return_gradients:
            return loss, avg_loss, global_step, train_op, gradients
        else:
            return loss, avg_loss, global_step, train_op
        
    
def triplet_summary(logits1, logits2, logits3, loss_mode='hinge', hinge_alpha=1,
                        scope_name='triplet_summary', **to_log):
    var_dict = {}
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        dist_same  = tf.square(logits1 - logits2)
        dist_same  = tf.reduce_sum(dist_same, -1) 
        dist_other = tf.square(logits1 - logits3)
        dist_other = tf.reduce_sum(dist_other, -1)

        if loss_mode == 'exp':
            dist_same  = tf.sqrt(dist_same)
            dist_other = tf.sqrt(dist_other)
            loss = tf.nn.softmax([dist_same, dist_other], axis=0)[0]
            loss = loss**2
            avg_loss = tf.reduce_mean(loss)
            tf.summary.scalar("Softmax_loss", avg_loss)
        elif loss_mode == 'hinge':
            loss = dist_same - dist_other + hinge_alpha
            loss = tf.nn.relu(loss) # difference better than alpha, don't count
            avg_loss = tf.reduce_mean(loss)
            tf.summary.scalar("Hinge_loss", avg_loss)
        else: 
            raise Exception('Loss mode not implemented')
        
        closer = dist_other > dist_same
        accuracy = tf.reduce_mean(tf.cast(closer, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        var_dict['accuracy'] = accuracy
        
        tf.summary.scalar('dist_same_avg',  tf.reduce_mean(dist_same))
        tf.summary.scalar('dist_other_avg', tf.reduce_mean(dist_other))
        var_dict['dist_same']  = dist_same
        var_dict['dist_other'] = dist_other

        return tf.summary.merge_all(scope=scope_name), var_dict

    
def classification_summary(logits, labels, name, **to_log):
    """ Creates summaries for Tensorboard based on classification logits and labels
    Args: 
        logits:      Batch of logits
        labels:      Label ground truth
        name:        Scope name of summary
    Returns:
        summary:     Summary
        var_dict:    Calculated metrics
    """
    var_dict = {}
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):   
        preds = tf.argmax(logits, axis=1)
        correct_preds = tf.equal(preds, labels)
        classification_accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

        tf.summary.scalar('classification_accuracy', classification_accuracy)
        var_dict['classification_accuracy'] = classification_accuracy
        
        # Additonal tensors to log
        for tname, tensor in to_log.items():
            tf.summary.histogram(tname, tensor)
            
        return tf.summary.merge_all(scope=name), var_dict
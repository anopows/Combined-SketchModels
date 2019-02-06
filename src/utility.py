import os
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime
from pytz import timezone
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file as ptensors
from tensorflow.python.client.device_lib import list_local_devices

def get_time():
    fmt = "%Y-%m-%d_%H:%M"
    location = "Europe/Zurich"
    return datetime.now(timezone(location)).strftime(fmt)

def limit_cpu(num_threads):
    return tf.ConfigProto(intra_op_parallelism_threads=num_threads,inter_op_parallelism_threads=num_threads)

def in_jupyter():
    try:
        cfg = get_ipython().config # Name error if not exists --> CPython,..
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell' # ipython or jupyter_
    except NameError:
        return False

def names_from_ckpt(model_folder, name=''):
    if not os.path.isdir(model_folder):
        raise Exception("Directory '{}' does not exist".format(model_folder))
    latest_ckpt = tf.train.latest_checkpoint(model_folder)
    if latest_ckpt is None:
        raise Exception("No checkpoints found")
    ptensors(latest_ckpt, all_tensors=False, tensor_name=name)

def list_devices():
    return list_local_devices()

def get_logger(file_name, log_dir='../log', print_to_stdout=True):
    logging.basicConfig(level=logging.INFO, filename="{}/{}.log".format(log_dir,file_name),
                        filemode="a+", format="%(asctime)-15s %(levelname)-8s %(message)s")
    if print_to_stdout:
        logging.getLogger().addHandler(logging.StreamHandler())
    return logging

def plot_histogram(names, values, num_choices=None, figsize=(16,10)):
    # Ugly but other functions not dependent on these imports. TODO: refactor into other module
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats
    sns.set(rc={'figure.figsize':figsize})
    sns.set(color_codes=True)
    for name,val in zip(names, values):
        if num_choices:
            val = np.random.choice(val, num_choices)
        val = np.log10(abs(val)+1e-20)
        sns.distplot(val, label=name)
    plt.legend()

# from https://stackoverflow.com/a/39649614
def tf_print(tensor, transform=None, message="", precision=2, linewidth=150, suppress=False):
    np.set_printoptions(precision=precision, suppress=suppress, linewidth=linewidth)
    # Insert a custom python operation into the graph that does nothing but print a tensors value 
    def print_tensor(x):
        # x is typically a numpy array here so you could do anything you want with it,
        # but adding a transformation of some kind usually makes the output more digestible
        print(message, x if transform is None else transform(x))
        return x
    log_op = tf.py_func(print_tensor, [tensor], [tensor.dtype])[0]
    with tf.control_dependencies([log_op]):
        res = tf.identity(tensor)

    # Return the given tensor
    return res

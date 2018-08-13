import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools.freeze_graph import freeze_graph
from keras.models import load_model
from keras import backend as K
import keras
import os
import numpy as np
import sys
sys.path.insert(0, '../')
from common.utils import *
from common.nn import *
from common.eval import *
import common
import argparse

def freeze(tf_session, model_name, model_input_name, width, height, channels, model_output_name):

    input_binary = True
    graph_def = tf_session.graph.as_graph_def()
    tf.train.Saver().save(tf_session, model_name + '.ckpt')
    tf.train.write_graph(tf_session.graph.as_graph_def(), logdir='.', name=model_name + '.binary.pb', 
                         as_text=not input_binary)

    # We save out the graph to disk, and then call the const conversion routine.
    checkpoint_state_name = model_name + ".ckpt.index"
    input_graph_name = model_name + ".binary.pb"
    output_graph_name = model_name + ".pb"

    input_graph_path = os.path.join(".", input_graph_name)
    input_saver_def_path = ""
    input_checkpoint_path = os.path.join(".", model_name + '.ckpt')

    output_node_names = model_output_name
    
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"

    output_graph_path = os.path.join('.', output_graph_name)
    clear_devices = True
    freeze_graph(input_graph_path, input_saver_def_path,
                 input_binary, input_checkpoint_path,
                 output_node_names, restore_op_name,
                 filename_tensor_name, output_graph_path,
                 clear_devices, "","conv2d_37/convolution/Shape")
    print("Model loaded from: %s" % model_name)
    print("Output written to: %s" % output_graph_path)
    print("Model input name : %s" % model_input_name)
    print("Model input size : %dx%dx%d (WxHxC)" % (width, height, channels))
    print("Model output name: %s" % model_output_name)

K.set_learning_phase(0) 

parser = argparse.ArgumentParser()
parser.add_argument('--full-model-path', type=str)
args = parser.parse_args()


custom_objects={'dice_crossentopy_loss': dice_crossentopy_loss,
                'f1_score': f1_score,
                'dice_coef' : dice_coef,
               }
full_model_path = args.full_model_path
model = load_model(full_model_path,custom_objects=custom_objects)
model_input = model.input.name.replace(':0', '')

out_names = model.output
model_output = model.output.name.replace(':0', '')


model_file_basename = full_model_path[:-4]

sess = K.get_session()
width, height, channels = int(model.input.shape[2]), int(model.input.shape[1]), int(model.input.shape[3])

freeze(sess, model_file_basename, model_input, width, height, channels, model_output)
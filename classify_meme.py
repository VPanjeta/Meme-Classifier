from __future__ import print_function
import sys
import warnings
import os
import tensorflow as tf


image = sys.argv[1]
#^ Path to the image from command line

image_file = tf.gfile.FastGFile(image, 'rb')
#^ Image being read

data = image_file.read()
#^ Data from image file

classes = [line.rstrip() for line in tf.gfile.GFile("meme_classes.txt")]
# Unpersisting graph from file
with tf.gfile.FastGFile("meme_graph.pb", 'rb') as graph:
    definition = tf.GraphDef()
    definition.ParseFromString(graph.read())
    _ = tf.import_graph_def(definition, name='')

with tf.Session() as session:
    tensor = session.graph.get_tensor_by_name('final_result:0')
    #^ Feeding data as input and find the first prediction
    result = session.run(tensor, {'DecodeJpeg/contents:0': data})
    
    top_results = result[0].argsort()[-len(result[0]):][::-1][:5] 
    for type in top_results:
        meme_type = classes[type]
        score = result[0][type]
        print('%-27s : %.5f' % (meme_type, score))
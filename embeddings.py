import io
import time
import logging
from urllib.request import urlopen
from urllib.error import HTTPError
from importlib import import_module

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io


# This class is modified from 
# https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/flowers/trainer/preprocess.py
# - supports inception v4 model
class EmbeddingsGraph(object):
  """Builds a graph and uses it to extract embeddings from images.
  """

  # These constants are set by Inception v3's expectations.
  WIDTH = 299
  HEIGHT = 299
  CHANNELS = 3

  def __init__(self, tf_session, version='inception_v3'):
    self.version = version
    self.tf_session = tf_session
    # input_jpeg is the tensor that contains raw image bytes.
    # It is used to feed image bytes and obtain embeddings.
    self.input_jpeg, self.embedding = self.build_graph()

    init_op = tf.global_variables_initializer()
    self.tf_session.run(init_op)
    self.restore_from_checkpoint('models/%s.ckpt' % self.version)

  def dim(self):
    return self.embedding.shape[-1].value

  def build_graph(self):
    """Forms the core by building a wrapper around the inception graph.

      Here we add the necessary input & output tensors, to decode jpegs,
      serialize embeddings, restore from checkpoint etc.

      To use other Inception models modify this file. Note that to use other
      models beside Inception, you should make sure input_shape matches
      their input. Resizing or other modifications may be necessary as well.
      See tensorflow/contrib/slim/python/slim/nets/inception_v3.py for
      details about InceptionV3.

    Returns:
      input_jpeg: A tensor containing raw image bytes as the input layer.
      embedding: The embeddings tensor, that will be materialized later.
    """

    input_jpeg = tf.placeholder(tf.string, shape=None)
    image = tf.image.decode_jpeg(input_jpeg, channels=self.CHANNELS)

    # Note resize expects a batch_size, but we are feeding a single image.
    # So we have to expand then squeeze.  Resize returns float32 in the
    # range [0, uint8_max]
    image = tf.expand_dims(image, 0)

    # convert_image_dtype also scales [0, uint8_max] -> [0 ,1).
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_bilinear(
        image, [self.HEIGHT, self.WIDTH], align_corners=False)

    # Then rescale range to [-1, 1) for Inception.
    image = tf.subtract(image, 0.5)
    inception_input = tf.multiply(image, 2.0)

    embedding = self.get_pre_logits(inception_input)
    return input_jpeg, embedding

  def get_pre_logits(self, inception_input):
    # import model
    if self.version == 'inception_v3':
        from tensorflow.contrib.slim.python.slim.nets import inception_v3 as inception
        inception = import_module('tensorflow.contrib.slim.python.slim.nets.inception_v3')
    else:
        inception = import_module('nets.inception_v4')

    # Build Inception layers, which expect a tensor of type float from [-1, 1)
    # and shape [batch_size, height, width, channels].
    arg_scope = getattr(inception, '%s_arg_scope' % self.version)
    model = getattr(inception, self.version)
    with tf.contrib.slim.arg_scope(arg_scope()):
      _, end_points = model(inception_input, is_training=False)

    pre_logits_name = (self.version == 'inception_v3' and 'PreLogits' or 'PreLogitsFlatten')
    return end_points[pre_logits_name]

  def restore_from_checkpoint(self, checkpoint_path):
    """To restore inception model variables from the checkpoint file.

       Some variables might be missing in the checkpoint file, so it only
       loads the ones that are avialable, assuming the rest would be
       initialized later.
    Args:
      checkpoint_path: Path to the checkpoint file for the Inception graph.
    """
    # Get all variables to restore. Exclude Logits and AuxLogits because they
    # depend on the input data and we do not need to intialize them from
    # checkpoint.
    scope = ''.join([x.capitalize() for x in self.version.split('_')])
    all_vars = tf.contrib.slim.get_variables_to_restore(
            exclude=['%s/AuxLogits' % scope, '%s/Logits' % scope, 'global_step'])

    saver = tf.train.Saver(all_vars)
    saver.restore(self.tf_session, checkpoint_path)

  def calculate_embedding(self, batch_image_bytes):
    """Get the embeddings for a given JPEG image.

    Args:
      batch_image_bytes: As if returned from [ff.read() for ff in file_list].

    Returns:
      The Inception embeddings (bottleneck layer output)
    """
    return self.tf_session.run(
        self.embedding, feed_dict={self.input_jpeg: batch_image_bytes})


class ImageEmbeddingService:
    def __init__(self, version):
        tf_session = tf.InteractiveSession()
        self.graph = EmbeddingsGraph(tf_session, version)

    def dim(self):
        return self.graph.dim()

    def get_image_bytes(self, uri):
        def _open_file_read_binary(uri):
            try:
                return file_io.FileIO(uri, mode='rb')
            except errors.InvalidArgumentError:
                return file_io.FileIO(uri, mode='r')

        try:
          if uri.startswith('https://'):
              image_bytes = urlopen(uri, timeout=5).read()
          else:
              with _open_file_read_binary(uri) as f:
                  image_bytes = f.read()
          img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except HTTPError as e:
          logging.warning('%s(%s) uri: %s', e.msg, e.code, uri)
          if e.code == 403:
            return
          raise e

        # A variety of different calling libraries throw different exceptions here.
        # They all correspond to an unreadable file so we treat them equivalently.
        except Exception as e:  # pylint: disable=broad-except
          logging.exception('Error processing image %s: %s', uri, str(e))
          raise e

        # Convert to desired format and output.
        output = io.BytesIO()
        img.save(output, 'jpeg')
        image_bytes = output.getvalue()
        return image_bytes

    def get_embedding(self, url):
        image_bytes = self.get_image_bytes(url)
        if image_bytes is None:
            return
        embedding = self.graph.calculate_embedding(image_bytes)
        return np.squeeze(embedding)

if __name__ == '__main__':
    url = ''
    service = ImageEmbeddingService()
    embedding = service.get_embedding(url)
    print(embedding)
    print(embedding.shape)

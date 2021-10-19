import tensorflow as tf
import tensorflow_datasets as tfds

# Construct a tf.data.Dataset
ds = tfds.load('sun397', split='train', shuffle_files=True)

print("done")

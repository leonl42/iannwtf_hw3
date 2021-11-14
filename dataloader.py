import tensorflow as tf
import tensorflow_datasets as tfds
import os


class DataLoader:
  
  def save_data(self, path):
    for data in self._data.items():
      print("save: ", data)
      directory = path+data[0]+"/"
      if not os.path.exists(directory):
        os.makedirs(directory)
      tf.data.experimental.save(data[1], directory, compression=None, shard_func=None, checkpoint_args=None) 

      
  def load_data(self, path):
    data_dict = {}
    for dir in os.listdir(path):
      if dir.startswith("."):
        continue
      data_dict[dir] = tf.data.experimental.load(path+dir+"/", element_spec=None, compression=None, reader_func=None)
    self._data = data_dict


  def get_data(self):
    return self._data


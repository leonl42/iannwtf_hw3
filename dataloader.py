import tensorflow as tf
import os


class DataLoader:

  
  def set_data(self, data):
    self._data = data
  
  def save_data(self, path):
    for data in self._data.items():
      directory = path+data[0]+"/"
      if not os.path.exists(directory):
        os.makedirs(directory)
      print(data[1])
      tf.data.experimental.save(data[1], directory) 

      
  def load_data(self, path):
    data_dict = {}
    for dir in os.listdir(path):
      if dir.startswith("."):
        continue
      data_dict[dir] = tf.data.experimental.load(path+dir+"/")
    self._data = data_dict


  def get_data(self):
    return self._data


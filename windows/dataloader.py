import tensorflow as tf
import tensorflow_datasets as tfds
import os


class DataLoader:
  """
  Class for loading and saving multiple tensorflow Datasets

    Variables:
      - _data: Dictionary containing all datasets of this class. The key for a dataset is its name.
               Ensure that a name never starts with ".", otherwise the dataset can't be loaded again.

    Functions:
      - set _data
      - set_data: Setter for the _data variable
      - get_data: Getter for the _data variable
      - save_data: Save all currently stored datasets in a given path
      - load_data: Load all datasets from a given path
  """

  def save_data(self, path):
    """
    Save all datasets which are currently stored in _data

      Args:
        - path: Where to store the datasets
    """

    for data in self._data.items():

      # the directory where this dataset will be stored is the given path + its key in the dictionary
      directory = path+data[0]+"/"

      # create path if not already existing
      if not os.path.exists(directory):
        os.makedirs(directory)
      
      # save the dataset in the path
      tf.data.experimental.save(data[1], directory) 

      
  def load_data(self, path):
    """
    Load all datasets which have been stored in a given path

      Args:
        - path: From where to load the data. Ensure that only folders/files which have been saved by this class are in the directory
                If there is a need to store other things, make sure that the directory/file starts with a ".".
    """

    data_dict = {}
    for dir in os.listdir(path):

      # exclude all folders/files that start with a "." from loading
      if dir.startswith("."):
        continue

      # load stored data. The respective subfolder in the given path will be the name of the dataset
      data_dict[dir] = tf.data.experimental.load(path+dir+"/")
      
    self.set_data(data_dict)


  def set_data(self, data):
    """
    Setter for setting the data
    
      Args:
        - data: data to set
    """

    self._data = data


  def get_data(self):
    """
    Getter for the data

      Returns:
        - the currently stored data
    
    """

    return self._data


import tensorflow as tf

class MyDense(tf.keras.layers.Layer):
  """
      Our own custom layer function, which inherits from the keras.layers.Layer class

        Functions:
          init: constructor
          call: calculates output tensor for this layer
          build: creates weights and bias when call is the first time run
  """

  def __init__(self, units, activation=tf.nn.softmax):
    """
      Constructs an fully connected layer.

        Args:
          units: perceptrons of our layer
          activation: activation function of our layer
    """

    super(MyDense, self).__init__()

    self.units = units
    self.activation = activation


  def call(self, inputs):
    """
      Calculates the output of our layer. (forwads_step)

        Args:
          inputs: input tensor of our layer

        Returns:
          x: the output of our layer
    """

    x = tf.matmul(inputs, self.w) + self.b
    x = self.activation(x)

    return x


  def build(self,input_shape):
    """
      Creates random weights and bias from a normal distribution for our layer.

        Args:
          input_shape: dimension of our input-tensor
    """

    self.w = self.add_weight(shape=(input_shape[-1],self.units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                            initializer='random_normal',
                            trainable=True)



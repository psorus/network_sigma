#from https://stackoverflow.com/questions/55282481/keras-custom-layer-without-inputs
from tensorflow.keras.layers import Layer
import tensorflow.keras as keras

class Simple(Layer):


    def __init__(self, output_dim, **kwargs):
       self.output_dim = output_dim
       super(Simple, self).__init__(**kwargs)

    def build(self, input_shapes):
       #self.kernel = self.add_weight(name='kernel', shape=self.output_dim, initializer=keras.initializers.TruncatedNormal(0,0.1), trainable=True)

        self.kernel = self.add_weight(shape=self.output_dim,
                               initializer='random_normal',
                               trainable=True,
                               name="kernel")

        super(Simple, self).build(input_shapes)  

    def call(self, inputs):
       return self.kernel

    def compute_output_shape(self):
       return self.output_dim

#X = Simple((1, 784))([])
#print(X.shape)

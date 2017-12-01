from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D
from keras.layers import concatenate, Activation, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Lambda
from keras import backend as K

from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, EarlyStopping


# refer from : https://gist.github.com/melgor/0e43cadf742fe3336148ab64dd63138f
def convBNRelu(input, filters, kernel_size, stride, layer_id, padding='same'):
    x = Conv2D(filters, (kernel_size, kernel_size), activation='relu', padding='same', name='conv1_' + s_id)(x)
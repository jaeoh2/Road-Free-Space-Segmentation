from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D
from keras.layers import concatenate, Activation, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Lambda
from keras import backend as K

from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

def unpool_with_argmax(pool, ind, name = None, ksize=[1,2,2,1]):
    #refer from : https://github.com/mathildor/TF-SegNet/blob/master/AirNet/layers.py
    with tf.variable_scope(name):
        input_shape = pool.get_shape().as_list()
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

        flat_input_size = np.prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
    
    return ret
    

def max_pool_with_argmax(x):
    return tf.nn.max_pool_with_argmax(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='max_pool_with_argmax')[0]

def convBnRelu(x, filters=64):
    
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    
    return x

def segnet(nb_classes=12, input_shape=(480,480,3)):
    inputs = Input((input_shape))
    
    #Encoders
    x = convBnRelu(inputs, filters=64)
    x = convBnRelu(x, filters=64)
    x = MaxPooling2D(pool_size=(2,2))(x)
        
    x = convBnRelu(x, filters=128)
    x = convBnRelu(x, filters=128)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    
    x = convBnRelu(x, filters=256)
    x = convBnRelu(x, filters=256)
    x = convBnRelu(x, filters=256)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    x = convBnRelu(x, filters=512)
    x = convBnRelu(x, filters=512)
    x = convBnRelu(x, filters=512)
#     x = MaxPooling2D(pool_size=(2,2))(x)
    
    #Decoder
#     x = UpSampling2D(size=(2,2))(x)
    x = convBnRelu(x, filters=512)
    x = convBnRelu(x, filters=512)
    x = convBnRelu(x, filters=512)
        
    x = UpSampling2D(size=(2,2))(x)
    x = convBnRelu(x, filters=256)
    x = convBnRelu(x, filters=256)
    x = convBnRelu(x, filters=256)
    
    x = UpSampling2D(size=(2,2))(x)
    x = convBnRelu(x, filters=128)
    x = convBnRelu(x, filters=128)
    
    x = UpSampling2D(size=(2,2))(x)
    x = convBnRelu(x, filters=64)
    x = convBnRelu(x, filters=64)
    
    x = Conv2D(nb_classes, (1,1), activation='softmax', padding='same', name='output')(x)

    model = Model(inputs=inputs, outputs=x)
    model.summary()
    
    parallel_model = multi_gpu_model(model, gpus=8)

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    parallel_model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    return parallel_model, model

def segnet1(nb_classes=12, input_shape=(480,480,3)):
    inputs = Input((input_shape))
    
    #Encoders
    x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
        
    x = Conv2D(64*2, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*2, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    x = Conv2D(64*4, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*4, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*4, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    x = Conv2D(64*8, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*8, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*8, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    #Decoder
#     x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(64*8, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*8, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*8, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(64*4, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*4, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*4, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(64*2, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64*2, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(nb_classes, (1,1), activation='softmax', padding='same', name='output')(x)

    model = Model(inputs=inputs, outputs=x)
    model.summary()
    
    parallel_model = multi_gpu_model(model, gpus=8)

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    parallel_model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    return parallel_model, model
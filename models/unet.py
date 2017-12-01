from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Cropping2D
from keras.layers import concatenate, Activation, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Lambda
from keras import backend as K

from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def encoder(x, layer_id, filters=64):
    s_id = 'encoder' + str(layer_id)
    
    x = Conv2D(filters, (3,3), activation='relu', padding='same', name='conv1_' + s_id)(x)
    x = Conv2D(filters, (3,3), activation='relu', padding='same', name='conv2_' + s_id)(x)
    x = BatchNormalization(name='BN_' + s_id)(x)
    xp = MaxPooling2D(pool_size=(2,2), name='pool_' + s_id)(x)
    
    return xp, x

def decoder(xp, x, layer_id, filters=32, cropfilters=((0,0),(0,0))):
    s_id = 'decoder' + str(layer_id)
    
    x = Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same', name='dconv_' + s_id)(x)
    xp = Cropping2D(cropping=(cropfilters))(xp)
    x = concatenate([xp, x], axis=-1, name='concat'+s_id)
    x = Conv2D(filters, (3,3), activation='relu', padding='same', name='conv1_' + s_id)(x)
    x = Conv2D(filters, (3,3), activation='relu', padding='same', name='conv2_' + s_id)(x)
    
    return x

def unet(nb_classes=32, input_shape=(480,480,3)):
    inputs = Input((input_shape))
    x, x1 = encoder(inputs, layer_id=1, filters=64)
    x, x2 = encoder(x, layer_id=2, filters=128)
    x, x3 = encoder(x, layer_id=3, filters=256)
#     x, x4 = encoder(x, layer_id=4, filters=512)

#     x = Conv2D(1024, (3,3), activation='relu', padding='same', name='conv_layer5')(x)
    x = Conv2D(512, (3,3), activation='relu', padding='same', name='conv_layer5')(x)

#     x = decoder(x4, x, layer_id=4, filters=512)
    x = decoder(x3, x, layer_id=3, filters=256)
    x = decoder(x2, x, layer_id=2, filters=128)
    x = decoder(x1, x, layer_id=1, filters=64)

    x = Conv2D(nb_classes, (1,1), activation='softmax', padding='same', name='output')(x)

    model = Model(inputs=inputs, outputs=x)
    model.summary()

    parallel_model = multi_gpu_model(model, gpus=8)
    parallel_model.compile(optimizer='adam', loss=[dice_coef_loss], metrics=[dice_coef])
    model.compile(optimizer='adam', loss=[dice_coef_loss], metrics=[dice_coef])
    
    return parallel_model, model
#     return model
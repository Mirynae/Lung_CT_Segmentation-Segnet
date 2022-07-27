from numpy import ndarray, float64
from keras import backend as K
from keras import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D

def normalization(data:ndarray):
    # floating
    data = float64(data)
    
    # normalization
    for i in range(data.shape[0]):
        data[i] -= data[i].min()
        max_value = data[i].max()
        if max_value != 0:
            data[i] /= max_value
    
    return data
    
def dice_coef(y_true, y_pred):
    
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-7) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-7)
def dice_coef_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return 1-(2. * intersection + 1e-7) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-7)

def segnet(channels, kernel_size, padding, activation, input_shape):
    model = Sequential()
    # Encoder
    model.add(Conv2D(channels, kernel_size=kernel_size, padding=padding, activation=activation, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(channels, kernel_size=kernel_size, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))

    model.add(Conv2D(channels*2, kernel_size=kernel_size, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(Conv2D(channels*2, kernel_size=kernel_size, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))

    model.add(Conv2D(channels*4, kernel_size=kernel_size, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(Conv2D(channels*4, kernel_size=kernel_size, padding=padding, activation=activation))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))

    # Decoder
    model.add(UpSampling2D(size=2))
    model.add(Conv2D(channels*4, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(channels*4, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=2))
    model.add(Conv2D(channels*2, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(channels*2, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size=2))
    model.add(Conv2D(channels, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())
    model.add(Conv2D(channels, kernel_size=kernel_size, activation=activation, padding=padding))
    model.add(BatchNormalization())

    # Output
    model.add(Conv2D(1,kernel_size=1, activation='sigmoid'))
    model.summary(line_length=120)
    return model

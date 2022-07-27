import numpy as np
import custom_utils
from tensorflow import device

# Load
x_train = np.load('dataset/x_train.npy')
x_val = np.load('dataset/x_val.npy')
y_train = np.load('dataset/y_train.npy')
y_val = np.load('dataset/y_val.npy')

# Processing
x_train = custom_utils.normalization(x_train)
x_val = custom_utils.normalization(x_val)
y_train = custom_utils.normalization(y_train)
y_val = custom_utils.normalization(y_val)


# Create model
model = custom_utils.segnet(channels=32, kernel_size=3, padding='same', activation='relu', input_shape=x_train[0].shape)
model.compile(optimizer='adam', loss=[custom_utils.dice_coef_loss], metrics=[custom_utils.dice_coef])

# Learning
epochs = 100
with device('/GPU:2'):
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=epochs, batch_size=8)
    
model.save('Segnet')
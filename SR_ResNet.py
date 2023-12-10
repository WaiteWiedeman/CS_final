# import packages
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers

AUTOTUNE = tf.data.AUTOTUNE

from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_lfw_pairs
from skimage.transform import resize
from sklearn.model_selection import train_test_split

# define class to load lfw dataset
class data_loader():
    def __init__(self,color):
        # use "fetch_lfw_people" to import dataset
        self.lfw_people = fetch_lfw_people(data_home="./data", color=color, resize=None, min_faces_per_person=70,
                                           funneled=False,)
        # make train/ test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.lfw_people.images,
                                                                                self.lfw_people.target,
                                                                                test_size=0.25,
                                                                                random_state=42)
        # make train/validation split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                              test_size=0.10, random_state=42)
    # function to return training features and targets
    def get_train(self):
        return self.X_train, self.y_train
    # function to return validation features and targets
    def get_val(self):
        return self.X_val, self.y_val
    # function to return testing features and targets
    def get_test(self):
        return self.X_test, self.y_test
    # function to resize images
    def get_resized(self, split="train", size=[25,25,3]):
        if split=="train":
            X_resized = resize(self.X_train, (self.X_train.shape[0], size[0], size[1], size[2]))
        if split=="test":
            X_resized = resize(self.X_test, (self.X_test.shape[0], size[0], size[1], size[2]))
        if split=="val":
            X_resized = resize(self.X_val, (self.X_val.shape[0], size[0], size[1], size[2]))
        return X_resized

data_set = data_loader(color=True)  # declare variable for dataset
# get training, validation, and testing data
X_train, y_train = data_set.get_train()
X_val, y_val = data_set.get_val()
X_test, y_test = data_set.get_test()
# convert images to integars from floats
X_train = (X_train*255).astype(np.uint8)
X_val = (X_val*255).astype(np.uint8)
X_test = (X_test*255).astype(np.uint8)
# create variable for low res and high res data to train model
X_train_lr = resize(X_train, (X_train.shape[0], 16, 11, 3))
X_train_hr = resize(X_train, (X_train.shape[0], 144,99,3), anti_aliasing=True, order=0)

X_val_lr = resize(X_val, (X_val.shape[0], 16, 11, 3))
X_val_hr = resize(X_val, (X_val.shape[0], 144,99,3), anti_aliasing=True, order=0)

X_test_lr = resize(X_test, (X_test.shape[0], 16, 11, 3))
X_test_hr = resize(X_test, (X_test.shape[0], 144,99,3), anti_aliasing=True, order=0)
# print data shapes to check
print(X_train_lr.shape)
print(X_train_hr.shape)

# plot images from dataset
# plot 9 LR and HR images
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(X_train_hr[i])
    plt.title(X_train_hr[i].shape)
    plt.axis("off")

# Low Resolution Images
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(X_train_lr[i])
    plt.title(X_train_lr[i].shape)
    plt.axis("off")

def PSNR(super_resolution, high_resolution):
    """Compute the peak signal-to-noise ratio, measures quality of image."""
    # Max value of pixel is 255
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)[0]
    return psnr_value

# create class for model
class EDSRModel(tf.keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    # prediction takes one image
    def predict_step(self, x):
        # Adding dummy dimension using tf.expand_dims and converting to float32 using tf.cast
        x = tf.cast(tf.expand_dims(x, axis=0), tf.float32)
        # Passing low resolution image to model
        super_resolution_img = self(x, training=False)
        # Clips the tensor from min(0) to max(255)
        super_resolution_img = tf.clip_by_value(super_resolution_img, 0, 255)
        # Rounds the values of a tensor to the nearest integer
        super_resolution_img = tf.round(super_resolution_img)
        # Removes dimensions of size 1 from the shape of a tensor and converting to uint8
        super_resolution_img = tf.squeeze(
            tf.cast(super_resolution_img, tf.uint8), axis=0
        )
        return super_resolution_img


# Residual Block
def ResBlock(inputs):
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.Add()([inputs, x])
    return x


# Upsampling Block
def Upsampling(inputs, factor=3, **kwargs):
    x = layers.Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(inputs)
    x = tf.nn.depth_to_space(x, block_size=factor)
    x = layers.Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(x)
    x = tf.nn.depth_to_space(x, block_size=factor)
    return x


def make_model(num_filters, num_of_residual_blocks):
    # Flexible Inputs to input_layer
    input_layer = layers.Input(shape=(None, None, 3))
    # Scaling Pixel Values
    x = layers.Rescaling(scale=1.0 / 255)(input_layer)
    x = x_new = layers.Conv2D(num_filters, 3, padding="same")(x)

    # 16 residual blocks
    for _ in range(num_of_residual_blocks):
        x_new = ResBlock(x_new)

    x_new = layers.Conv2D(num_filters, 3, padding="same")(x_new)
    x = layers.Add()([x, x_new])

    x = Upsampling(x)
    x = layers.Conv2D(3, 3, padding="same")(x)

    output_layer = layers.Rescaling(scale=255)(x)
    return EDSRModel(input_layer, output_layer)


model = make_model(num_filters=64, num_of_residual_blocks=16)

# Using adam optimizer with initial learning rate as 1e-4, changing learning rate after 5000 steps to 5e-5
optim_edsr = keras.optimizers.Adam(
    learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[5000], values=[1e-4, 5e-5]
    )
)
# Compiling model with loss as mean absolute error(L1 Loss) and metric as psnr
model.compile(optimizer=optim_edsr, loss="mae", metrics=[PSNR])
# Training for more epochs will improve results
history = model.fit(X_train_lr, X_train_hr, epochs=100, steps_per_epoch=200, validation_data=(X_val_lr,X_val_hr))
model.save('SRResNet.keras')

# print test shape as check
print(X_test_lr.shape)

# function to plot HR, LR, and prediction images together
def plot_results(highres, lowres, preds):
    """
    Displays low resolution image and super resolution image
    """
    plt.figure(figsize=(24, 14))
    plt.subplot(131), plt.imshow(highres), plt.title("High resolution")
    plt.subplot(132), plt.imshow(lowres), plt.title("Low resolution")
    plt.subplot(133), plt.imshow(preds), plt.title("Prediction")
    plt.savefig('prediction.png') # save image
    plt.show()

preds = model.predict_step(X_test_lr[0])
plot_results(X_test_hr[0], X_test_lr[0], preds)

# plot metrics from training
# loss plot
plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend()

plt.savefig('SR_loss-plt.png')
# PSNR plot
plt.figure()
plt.plot(history.history['PSNR'], label='PSNR')
plt.plot(history.history['val_PSNR'], label='val_PSNR')
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch #')
plt.ylabel('PSNR')
plt.legend()

plt.savefig('SR_PSNR-plt.png')

# import packages for FR model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
# from data_loader import data_loader
# from model import FacialRecognitionModel
from keras.utils import to_categorical
from skimage.transform import resize

# create class for FR model
class FacialRecognitionModel:
    def __init__(self, in_shape=(144,99,3)):
        self._model = Sequential()

        self._model.add(Conv2D(32, (5,5), activation='relu', input_shape=in_shape))
        self._model.add(MaxPooling2D(pool_size=(2,2)))
        self._model.add(Conv2D(64, (3,3,) , activation='relu'))
        self._model.add(MaxPooling2D(pool_size=(2,2)))
        self._model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
        self._model.add(MaxPooling2D(pool_size=(2,2)))
        self._model.add(Flatten())
        self._model.add(Dense(units=256, activation='relu'))
        self._model.add(Dense(units=7, activation='softmax'))

        self._model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val):
        self._model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val))

    def test(self, X_test, y_test):
        score = self._model.evaluate(X_test, y_test, verbose=0)
        print(f'Test loss: {score[0]:.4f}')
        print(f'Test accuracy: {score[1]:.4f}')

    def save(self):
        self._model.save("fr_model.keras")

    def load(self):
        self._model = tf.keras.models.load_model("fr_model.keras")

# print length of training data as check
print(len(X_train_hr))

data_set = data_loader(color=True)

# Build and Train FR model on HR data split
fr_model = FacialRecognitionModel()
fr_model.train(X_train_hr, to_categorical(y_train),
               X_val_hr, to_categorical(y_val))

# Save fr_model
fr_model.save()

# get various LR image sizes
X_test_8 = resize(X_test, (X_test.shape[0], 16, 11, 3))
X_test_6 = resize(X_test, (X_test.shape[0], 21, 16, 3))
X_test_4 = resize(X_test, (X_test.shape[0], 31, 24, 3))
X_test_2 = resize(X_test, (X_test.shape[0], 63, 47, 3))

# make predictions for each LR size
predsby8 = np.zeros((X_test.shape[0], 144, 99, 3), dtype=np.uint8)
predsby6 = np.zeros((X_test.shape[0], 189, 144, 3), dtype=np.uint8)
predsby4 = np.zeros((X_test.shape[0], 279, 216, 3), dtype=np.uint8)
predsby2 = np.zeros((X_test.shape[0], 567, 423, 3), dtype=np.uint8)
# add prediction to empty lists declared above
for im in range(len(X_test)):
    predsby8[im] = model.predict_step(X_test_8[im])
    predsby6[im] = model.predict_step(X_test_6[im])
    predsby4[im] = model.predict_step(X_test_4[im])
    predsby2[im] = model.predict_step(X_test_2[im])
# print shapes
'''
print(predsby8.shape)
print(predsby6.shape)
print(predsby4.shape)
print(predsby2.shape)
'''
# resize prediction images for FR model
predsby8 = resize(predsby8, (predsby8.shape[0], 144,99,3), anti_aliasing=True, order=0)
predsby6 = resize(predsby6, (predsby6.shape[0], 144,99,3), anti_aliasing=True, order=0)
predsby4 = resize(predsby4, (predsby4.shape[0], 144,99,3), anti_aliasing=True, order=0)
predsby2 = resize(predsby2, (predsby2.shape[0], 144,99,3), anti_aliasing=True, order=0)
# test FR model on SR images
fr_model.test(predsby8,to_categorical(y_test))
fr_model.test(predsby6,to_categorical(y_test))
fr_model.test(predsby4,to_categorical(y_test))
fr_model.test(predsby2,to_categorical(y_test))
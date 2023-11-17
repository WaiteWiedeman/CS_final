from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from model import create_gen, create_comb, create_disc, build_vgg
from keras.layers import Input



lfw_people = fetch_lfw_people(resize=None,min_faces_per_person=70, color=True,slice_=None)
#print(lfw_people.DESCR)

print(lfw_people.images[0].shape)
print(lfw_people.target[0])
# plots images

fig, arr = plt.subplots(1, 2, figsize=(15, 15))
arr[0].imshow(lfw_people.images[0])
arr[0].set_title(lfw_people.target_names[0])
arr[1].imshow(lfw_people.images[1])
arr[1].set_title(lfw_people.target_names[1])
plt.show()
target_shape_hr_img = [128, 128, 3]
target_shape_lr_img = [25, 25, 3]

i_h, i_w, i_c = target_shape_hr_img
i_h_lr, i_w_lr, i_c_lr = target_shape_lr_img

m = len(lfw_people.images)  # number of images
X = np.zeros((m, i_h, i_w, i_c), dtype=np.float32)
y = np.zeros((m, i_h_lr, i_w_lr, i_c_lr), dtype=np.float32)

for i in range(len(lfw_people.images)):
    single_img = np.resize(lfw_people.images[i], (i_h, i_w, i_c))
    X[i] = single_img
for j in range(len(lfw_people.images)):
    single_img = np.resize(lfw_people.images[j], (i_h_lr, i_w_lr, i_c_lr))
    y[j] = single_img

lr_ip = Input(shape=(25,25,3))
hr_ip = Input(shape=(128,128,3))
train_lr, test_lr, train_hr, test_hr = train_test_split(X, y, test_size=0.1, random_state=123)#training images arrays normalized between 0 & 1

'''
generator = create_gen(lr_ip)
discriminator = create_disc(hr_ip)
discriminator.compile(loss="binary_crossentropy", optimizer="adam",
  metrics=['accuracy'])
vgg = build_vgg()
vgg.trainable = False
gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)
gan_model.compile(loss=["binary_crossentropy","mse"], loss_weights=
  [1e-3, 1], optimizer="adam")
batch_size = 20
train_lr_batches = []
train_hr_batches = []
for it in range(int(train_hr.shape[0] / batch_size)):
    start_idx = it * batch_size
    end_idx = start_idx + batch_size
    train_hr_batches.append(train_hr[start_idx:end_idx])
    train_lr_batches.append(train_lr[start_idx:end_idx])
train_lr_batches = np.array(train_lr_batches)
train_hr_batches = np.array(train_hr_batches)

epochs = 100
for e in range(epochs):
    gen_label = np.zeros((batch_size, 1))
    real_label = np.ones((batch_size, 1))
    g_losses = []
    d_losses = []
    for b in range(len(train_hr_batches)):
        lr_imgs = train_lr_batches[b]
        hr_imgs = train_hr_batches[b]
        gen_imgs = generator.predict_on_batch(lr_imgs)
        # Dont forget to make the discriminator trainable
        discriminator.trainable = True

        # Train the discriminator
        d_loss_gen = discriminator.train_on_batch(gen_imgs,
                                                  gen_label)
        d_loss_real = discriminator.train_on_batch(hr_imgs,
                                                   real_label)
        discriminator.trainable = False
        d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)
        image_features = vgg.predict(hr_imgs)

        # Train the generator
        g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs],
                                                [real_label, image_features])

        d_losses.append(d_loss)
        g_losses.append(g_loss)
    g_losses = np.array(g_losses)
    d_losses = np.array(d_losses)

    g_loss = np.sum(g_losses, axis=0) / len(g_losses)
    d_loss = np.sum(d_losses, axis=0) / len(d_losses)
    print("epoch:", e + 1, "g_loss:", g_loss, "d_loss:", d_loss)

    label = np.ones((len(test_lr), 1))
    test_features = vgg.predict(test_hr)
    eval, _, _ = gan_model.evaluate([test_lr, test_hr], [label, test_features])

    test_prediction = generator.predict_on_batch(test_lr)
'''

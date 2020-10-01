import os
import tensorflow as tf
tf.get_logger().setLevel(3)
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
from keras import models

# The pretrained model
from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

url = 'https://github.com/gpk2000/Kitty-vs-Doggo/blob/master/pretrained_nodataug.h5?raw=true'
filename, headers = urllib.request.urlretrieve(url, filename="/content/model.h5")

for fname in os.listdir('/content'):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = image.load_img(os.path.join('/content', fname), target_size=(150, 150))
        img = image.img_to_array(img)
        img = np.array([img])
        img = img / 255
        plt.xticks([])
        plt.yticks([])
        _ = plt.imshow(img.reshape((150, 150, 3)))
        features = conv_base.predict(img)
        features = features.reshape((1, 4*4*512))
        model = models.load_model('/content/model.h5') 
        acc = model.predict(features)
        if acc < 0.5:
            accuracy_per = (1 - acc) * 100
            print("The model is %.2f percent sure that the below image is a cat" % accuracy_per)
            os.remove(os.path.join('/content', fname))
        else:
            accuracy_per = acc * 100
            print("The model is %.2f percent sure that the below image is a dog" % accuracy_per)
            os.remove(os.path.join('/content', fname))
        plt.show()
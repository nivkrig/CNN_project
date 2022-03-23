```
import tensorflow as tf
```
```
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
```
```
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import sklearn
import numpy as np
import cv2
import json
import h5py
```
```
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```
```
num_filters = 60
size_filter1 = (5, 5)
size_filter2 = (3, 3)
size_pool = (2, 2)
num_node = 500
```
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(num_filters, size_filter1, activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(num_filters, size_filter1, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32//2, size_filter2, activation="relu"),
    tf.keras.layers.Conv2D(32//2, size_filter2, activation="relu"),
    tf.keras.layers.MaxPool2D(pool_size=size_pool),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_node, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax")

])
```
```
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```
```
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

```
```
history = model.fit(x_train, y_train, epochs=6)
model.evaluate(x_test,  y_test, verbose=2)
```
```
### Model saves to a .py file
```

```
model.save('model')
```

```
cv2.imshow('test', x_train[0])
cv2.waitKey()

example = x_train[0]
predict =  model.predict(example.reshape(1, 28, 28, 1))
predict = np.argmax(predict)
print(predict)
```

```
### In a separate document
```

```
import numpy as np
import cv2
import json
import h5py
import tensorflow as tf
from tensorflow.keras.models import model_from_json
print(h5py.__version__)
```
```
### opening the file crashes my computers
```
```
model = keras.models.load_model('model')

```

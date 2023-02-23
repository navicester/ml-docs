https://medium.com/analytics-vidhya/object-localization-using-keras-d78d6810d0be

In this blog, I will explain the task of object localization and how to implement a CNN based architecture that solves this task.

Resources:
- code: https://colab.research.google.com/drive/169pJ-xECBWDW9Q92naaNE3oRyBr7D-uh#scrollTo=IjFHABNA7nZL
- images: https://drive.google.com/drive/folders/1YsxDywjUZi-l_YgzXt__9__eUU-I_6EH?usp=sharing

## Agenda
- What is Object Localization?
- How to perform Object Localization using Deep Neural Networks
- step-by-step Keras implementation

## Object Localization
Object localization is the name of the task of “classification with localization”. Namely, given an image, classify the object that appears in it, and find its location in the image, usually by using a bounding-box. In Object Localization, only a single object can appear in the image. If more than one object can appear, the task is called “Object Detection”.

![Classification vs Object Localization](https://user-images.githubusercontent.com/16224205/220897230-92714ca7-c95c-4e41-a910-38d15ce7cb27.png)

Figure: Classification vs Object Localization


## How to perform Object Localization using DNNs?
Object Localization can be treated as a regression problem - predicting a continuous value, such as a weight or a salary. For instance, we can represent our output (a bounding-box) as a tuple of size 4, as follows:

(x,y, height, width)
- (x,y): the coordination of the left-top corner of the bounding box
- height: the height of the bounding box
- width: the width of the bounding box

![(x,y,height,width) as a bounding-box representation](https://user-images.githubusercontent.com/16224205/220897755-d0968b50-9089-4ff1-bebf-ae469ae6488e.png)

Figure: (x,y,height,width) as a bounding-box representation

Hence, we can use a CNN architecture that will output 4 numbers, but what will be our architecture?

First, we will notice that these 4 numbers have some constraints: (x.y) must be inside the image and so do x+width and y+height. We will scale the image width and height to be 1.0. To make sure that the CNN outputs will be in the range `[0,1]`, we will use the **sigmoid** activation layer- it will enforce that (x,y) will be inside the image, but not necessarily x+width and y+height. This property will be learned by the network during the training process.

suggested architecture:

![Given an image, the network will output a 4 numbers representation bounding box](https://user-images.githubusercontent.com/16224205/220898016-a1b8bc76-df31-49d7-a484-74986464aac1.png)

Figure: Given an image, the network will output a 4 numbers representation bounding box

What about the loss function? the output of a sigmoid can be treated as probabilistic values, and therefore we can use binary_crossentropy loss.

![binary_crossentropy formula](https://user-images.githubusercontent.com/16224205/220898364-b2a58721-036d-4322-af39-2b56f58a92f9.png)

Figure: binary_crossentropy formula

Usually, this loss is being used with binary values as the ground-truth ({0,1}), but it doesn’t have to be this way- we can use values from [0,1]. For our use, the ground-truth values are indeed in the range [0,1], since it represents location inside an image and dimensions.

## Step-by-Step Keras implementation

We will solve an Object Localization task in three steps:  
1) Synthetic usecase- Detect white blobs on a black background  
2) semi-synthetic usecase- Detect cats on a black background  
3) final usecase- Detect cats on a natural background  

### 1. Synthetic usecase:
first of all, we will import and define the followings:

``` python
%tensorflow_version 2.x
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.utils import  plot_model
import matplotlib.pyplot as plt

BATCH_SIZE = 64
EPOCH_SIZE = 64
```

Then, we will define our CNN based model. We will perform transfer learning: using a pre-trained version of VGG.

``` python
# transfer learning - load pre-trained vgg and replace its head
vgg = tf.keras.applications.VGG16(input_shape=[128, 128, 3], include_top=False, weights='imagenet')
x = Flatten()(vgg.output)
x = Dense(3, activation='sigmoid')(x)
model1 = Model(vgg.input, x)
model1.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))
# plot the model
plot_model(model1, "first_model.png",show_shapes=True,expand_nested=False)
```

![The model’s architecture](https://user-images.githubusercontent.com/16224205/220899122-2bdd6d9b-fc33-499b-9338-040a000e86e9.png)

Our next step will be creating a dataset using a python generator. It will create batches of white blobs over black background.

``` python
from matplotlib.patches import Circle

def synthetic_gen(batch_size=64):
  # enable generating infinite amount of batches
  while True:
      # generate black images in the wanted size
      X = np.zeros((batch_size, 128, 128, 3))
      Y = np.zeros((batch_size, 3))
      # fill each image
      for i in range(batch_size):
        x = np.random.randint(8,120)
        y = np.random.randint(8,120)
        a = min(128 - max(x,y), min(x,y))
        r = np.random.randint(4,a)
        for x_i in range(128):
          for y_i in range(128):
            if ((x_i - x)**2) + ((y_i - y)**2) < r**2:
              X[i, x_i, y_i,:] = 1
        Y[i,0] = (x-r)/128.
        Y[i,1] = (y-r)/128.
        Y[i,2] = 2*r / 128.
      yield X, Y

# sanity check - plot the images
x,y = next(synthetic_gen())
plt.imshow(x[0])
```

X is a batch of images and Y is the ground-truth bounding boxes. for each image, we will create a white blob, and we will make sure that it will be inside the borders of the image.

![an example of the circle generator](https://user-images.githubusercontent.com/16224205/220899579-35dde09f-70e8-4462-a48e-ad63c33f2c3f.png)

Figure: an example of the circle generator

Now, we can train the model to predict the ground truth bounding-box, and visualize its performance:

``` python
# needs steps per epoch since the generator is infinite
model1.fit_generator(synthetic_gen(),steps_per_epoch=EPOCH_SIZE,epochs=5)
```

``` python
from matplotlib.patches import Rectangle

# given image and a label, plots the image + rectangle
def plot_pred(img,p):
  fig, ax = plt.subplots(1)
  ax.imshow(img)
  rect = Rectangle(xy=(p[1]*128,p[0]*128),width=p[2]*128, height=p[2]*128, linewidth=1,edgecolor='g',facecolor='none')
  ax.add_patch(rect)
  plt.show()


# generate new image
x, _ = next(synthetic_gen())
# predict
pred = model1.predict(x)
# examine 1 image
im = x[0]
p = pred[0]
plot_pred(im,p)
```

![image](https://user-images.githubusercontent.com/16224205/220899943-d600dc91-eb70-4291-90f6-a2cb02ef6165.png)

### 2. Semi-synthetic usecase:
For this usecase, we will use the same architecture, with a different type of images - images of a cat over a black background. This will be achieved using a different generator:

``` python
from PIL import Image
from matplotlib.patches import Circle

cat_pil = Image.open("cat.png")
cat_pil = cat_pil.resize((64,64))
cat = np.asarray(cat_pil)

def cat_gen(batch_size=64):
  # enable generating infinite amount of batches
  while True:
      # generate black images in the wanted size
      X = np.zeros((batch_size, 128, 128, 3))
      Y = np.zeros((batch_size, 3))
      # fill each image
      for i in range(batch_size):
        # resize the cat
        size = np.random.randint(32,64)
        temp_cat = cat_pil.resize((size,size))
        cat = np.asarray(temp_cat) / 255.
        cat_x, cat_y, _ = cat.shape
        # create a blank background image
        bg = Image.new('RGB', (128, 128))
        # generate 
        x1 = np.random.randint(1,128 - cat_x)
        y1 = np.random.randint(1,128 - cat_y)
        # paste the cat over the image
        bg.paste(temp_cat, (x1, y1))
        cat = np.asarray(bg) / 255. # transform into a np array
        X[i] = cat

        Y[i,0] = x1/128.
        Y[i,1] = y1/128.
        Y[i,2] = cat_x / 128.
      yield X, Y

# plot the images
x,y = next(cat_gen())
plt.imshow(x[0])
```

For each image, this generator will resize the cat to a random size and will draw it in a random position over a black background image.

The output of the above is:

![image](https://user-images.githubusercontent.com/16224205/220900232-d846db98-595f-4630-b98a-498b0d89c7d4.png)

Now, we can define a similar model, train it and examine its results:

``` python
# transfer learning - load pre-trained vgg and replace its head
vgg = tf.keras.applications.VGG16(input_shape=[128, 128, 3], include_top=False, weights='imagenet')
x = Flatten()(vgg.output)
x = Dense(3, activation='sigmoid')(x)
model2 = Model(vgg.input, x)
model2.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))
# plot the model
plot_model(model2, "second_model.png",show_shapes=True)

# needs steps per epoch since the generator is infinite
model2.fit_generator(cat_gen(),steps_per_epoch=EPOCH_SIZE,epochs=5)

from matplotlib.patches import Rectangle

# given image and a label, plots the image + rectangle
def plot_pred(img,p):
  fig, ax = plt.subplots(1)
  ax.imshow(img)
  rect = Rectangle(xy=(p[0]*128,p[1]*128),width=p[2]*128, height=p[2]*128, linewidth=1,edgecolor='g',facecolor='none')
  ax.add_patch(rect)
  plt.show()


# generate new image
x, _ = next(cat_gen())
# predict
pred = model2.predict(x)
# examine 1 image
im = x[0]
p = pred[0]
plot_pred(im,p)
```

Our trained model is capable of locating a cat in black background images:

![image](https://user-images.githubusercontent.com/16224205/220900371-d4168f91-f56a-4577-929e-dbdcbf6c845f.png)

### 3. Final usecase
In this usecase, we will draw our cat over natural backgrounds. Similarly to the 2nd usecase, we will need to implement a new image generator - each image will consist of a random-sized cat in a random position. The cat will be drawn over a natural image, that will be its background. The natural image will be drawn randomly from a set of images.

``` python
from PIL import Image

cat_pil = Image.open("cat.png")
cat_pil = cat_pil.resize((64,64))
cat = np.asarray(cat_pil)

def natural_cat_gen(batch_size=64):
  # enable generating infinite amount of batches
  while True:
      # generate black images in the wanted size
      X = np.zeros((batch_size, 128, 128, 3))
      Y = np.zeros((batch_size, 3))
      # fill each image
      for i in range(batch_size):
        # resize the cat
        size = np.random.randint(32,64)
        temp_cat = cat_pil.resize((size,size))
        cat = np.asarray(temp_cat) / 255.
        cat_x, cat_y, _ = cat.shape
        # background image
        bg_name = f'bg{np.random.randint(1,4)}.jpg'
        bg = Image.open(bg_name)

        x1 = np.random.randint(1,128 - cat_x)
        y1 = np.random.randint(1,128 - cat_y)
        h = cat_x
        w = cat_y
        # draw the cat over the selected background image
        bg.paste(temp_cat, (x1, y1),mask=temp_cat)
        cat = np.asarray(bg) / 255.
        X[i] = cat

        Y[i,0] = x1/128.
        Y[i,1] = y1/128.
        Y[i,2] = cat_x / 128.
      yield X, Y

# sanity check - plot the images
x,y = next(natural_cat_gen())
plt.imshow(x[0])
```
This generator draws per each image a random natural background image from a given set. It will resize the cat to a random size and draw it over the natural background in a random position.

The natural_cat_gen() output will be similar to:

![image](https://user-images.githubusercontent.com/16224205/220900613-c385b20f-0027-4688-82a6-81edc8690e7f.png)

Next, we will define a new model, train it and visualize some of its results:

``` python
# define a mode
# transfer learning - load pre-trained vgg and replace its head
vgg = tf.keras.applications.VGG16(input_shape=[128, 128, 3], include_top=False, weights='imagenet')
x = Flatten()(vgg.output)
x = Dense(3, activation='sigmoid')(x)
model3 = Model(vgg.input, x)
model3.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))
# plot the model
plot_model(model3, "third_model.png",show_shapes=True)

# train it
# needs steps per epoch since the generator is infinite
model3.fit_generator(natural_cat_gen(),steps_per_epoch=EPOCH_SIZE,epochs=5)

from matplotlib.patches import Rectangle

# given image and a label, plots the image + rectangle
def plot_pred(img,p):
  fig, ax = plt.subplots(1)
  ax.imshow(img)
  rect = Rectangle(xy=(p[0]*128,p[1]*128),width=p[2]*128, height=p[2]*128, linewidth=1,edgecolor='r',facecolor='none')
  ax.add_patch(rect)
  plt.show()


# generate new image
x, _ = next(natural_cat_gen())
# predict
pred = model3.predict(x)
# examine 1 image
im = x[0]
p = pred[0]
plot_pred(im,p)
```

![image](https://user-images.githubusercontent.com/16224205/220900706-bfa95d4e-fdc3-487e-ac32-1bdf57522f9e.png)


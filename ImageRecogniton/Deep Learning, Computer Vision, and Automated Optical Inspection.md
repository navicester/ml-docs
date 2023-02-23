# Deep Learning, Computer Vision, and Automated Optical Inspection

> a case study on classifying surface defects in hot-rolled steel strips

https://towardsdatascience.com/deep-learning-computer-vision-and-automated-optical-inspection-774e8ca529d3
    - https://www.intel.com/content/www/us/en/developer/articles/technical/use-machine-learning-to-detect-defects-on-the-steel-surface.html

**Automated Optical Inspection** is commonly used in electronics industry and manufacturing industry to detect defects in products or components during production. Conceptually, common practices in deep learning for image classification, object detection, and semantic segmentation could be all applied to Automated Optical Inspection. Figure 1 shows some common tasks in image recognition and Figure 2 shows some examples of surface defects in steel parts for cross reference.

![image](https://user-images.githubusercontent.com/16224205/220939294-600711df-613b-4dec-9792-5d5678d1a21e.png)

Figure 1. Different tasks in image recognition. [source]

![image](https://user-images.githubusercontent.com/16224205/220939376-354b8849-2f9a-4ae7-8343-d49fcf6071e8.png)

Figure 2. NEU Surface Defect Dataset: Image Classification (left) and Object Detection (right). [source]


This post describes a study about applying deep learning for image classification on surface defects in hot-rolled steel strips. The rest of this post is organized as follows. 
- First, we will discuss the data and the task. 
- Second, we will address three deep learning models for evaluation: the baseline model, InceptionV3, and MobileNet. 
- Finally, we will present the experiments then evaluate the results.

## NEU Surface Defect Database
[The NEU Surface Defect Database](http://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/list/index.htm) collects six kinds of typical surface defects of the hot-rolled steel strips. This dataset consists of 1,800 200x200 grayscale images, evenly labeled over 6 categories, i.e., rolled-in scale, patches, crazing, pitted surface, inclusion, and scratches. Figure 3 shows sample images in the database.

![image](https://user-images.githubusercontent.com/16224205/220939961-55fb5cc5-d5b9-4d3b-8746-7d78f3a0b20c.png)

Figure 3. Sample images in NEU Surface Defect Database. [source]

Our task here is to explore deep learning models for this multi-class classification problem. Note that, according to the source of the database, this dataset presents two difficult challenges for image classification, i.e., **intra-class variation** and **inter-class similarity**:

- **Intra-Class Variation**, i.e., different patterns might appear within the same class. For example, the category of scratches (the last column in Figure 3) includes horizontal scratch, vertical scratch, slanting scratch, etc. Further, the grayscale of the intra-class defect images is varied due to the influence of the illumination and material changes.
- **Inter-Class Similarity**, i.e., similar aspects might appear under different classes. For example, some images in rolled-in scale, crazing, and pitted surface might look alike.

Although intra-class variation and inter-class similarity have been addressed in different applications, these issues certainly might become a problem in this case.

## Model Selection and Fine-Tuning
Three deep learning models have been implemented by Keras 2.2.2 (with Tensorflow 1.10.0 as the backend) for evaluation:

- **a simple convolutional neural network** as the baseline model: 3 Conv2D/MaxPooling2D pairs as the feature extractor and 3 Dense layers as the classifier; (Figure 4)
- **InceptionV3**: [Keras Application InceptionV3](https://keras.io/applications/#inceptionv3) fine-tuning the classifier by using 1 GlobalAveragePooling2D layer and 2 Dense layers; (Figure 5)
- **MobileNet**: [Keras Application MobileNet](https://keras.io/applications/#mobilenet) fine-tuning the classifier by using 1 GlobalAveragePooling2D layer and 2 Dense layers; (Figure 6)


More engineering decisions on data processing and model training in common:

- no data augmentation;
- training/validation split as 90/10;
- no pre-trained weights;
- 20 epochs without early stopping;

``` python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# parameters for architecture
input_shape = (224, 224, 3)
num_classes = 6
conv_size = 32

# parameters for training
batch_size = 32
num_epochs = 20

# build the model
model = Sequential()

model.add(Conv2D(conv_size, (3, 3), activation='relu', padding='same', input_shape=input_shape)) 
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(conv_size, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
    
model.add(Conv2D(conv_size, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
    
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
 
# train the model                    
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1,
                    validation_split=0.1)
```
Figure 4. simple convolutional neural network.

``` python
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

# parameters for architecture
input_shape = (224, 224, 3)
num_classes = 6
conv_size = 32

# parameters for training
batch_size = 32
num_epochs = 20

# load InceptionV3 from Keras
InceptionV3_model = InceptionV3(include_top=False, input_shape=input_shape)

# add custom Layers
x = InceptionV3_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
Custom_Output = Dense(num_classes, activation='softmax')(x)

# define the input and output of the model
model = Model(inputs = InceptionV3_model.input, outputs = Custom_Output)
        
# compile the model
model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

model.summary()

# train the model 
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1,
                    validation_split=0.1)
```
Figure 5. InceptionV3 Fine-Tuning.
``` python
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

# parameters for architecture
input_shape = (224, 224, 3)
num_classes = 6
conv_size = 32

# parameters for training
batch_size = 32
num_epochs = 20

# load MobileNet from Keras
MobileNet_model = MobileNet(include_top=False, input_shape=input_shape)

# add custom Layers
x = MobileNet_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
Custom_Output = Dense(num_classes, activation='softmax')(x)

# define the input and output of the model
model = Model(inputs = MobileNet_model.input, outputs = Custom_Output)
        
# compile the model
model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

model.summary()

# train the model 
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    verbose=1,
                    validation_split=0.1)
```
Figure 6. MobileNet Fine-Tuning.

## Experiment and Evaluation
- Figure 7 simple convolutional neural network (baseline): validation accuracy is up to 52.78%;
- Figure 8 InceptionV3 Fine-Tuning: validation accuracy is up to 98.89%. This model seems to be overfitting after 14 epochs;
- Figure 9 MobileNet Fine-Tuning: validation accuracy is up to 100.00%. This model reaches 99.44% on epoch 4, then 100.00% on epoch 10 and epoch 14. (too good to be true?! see the full log here)

![image](https://user-images.githubusercontent.com/16224205/220942367-0261f68a-6c63-48b4-86f3-4f5d29e4a639.png)

Figure 7. training/validation history of the simple convolutional neural network.

![image](https://user-images.githubusercontent.com/16224205/220942425-7afec572-6a80-4b76-ad72-81fd3d91f89c.png)

Figure 8. training/validation history of InceptionV3 Fine-Tuning.

![image](https://user-images.githubusercontent.com/16224205/220942482-a4a92158-8891-4776-abc3-8eafc41c1294.png)

Figure 9. training/validation history of MobileNet Fine-Tuning.

## Conclusions
This post has presented a study on classifying surface defects in hot-rolled steel strips. Implementing a defect classification solution for **Automated Optical Inspection** is indeed similar to conducting common practices in deep learning for image classification. It seems to be obvious that people who know how to classify the MNIST dataset and import/tuning a model from Keras Application should be able to do the job. However, I still would like to write down this post for several reasons.

- Although **Automated Optical Inspection** is very important in industry, public datasets for Automated Optical Inspection is relatively rare because probably most of the companies who own the data might not want to release it. As a result, researchers and practitioners might lack of datasets and benchmarks for investigation. **The NEU Surface Defect Dataset** providing data for both image classification and object detection is certainly a good setup for practices.
- Generally speaking, the size of the dataset for Automated Optical Inspection is smaller than textbook datasets (e.g., MNIST or CIFAR) and popular datasets (e.g., ImageNet or COCO.) In the case of the NEU Surface Defect Dataset, there are only 300 images per class for training and validation. Further, correctly applying data augmentation might be a challenge due to the consideration of intra-class variation, inter-class similarity, and the nature of products/components/materials under inspection.
- Some deep learning models are suitable for the tasks of Automated Optical Inspection, but some models might not be good choices. However, there are no golden rules for model selection and fine-tuning. Whether transfer learning with the weights learned from ImageNet (or how much pre-trained weights should be used) could help us to conduct the tasks of Automated Optical Inspection is unknown beforehand.


Final comment: If readers would like to know the difference between deep learning approaches and machine learning approaches for this task, please check [this blog](https://software.intel.com/en-us/articles/use-machine-learning-to-detect-defects-on-the-steel-surface) under Intel AI Academy.



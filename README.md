# Human-Face-Generator

## Introduction

Human Face Generator is a deep learning project that uses Generative Adversarial Networks (GANs) to generate human faces based on celebrity images. The project uses the CelebFaces Attributes Dataset (CelebA) to train the GANs. The project is implemented in Python using the Keras API and Tensorflow Backend.

The end goal for the project is to create realistic human faces that can be used for syntehtic data generation and other applications.

## Exploratory Data Analysis

The dataset used for this project is the CelebFaces Attributes Dataset (CelebA). The dataset contains 202,599 images of human faces with 40 binary attributes. The images are 218x178 pixels in size. The dataset can be downloaded from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

The following are some of the images from the dataset:

[<img src=images/faces.png height=300>](images/faces.png)

The attrbutes for the images also seems to be distributed very evenly. These can be possible incorporated into the model to generate faces with specific attributes in the future.

## Model Architecture

The model architecture is based on the DCGAN architecture proposed by [Goodfellow et al.](https://arxiv.org/abs/1511.06434). The model consists of two parts: the generator and the discriminator. The generator is a deep convolutional neural network that takes a random vector as input and generates a 64x64x3 image as output. The discriminator is a deep convolutional neural network that takes a 64x64x3 image as input and outputs a probability of the image being real or fake. The generator and discriminator are trained in an adversarial manner. The generator is trained to fool the discriminator into thinking that the generated images are real. The discriminator is trained to correctly classify the real and fake images. The generator and discriminator are trained in an alternating manner. The generator is trained first and then the discriminator is trained. This process is repeated until the generator is able to generate images that are indistinguishable from real images.

The following shows the summary of the generator:

```Model: "Generator"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_9 (Dense)             (None, 524288)            52428800  
                                                                 
 batch_normalization_16 (Bat  (None, 524288)           2097152   
 chNormalization)                                                
                                                                 
 leaky_re_lu_26 (LeakyReLU)  (None, 524288)            0         
                                                                 
 reshape_4 (Reshape)         (None, 32, 32, 512)       0         
                                                                 
 conv2d_transpose_16 (Conv2D  (None, 32, 32, 256)      3276800   
 Transpose)                                                      
                                                                 
 batch_normalization_17 (Bat  (None, 32, 32, 256)      1024      
 chNormalization)                                                
                                                                 
 leaky_re_lu_27 (LeakyReLU)  (None, 32, 32, 256)       0         
                                                                 
 conv2d_transpose_17 (Conv2D  (None, 64, 64, 128)      819200    
 Transpose)                                                      
                                                                 
 batch_normalization_18 (Bat  (None, 64, 64, 128)      512       
 chNormalization)                                                
                                                                 
 leaky_re_lu_28 (LeakyReLU)  (None, 64, 64, 128)       0         
                                                                 
 conv2d_transpose_18 (Conv2D  (None, 128, 128, 64)     204800    
 Transpose)                                                      
                                                                 
 batch_normalization_19 (Bat  (None, 128, 128, 64)     256       
 chNormalization)                                                
                                                                 
 leaky_re_lu_29 (LeakyReLU)  (None, 128, 128, 64)      0         
                                                                 
 conv2d_transpose_19 (Conv2D  (None, 128, 128, 3)      4800      
 Transpose)                                                      
                                                                 
=================================================================
Total params: 58,833,344
Trainable params: 57,783,872
Non-trainable params: 1,049,472
_________________________________________________________________
```

The following shows the summary of the discriminator:

```
Model: "Discriminator"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_8 (Conv2D)           (None, 64, 64, 64)        4864      
                                                                 
 leaky_re_lu_24 (LeakyReLU)  (None, 64, 64, 64)        0         
                                                                 
 dropout_8 (Dropout)         (None, 64, 64, 64)        0         
                                                                 
 conv2d_9 (Conv2D)           (None, 32, 32, 128)       204928    
                                                                 
 leaky_re_lu_25 (LeakyReLU)  (None, 32, 32, 128)       0         
                                                                 
 dropout_9 (Dropout)         (None, 32, 32, 128)       0         
                                                                 
 flatten_4 (Flatten)         (None, 131072)            0         
                                                                 
 dense_8 (Dense)             (None, 1)                 131073    
                                                                 
=================================================================
Total params: 340,865
Trainable params: 340,865
Non-trainable params: 0
_________________________________________________________________
```

## Training

The model was first tested on 10,000 images (5% of the dataset) to check the ability of the model to learn. The model was trained for 100 epochs with a batch size of 32. The following shows some images generated by the model:

[<img src=test_samples/image_50.jpg height=200>](test_samples/image_50.jpg)
[<img src=test_samples/image_70.jpg height=200>](test_samples/image_70.jpg)
[<img src=test_samples/image_90.jpg height=200>](test_samples/image_90.jpg)
[<img src=test_samples/image_100.jpg height=200>](test_samples/image_100.jpg)

This showed that the model was able to identify many specific features about the images such as the hair, eyes, nose, etc. This provided enough evidence that the model was able to learn the features of the images and generate new images. One thing to note is that the model still didn't generate images that were indistinguishable from real images. This could be because the model hasn't been trained for a long enough time and requires more time to learn the features of the images.

### Challenges

- `Training time` - The model takes a very long time to train. If the model is trained on the entire dataset, each epoch takes 40-70 minutes to train. This makes it hard to train the model for a long period of time and makes it hard to experiment with different hyperparameters.

- `Memory` - The model requires a lot of memory to train.

- `Discriminator overpowering the generator` - The discriminator is able to overpower the generator and the generator is not able to learn. This is because the discriminator is able to easily identify the fake images generated by the generator. This makes it hard for the generator to learn and generate images that are indistinguishable from real images. This can be clearly seen in the following images:

[<img src=images/overpowering_discriminator.jpg height=200>](images/overpowering_discriminator.jpg)

Here the images generated are very similar and have very little distinguishable features. This is because the discriminator is able to easily identify the fake images generated by the generator and the generator seems to have little incentive to try to generate better and unique images.



### Modifications

- `Image Selection per epoch` - To reduce the training time, we train only on a random subset of the images per epoch. This reduces the training time per epoch and allows us to train the model for a longer period of time and view the overall results. This `RANDOM_SELECT` can be used as a hyperparameter to control the number of images selected per epoch.

- `Batch Size` - The batch size can be used as a hyperparameter to control the number of images selected per epoch. Due to memory constraints, the batch size was set to 64 and can't be further incrased with the current hardware.

- `Discriminator training` - A variety of different methods were tried to successfully train the GAN in a zero sum game setting.

  - `Ratio-based training` - The discriminator was trained once for every 5 times the generator was trained. This was done to ensure that the discriminator doesn't overpower the generator and the generator is able to learn.
  - `Decay ratio training` - The discriminator was initially trained once every time the generator was trained for a few epochs and then this ratio was decreased as training went on. 
  - `Accuracy-based training` - The discriminator was trained once every time the generator was trained if the accuracy of the discriminator was less than a certain percentage.

## Results

## References
[1] The training data is from a dataset mentioned in the paper: S. Yang, P. Luo, C. C. Loy, and X. Tang, "From Facial Parts Responses to Face Detection: A Deep Learning Approach", in IEEE International Conference on Computer Vision (ICCV), 2015.

[2] I. Goodfellow, "NIPS 2016 Tutorial: Generative Adversarial Networks," in Proceedings of the 30th Conference on Neural Information Processing Systems (NIPS), Barcelona, Spain, Dec. 2016. [Online]. Available: https://doi.org/10.48550/arXiv.1701.00160. [Accessed: Mar. 21, 2023].

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gpk2000/Kitty-vs-Doggo/blob/master/Final_Notebook.ipynb)

# Classifying Dogs and Cats using Deep learning

## Motivation
The main source of motivation for me to do this notebook is [this](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438) book. I started learning deep learning about a month back and after reading some of the material and articles online I decided to implement these instead of just learning them. So here it is.

## What is this notebook about?

This notebook is the implementation of a computer vision task for **classifying images** of cats and dogs. Now I only implmented the basic problem in classifying images which is classifying only two categories. Most of the computer vision tasks use Convolutional Networks and I have also done the same. But instead of implementing a model from scratch and training it I used a pretrained model called `VGG16` which is a model trained on ImageNet dataset. Ofcouurse this is an old model and not the current state of the art but I choose this because its simple structure for me to understand.

## Tools and Technologies used
  - **Python**: Ofcourse its a go to language for deep learning and machine learning.
  - **TensorFlow**: This is used in my project as a backend to a Powerful API called Keras.
  - **Keras**: This is a powerful API that made the learning process and training process simple for me.
  - **Google Colab Notebooks**: Since affording GPU is a costly matter (for me too) I use Google Colab everyday for learning, building and training models.
  - **NumPy**: To manipulate images and send them as input to the model.
  - **Matplotlib**: To create graphs of my training progress.

## A brief note about VGG16
  ![](https://i.imgur.com/QDRXeXy.png)
  
VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. VGG16 was trained for weeks and was using NVIDIA Titan Black GPU’s.

> Source: [This article](https://neurohive.io/en/popular-networks/vgg16/)

## How to use this to try/test by yourself
All you need to do is to click on the "Open in Colab" badge at the top of this readme file. All instructions have been posted there.

## Sample screenshot
  ![](https://i.imgur.com/YmmXQrG.png)
 
## Things to note
- The current model isn't _perfect_. There can be times when you can get wrong prediction but that's ok. It's not the fault of the model. It's the fault of myself.
- When you try to give a image which has both dog and cat it will just tell its a dog **or** a cat not both. This is because its not trained to seperate dog and cat from the same image. Instead it is trained to identify whether the image has a dog or a cat.

## TODO for future
- Since the current way of trying things is difficult. I need to move this to a better environment which at the moment seems best is to host the same thing on a website.
- Improve the model, which obviously will take some time. Because I need to learn things, train(most time consuming part) it.

## Got any suggestions
I always welcome any type of suggestion whether it is related to the deep learning part, or whether to improvise the current code.

You can also ping me on Discord: TheSpooN#8963

---

### Thanks for reading this. Hope to see you again reading another readme of mine in near future.

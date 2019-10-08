# How To: Object Detection

## Introduction

This guide assumes the user is running Windows 10.
Also, at least an Nvidia GPU (though maybe there's an even more stringent requirement as to which kind).

### A quick foundation of what we're using:

#### TensorFlow
[TensorFlow](https://en.wikipedia.org/wiki/TensorFlow) is a free and open-source software library that is very useful for many tasks, including machine learning applications such as neural networks. It was originally developed by Google for internal use, but was released for public use on November 2015. It is written in Python, C++, and CUDA.

#### Object Detection API
*Some info about this + newer versions of TF -> requiring older versions of TF and Python*

#### CUDA and cuDNN
[CUDA](https://en.wikipedia.org/wiki/CUDA) is a parallel computing platform and API model created by Nvidia. Basically, this is what will let us use our GPU for general-purpose processing tasks, such as training a machine learning model.

[CuDNN](https://developer.nvidia.com/cudnn), or the CUDA Deep Neural Network library, is a GPU-accelerated library of primitives for deep neural networks. We'll be using this, too.


#### References/Bibliography

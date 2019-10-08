# How To: Object Detection

## Introduction

This guide assumes the user is running Windows 10.
Also, at least an Nvidia GPU (GTX 650 or newer).

### A quick foundation of what we're using:

#### TensorFlow
[TensorFlow](https://en.wikipedia.org/wiki/TensorFlow) is a free and open-source software library that is very useful for many tasks, including machine learning applications such as neural networks. It was originally developed by Google for internal use, but was released for public use on November 2015. It is written in Python, C++, and CUDA.

It is worth noting that you can run TensorFlow on your CPU or GPU, but for this tutorial, we will be running it on our GPU.

#### Object Detection API
*Some info about this + newer versions of TF -> requiring older versions of TF and Python*

#### CUDA and cuDNN
[CUDA](https://en.wikipedia.org/wiki/CUDA) is a parallel computing platform and API model created by Nvidia. Basically, this is what will let us use our GPU for general-purpose processing tasks, such as training a machine learning model.

[CuDNN](https://developer.nvidia.com/cudnn), or the CUDA Deep Neural Network library, is a GPU-accelerated library of primitives for deep neural networks. We'll be using this, too.

### Directions

#### GPU Setup
* [Download CUDA](https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork) Toolkit v9.0 for Windows 10

* Run through the installation
    
    -> [YouTube video](https://www.youtube.com/watch?v=RplXYjxgZbw) explaining the installation process (though, this installation uses different versions)
    
* Make a [Nvidia developer account](https://developer.nvidia.com/rdp/cudnn-download) so you can download cuDNN

* [Download cuDNN](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/9.0_20171129/cudnn-9.0-windows10-x64-v7) v7.0.5 Library for Windows 10

* Unzip this file and move its contents (the folder called `cuda`) to be inside `<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v9.0\`, where `<INSTALL_PATH>` points to the installation directory specified during the installation of the CUDA Toolkit. (By default `<INSTALL_PATH>` = `C:\Program Files`.)
    
#### Virtual Environment Setup
* [Download Anaconda](https://www.anaconda.com/distribution/) for Python 3.7 and Windows 64
* Launch the Anaconda prompt and setup a virtual env (we'll call ours TF-Env for this tutorial) with Python 3.5

    `conda create -n TF-Env python=3.5`
    
* Make sure you activate TF-Env

   `activate TF-Env`
   
* 

#### References/Bibliography

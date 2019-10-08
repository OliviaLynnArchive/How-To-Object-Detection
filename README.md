# How To: Object Detection

## Introduction

This guide assumes the user is running Windows 10.
Also, at least an Nvidia GPU (GTX 650 or newer).
A lot of this borrows heavily from [--'s guide](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html), couldn't have done it without him. I had to change enough things in my own installation process that

### A quick foundation of what we're using:

* [TensorFlow](https://en.wikipedia.org/wiki/TensorFlow): a free and open-source software library by Google that does multiple things, including machine learning. It is worth noting that you can run TensorFlow on your CPU or GPU, but for this tutorial, we will be running it on our GPU.

* TensorFlow Object Detection API: *Some info about this + newer versions of TF -> requiring older versions of TF and Python*

* [CUDA](https://en.wikipedia.org/wiki/CUDA): a parallel computing platform and API model created by Nvidia. Basically, this is what will let us use our GPU for general-purpose processing tasks, such as training a machine learning model.

* [CuDNN](https://developer.nvidia.com/cudnn) (aka the CUDA Deep Neural Network library): a GPU-accelerated library of primitives for deep neural networks. 

### Directions

#### GPU Setup
* [Download CUDA](https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork) Toolkit v9.0 for Windows 10

* Run through the installation
    
    -> [YouTube video](https://www.youtube.com/watch?v=RplXYjxgZbw) explaining the installation process (though, this installation uses different versions)
    
* Make a [Nvidia developer account](https://developer.nvidia.com/rdp/cudnn-download) so you can download cuDNN

* [Download cuDNN](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.0.5/prod/9.0_20171129/cudnn-9.0-windows10-x64-v7) v7.0.5 Library for Windows 10

* Unzip this file and move its contents (the folder called `cuda`) to be inside `<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v9.0\`, where `<INSTALL_PATH>` points to the installation directory specified during the installation of the CUDA Toolkit. (By default `<INSTALL_PATH>` = `C:\Program Files`.)

* Add the following to your path (Click *Start*, search `env` to find "Edit the system environment variables", click `Environment Variables...`, then edit the system variable `Path`):
    `<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin`
    `<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp`
    `<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v9.0\extras\CUPTI\libx64`
    `<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v9.0\cuda\bin`

* Optionally, you can update your GPU drivers now. If you selected "Express Installation" when downloading the CUDA Toolkit, your GPU drivers have probably been overwritten by the ones that come bundled with the toolkit, and are usually not the latest drivers. Visit Nvidia [here](http://www.nvidia.com/Download/index.aspx) and select the apropriate driver for your GPU and OS.

* Congrats! Your GPU should now be set up.

#### Virtual Environment Setup
* [Download Anaconda](https://www.anaconda.com/distribution/) for Python 3.7 and Windows 64

* Launch the Anaconda prompt and setup a virtual env (we'll call ours TF-Env for this tutorial) with Python 3.5

    `conda create -n TF-Env python=3.5`
    
* Make sure you activate TF-Env

   `activate TF-Env`
   
* Install Tensorflow GPU v1.5

    `pip install --ignore-installed --upgrade tensorflow-gpu=1.5`
    
* Test your installation by running python in TF-Env

    ````
    (TF-Env) python`
    >>> import tensorflow as tf
    >>> hello = tf.constant('Hello, TensorFlow')
    >>> sess = tf.Session()
    ```
    
    This should output something that looks similar to:
    
    ```
    2019-02-28 06:56:43.617192: I T:\src\github\tensorflow\tensorflow\core\platform\cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
    2019-02-28 06:56:43.792865: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1356] Found device 0 with properties:
    name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.7335
    pciBusID: 0000:01:00.0
    totalMemory: 8.00GiB freeMemory: 6.61GiB
    2019-02-28 06:56:43.799610: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1435] Adding visible gpu devices: 0
    2019-02-28 06:56:44.338771: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
    2019-02-28 06:56:44.348418: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:929]      0
    2019-02-28 06:56:44.351039: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:942] 0:   N
    2019-02-28 06:56:44.352873: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6387 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1)
    ```
    
    You can complete the test by running:
    
    `>>> print(sess.run(hello))`
    
    `b'Hello, Tensorflow!'`
   
* Good job, your virtual environment is working!

#### Python Packages

* Making sure you still have TF-Env activated, install the following packages:
    ```
    conda install opencv=3.2.0, pillow=5.3.0, pandas=0.22.0
    pip install matplotlib==3.0.0
    ```
    Note the change from conda to pip⁠—normally it's better to avoid mixing conda and pip like this, but later on in this process, I ended up encountering trouble with a conda-installed matplotlib.
    
 * TODO protobuf
 
 * TODO configure your PYTHONPATH (later i guess)
 A PYTHONPATH variable must be created that points to the \models, \models\research, and \models\research\slim directories.
 
 #### Directory Setup for the TensorFlow Object Detection API
    
 * Create a folder in `C:` called `TensorFlow`. This will hold pretty much everything we do from here on out.
 
 * Download the TensorFlow Object Detection repo from GitHub. We're using TensorFlow v1.5, so use [this GitHub commit](https://github.com/tensorflow/models/tree/079d67d9a0b3407e8d074a200780f3835413ef99) to download the corresponding version. *(For reference, the up-to-date version can be found [here](https://github.com/tensorflow/models), but this version will not work for our enviroment.)*
 
 * Unzip the file and extract the the “models-master” folder directly into the `C:\TensorFlow` folder you just created. Rename “models-master” to just “models”.
    
    

#### References/Bibliography

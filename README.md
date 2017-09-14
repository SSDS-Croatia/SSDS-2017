# SSDS-2017 - 2<sup>nd</sup> International Summer School on Data Science

Hands-on sessions for [2nd International Summer School on Data Science](https://sites.google.com/site/ssdatascience2017/) in Split, Croatia.

## Table Of Contents

* [Day 1 - Introduction to Tensorflow](https://github.com/SSDS-Croatia/SSDS-2017/tree/master/Day-1)
* [Day 2 - Convolutional neural networks](https://github.com/SSDS-Croatia/SSDS-2017/tree/master/Day-2)
* [Day 3 - Character-wise language modeling with LSTMs](https://github.com/SSDS-Croatia/SSDS-2017/tree/master/Day-3)
* [Day 4 - Autoregressive Generative Networks](https://github.com/SSDS-Croatia/SSDS-2017/tree/master/Day-4)
* [Day 5 - Generative Adversarial Networks](https://github.com/SSDS-Croatia/SSDS-2017/tree/master/Day-5)


## Windows installation instructions

Windows installation was tested on 64-bit Windows 7 and Windows 10 (there is currently no support for Tensorflow on 32-bit systems). Download Anaconda (Python 3.6) for Windows from [https://www.anaconda.com/download/](https://www.anaconda.com/download/) and install it by running setup in administrator mode. Run Anaconda prompt in administrator mode and create `ssds` environment:

```
conda create --name ssds python=3.6 anaconda
activate ssds
```

Now install Tensorflow into it:

```
pip install --ignore-installed --upgrade tensorflow
```

For more more details on how to install Tensorflow on Windows go to [https://www.tensorflow.org/install/install_windows](https://www.tensorflow.org/install/install_windows). If you already have older version of Anaconda installed where Python 3.6 is not available by default you can still follow the procedure above. It will install Python 3.6 and make it available in the `ssds` environment.


## Clone the repository

First make sure you have git versioning software, you can download and install it from [https://git-scm.com/](https://git-scm.com/). Open command prompt and position yourself to the directory of your choice, then clone the git repository:

```
git clone https://github.com/SSDS-Croatia/SSDS-2017.git
```

As an alternative, you can just download the zip file from [https://github.com/SSDS-Croatia/SSDS-2017/archive/master.zip](https://github.com/SSDS-Croatia/SSDS-2017/archive/master.zip), but then you will have to do it all over again if there are any changes in the meantime!


## Run the IPython notebook

From Anaconda command prompt run:

```
ipython notebook --notebook-dir=PATH-TO-REPOSITORY
```



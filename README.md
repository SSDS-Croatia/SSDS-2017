# SSDS 2017  - 2<sup>nd</sup> Int'l Summer School on Data Science

## Center of Research Excellence for Data Science and Advanced Cooperative Systems, Research Unit for Data Science

Hands-on sessions for [2nd International Summer School on Data Science](https://sites.google.com/site/ssdatascience2017/) organized by the Center of Research Excellence for Data Science and Advanced Cooperative Systems, [Research Unit for Data Science](http://across-datascience.zci.hr/zci/istrazivanje/znanost_o_podatcima), from September 25-29, 2017 in Split, Croatia.

## Table Of Contents

* [Day 1 - Introduction to Tensorflow](https://github.com/SSDS-Croatia/SSDS-2017/tree/master/Day-1)
* [Day 2 - Convolutional Neural Networks for Image Classification](https://github.com/SSDS-Croatia/SSDS-2017/tree/master/Day-2)
* [Day 3 - Character-wise Language Modeling with LSTMs](https://github.com/SSDS-Croatia/SSDS-2017/tree/master/Day-3)
* [Day 4 - Image Segmentation and Object Detection](https://github.com/SSDS-Croatia/SSDS-2017/tree/master/Day-4)
* [Day 5 - Generative Adversarial Networks](https://github.com/SSDS-Croatia/SSDS-2017/tree/master/Day-5)


## Installation instructions

The hands-on sessions are organized as Jupyter notebooks, which you can run on your local computer.
They were tested on Windows, Linux and Mac OS X within [Anaconda](https://www.anaconda.com) (Python 3.6) environment (Version 4.4.0) with [TensorFlow](https://www.tensorflow.org) (Version 1.3). It is adequate and recommended to have installed TensorFlow with CPU-support only, since it is much easier to install.

Please follow below the specific instructions for your OS (Windows, Linux, Mac OS X) to:
* Download & install Anaconda (from [https://www.anaconda.com/download/](https://www.anaconda.com/download/))  
* Create conda environment with Python 3.6 and standard Anaconda Python packages (see [conda-cheatsheet](https://conda.io/docs/_downloads/conda-cheatsheet.pdf))
* Install TensorFlow 1.3 with CPU-support only by using pip (follow [https://www.tensorflow.org/install/](https://www.tensorflow.org/install/))

### Windows
TensorFlow installation was tested on 64-bit Windows 7 and Windows 10 (there is currently no support for TensorFlow on 32-bit systems). We recommend to install TensorFlow in an Anaconda environment. Download Anaconda (Python 3.6) for Windows from [https://www.anaconda.com/download/](https://www.anaconda.com/download/) and install it by running setup in **Administrator mode**.

Afterwards, run Anaconda Command Prompt in **Administrator mode** and
* Create conda environment named `ssds`:
  ```
  conda create --name ssds python=3.6 anaconda
  ```
* Activate `ssds` with:
  ```
  activate ssds
  ```
* Install Tensorflow within `ssds`:
  ```
  pip install --ignore-installed --upgrade tensorflow
  ```

For more more details on how to install TensorFlow on Windows go to [https://www.tensorflow.org/install/install_windows](https://www.tensorflow.org/install/install_windows). If you already have older version of Anaconda installed where Python 3.6 is not available by default you can still follow the procedure above. It will install Python 3.6 and make it available in the `ssds` environment.

### Linux/Mac OS X
Similarly as for Windows, the most convenient way for TensorFlow installation on Linux or Mac OS X is to install TensorFlow with 'CPU-support only within Anaconda environment' following the similar procedure to the above-described steps. For detailed instructions:

* for Linux go to [https://www.tensorflow.org/install/install_linux#installing_with_anaconda](https://www.tensorflow.org/install/install_linux#installing_with_anaconda)

* or for Mac OS X go to [https://www.tensorflow.org/install/install_mac#installing_with_anaconda](https://www.tensorflow.org/install/install_mac#installing_with_anaconda)


## Clone the repository

First make sure you have git versioning software, you can download and install it from [https://git-scm.com/downloads](https://git-scm.com/downloads). Using the command line (i.e. [Anaconda] Command Prompt on Windows or Terminal on Linux/Mac OS X) position to the directory of your choice and then clone the git repository with command:
  ```
  git clone https://github.com/SSDS-Croatia/SSDS-2017.git
  ```

As an alternative, you can just download the zip file from [https://github.com/SSDS-Croatia/SSDS-2017/archive/master.zip](https://github.com/SSDS-Croatia/SSDS-2017/archive/master.zip), but then you will have to do it all over again if there are any changes in the meantime!


## Using hands-on sessions Jupyter notebooks:
All the hands-on sessions are run as Jupyter notebooks. To view and edit notebooks, in the command line (i.e. using Anaconda Command Prompt on Windows or Terminal on Linux/Mac OS X), run the following:

* activate `ssds` conda environment (consult [conda-cheatsheet](https://conda.io/docs/_downloads/conda-cheatsheet.pdf)):

  - Windows: `activate ssds`

  - Linux/Mac OS X: `source activate ssds`


* within `ssds` start Jupyter Notebook server (for [details](https://jupyter.readthedocs.io/en/latest/running.html#running)):
  ```
  jupyter notebook --notebook-dir=PATH-TO-REPOSITORY --port=NUMBER-OF-UNUSED-PORT
  ```
  This will open your default web browser at `http://localhost:[NUMBER-OF-UNUSED-PORT]` and
  you will be able to navigate the repository file structure and use notebooks for the hands-on sessions.

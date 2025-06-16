# EZannot: annotate images and measure the annotated with ease.

[![PyPI - Version](https://img.shields.io/pypi/v/EZannot)](https://pypi.org/project/EZannot/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/EZannot)](https://pypi.org/project/EZannot/)
[![Downloads](https://static.pepy.tech/badge/EZannot)](https://pepy.tech/project/EZannot)

<p>&nbsp;</p>

## What can EZannot do?

**1. Annotates the outline of objects or regions in images with ease (when AI-help enabled).**

   One mouse left click to detect the outline of an object/region, and hit 'enter' to specify its classname:
   
   ![alt text](https://github.com/yujiahu415/EZannot/blob/main/Examples/Left_select.gif?raw=true)

   One mouse left click to include an additional area in an annotation:
   
   ![alt text](https://github.com/yujiahu415/EZannot/blob/main/Examples/Left_include.gif?raw=true)

   Change the classname of an object/region:
   
   ![alt text](https://github.com/yujiahu415/EZannot/blob/main/Examples/Left_change.gif?raw=true)

   One mouse left click to select a region and one right click to exclude an area:
   
   ![alt text](https://github.com/yujiahu415/EZannot/blob/main/Examples/Right_remove.gif?raw=true)

   Press "shift" to enter/exit the editing mode and modify the annotated polygons:

   ![alt text](https://github.com/yujiahu415/EZannot/blob/main/Examples/Shift_modify.gif?raw=true)

<p>&nbsp;</p>

**2. Augments each annotated image to 134 additional manipulated images to improve the model training.**

   ![alt text](https://github.com/yujiahu415/EZannot/blob/main/Examples/Augmentation.png?raw=true)

<p>&nbsp;</p>

**3. Provides quantitative measures, such as area and pixel intensity, for each annotated object/region.**

   ![alt text](https://github.com/yujiahu415/EZannot/blob/main/Examples/Measurements.png?raw=true)

<p>&nbsp;</p>

**4. Completely free, and keeps your data private since the annotation is done locally on your computer.**

<p>&nbsp;</p>

## Installation

<p>&nbsp;</p>

**Important**: If you are a [LabGym](https://github.com/umyelab/LabGym) or [FluoSA](https://github.com/umyelab/FluoSA) or [Cellan](https://github.com/yujiahu415/Cellan) user, you need to install EZannot under a different version of Python3. For example, if your LabGym/FluoSA/Cellan installed under Python3.10, you need to install a Python3.11 or 3.12 and install EZannot under that different version of Python3. This is because LabGym/FluoSA/Cellan and EZannot use different versions of PyTorch. You can use commands like `py -3.10` and `py -3.11` to activate different versions of Python3.

<p>&nbsp;</p>

EZannot works for Windows, Mac and Linux systems. Installation steps can vary for different systems. But in general, you need to:
1) Install Python3 (>=3.10)
2) Set up CUDA (v11.8) for GPU usage
3) Install EZannot with pip
4) Install PyTorch (>=2.5.1)
5) Download [SAM2][] models for AI-help in annotation

Below is the guide for Windows.

1. Install [Git][].

   Select the `64-bit Git for Windows Setup` option. Run the installer, and accept all default values.

2. Install Python>=3.10, for example, [Python 3.12][].

   Scroll down to the bottom and click the `Windows installer (64-bit)` option. Run the installer and select "Add python to path" and "Disable long path limit".

   To test your Python installation, run the following command. If the version number prints out successfully, your Python installation is working.

   ```pwsh-session
   py -3.12 --version
   ```

3. If you're using an NVIDIA GPU, install CUDA Toolkit 11.8 and cuDNN.

   First, install and/or update your GPU drivers at [this link](https://www.nvidia.com/Download/index.aspx). Select your GPU model and click "Search", then click "Download". After installing the drivers, reboot your system to ensure they take effect.

   Then, install [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64). Select your version of Windows, select "exe (local)," then click "Download."

   To verify your installation of CUDA, use the following command.

   ```pwsh-session
   set CUDA_HOME=%CUDA_HOME_V11_8%
   nvcc --version
   ```

   Finally, install [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive). You will need to register an Nvidia Developer account, which you can do for free. You can choose cuDNN v8.9.7 that supports CUDA toolkit v11.8. Choose 'Local Installer for Windows (Zip)', download and extract it. And then copy the three folders 'bin', 'lib', and 'include' into where the CUDA toolkit is installed (typically, 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\'), and replace all the three folders with the same names. After that, you may need to add the 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8' to Path in environmental variables.

4. Upgrade `pip`, `wheel`, `setuptools`.
   
   ```pwsh-session
   py -3.12 -m pip install --upgrade pip wheel setuptools
   ```

5. Add Python path to your environment variables.

   This can be done by select "Add python to path" during step#2. But incase you haven't done this, you may Google how to do this--it's simple. Generally, open System Properties: you can do this by pressing the Windows key + Pause/Break, then clicking 'Advanced system settings'. And then access environment variables: Click on 'environment variables...'. And then locate the PATH variable: In the 'System variables' section, find the variable named 'Path' and click 'Edit...'. Add Python's path: click 'New', and then paste the path to your Python installation directory (typically, C:\Users\YourName\AppData\Local\Programs\Python\Python312) into the 'Variable value' field. Also, add the path to the Scripts folder (typically, C:\Users\YourName\AppData\Local\Programs\Python\Python312\Scripts). You may type `where python` in the command prompt to see the path where Python3.12 is installed.

6. Install EZannot via `pip`.
   
   ```pwsh-session
   py -3.12 -m pip install EZannot
   ```

7. Install PyTorch>=v2.5.1 with CUDA v11.8:

   ```pwsh-session
   py -3.12 -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
   ```
   
   If you are using EZannot without a GPU, use the following command instead.
   
   ```pwsh-session
   py -3.12 -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
   ```
   
8. Setup [SAM2][] models.
 
   Find the 'site-packages' folder where the EZannot is by typing:
   ```pwsh-session
   pip show EZannot
   ```
   And download the [SAM2 models](https://github.com/yujiahu415/EZannot/tree/main/SAM2%20models) ('large' is more accurate but slower than 'tiny'; if you have a good GPU, go for large). You can download the individual files and organize them in the same way as they appear in the folder, and put them (e.g., the entire 'large' folder) inside the '.../site-packages/EZannot/sam2 models/' folder for easy access within EZannot's user interface, or store them somewhere else and choose the option of 'Choose a new directory of the SAM2 model' and navigate to where you store them to access these models.

<p>&nbsp;</p>

## Usage

1. Launch EZannot:

   ```pwsh-session
   EZannot
   ```
   
   The user interface may take a few minutes to start up during the first launch. If the user interface fails to initiate with the above method, you can still make it show up by three lines of code:
   ```pwsh-session
   py -3.12
   from EZannot import __main__
   __main__.main()
   ```

2. Follow the hint for each button in the user interface to annotate images with ease:

   ![alt text](https://github.com/yujiahu415/EZannot/blob/main/Examples/User_interface.png?raw=true)

3. Recommend specifying a new folder for exporting the annotation. You may do this without selecting any augmentation method, which will generate an annotation file for all your original images. And later you can load your original images together with the annotation file to the user interface and perform the augmentation at any time.

4. When loading the annotated images to the user interface, EZannot automatically looks for the annotation file in json format in the same folder where the images are stored, and reads the annotations and displays them in images within the user interface.

[Git]: https://git-scm.com/download/win
[Python 3.12]: https://www.python.org/downloads/release/python-31210/
[SAM2]: https://github.com/facebookresearch/sam2

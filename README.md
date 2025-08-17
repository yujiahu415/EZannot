# EZannot: annotate and measure the annotated with ease.

[![PyPI - Version](https://img.shields.io/pypi/v/EZannot)](https://pypi.org/project/EZannot/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/EZannot)](https://pypi.org/project/EZannot/)
[![Downloads](https://static.pepy.tech/badge/EZannot)](https://pepy.tech/project/EZannot)

<p>&nbsp;</p>

## Key features & Workflow:

**1. AI ([SAM2](https://github.com/facebookresearch/sam2)) help enables manual annotation of object/region outlines with single mouse clicks.**

**2. The augmentation is up to 135 X for each image and enhances the generalizability of the training.**

**3. Manually annotate a few to train an AI Annotator and use it to automatically annotate the rest.**

**4. Refine the annotations performed by the automatic Annotator and iterate a better Annotator.**

**5. All the annotations can be quantified by diverse measurements such as area and pixel intensity.**

**6. Totally free, and keeps your data private since the annotation is done locally on your computer.**

<p>&nbsp;</p>

## How to use?

**1. Annotate the outline of objects or regions in images with ease (when AI-help enabled).**

   A `Mouse left click` to detect the outline and hit `Enter` to specify the classname to finalize the annotation:
   
   ![alt text](https://github.com/yujiahu415/EZannot/blob/main/Examples/Left_select.gif?raw=true)

   Another `Mouse left click` to include an additional area in an annotation before finalizing it: 
   
   ![alt text](https://github.com/yujiahu415/EZannot/blob/main/Examples/Left_include.gif?raw=true)

   A `Mouse left click` to select a region and a `Mouse right click` to exclude some areas in an annotation:
   
   ![alt text](https://github.com/yujiahu415/EZannot/blob/main/Examples/Right_exclude.gif?raw=true)

   To modify the polygon of an annotation, press `Shift` once to enter or exit the editing mode:

   ![alt text](https://github.com/yujiahu415/EZannot/blob/main/Examples/Shift_modify.gif?raw=true)

   To show the classnames of all annotations, press `Space` once to enter or exit the showing mode:

   ![alt text](https://github.com/yujiahu415/EZannot/blob/main/Examples/Space_show.gif?raw=true)

   To delete an annotation, a `Mouse right click` on any area inside the outline of that annotation:

   ![alt text](https://github.com/yujiahu415/EZannot/blob/main/Examples/Right_delete.gif?raw=true)

<p>&nbsp;</p>

**2. Augments each annotated image to 134 additional manipulated images to improve the model training.**

   ![alt text](https://github.com/yujiahu415/EZannot/blob/main/Examples/Augmentation.png?raw=true)

<p>&nbsp;</p>

**3. Provides quantitative measures, such as area and pixel intensity, for each annotated object/region.**

   ![alt text](https://github.com/yujiahu415/EZannot/blob/main/Examples/Measurements.png?raw=true)

<p>&nbsp;</p>

## Installation

<p>&nbsp;</p>

**Important**: If you are a [LabGym](https://github.com/umyelab/LabGym) or [FluoSA](https://github.com/umyelab/FluoSA) user, you need to install EZannot under a different version of Python3 or environment. For example, if your LabGym/FluoSA is installed under Python3.10, you need to install EZannot under another version of Python (e.g., Python3.12). This is because LabGym/FluoSA and EZannot use different versions of PyTorch and different versions of PyTorch cannot be installed within the same Python3 or environment. You can use commands like `py -3.10` and `py -3.11` to activate different versions of Python3.

<p>&nbsp;</p>

EZannot works for Windows, Mac and Linux systems. Installation steps can vary for different systems. But in general, you need to:
1) Install Python3 (>=3.10)
2) If using an NVIDIA GPU, set up CUDA (v11.8) and install PyTorch with cu118 support
3) Install EZannot with pip
4) Download [SAM2](https://github.com/facebookresearch/sam2) models for AI-help in annotation

<p>&nbsp;</p>

### Windows

You need to access the terminal. To do this, open the start menu by clicking the `Win` key, type "PowerShell", and hit enter. All terminal commands going forward should be entered in this terminal.

1. Install Python>=3.10, for example, [Python 3.12](https://www.python.org/downloads/release/python-31210/).

   Scroll down to the bottom and click the `Windows installer (64-bit)` option. Run the installer and select "Add python to path" and "Disable long path limit".

2. If you're using an NVIDIA GPU, install CUDA Toolkit 11.8 and cuDNN, and install PyTorch>=v2.5.1 with cu118 support.

   First, install [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64). Select your version of Windows, select "exe (local)," then click "Download."

   Next, install [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive). You will need to register an NVIDIA Developer account, which you can do for free. You can choose cuDNN v8.9.7 that supports CUDA toolkit v11.8. Choose 'Local Installer for Windows (Zip)', download and extract it. And then copy the three folders 'bin', 'lib', and 'include' into where the CUDA toolkit is installed (typically, 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\'), and replace all the three folders with the same names.

   Finally, install PyTorch>=v2.5.1 with cu118 support:

   ```pwsh-session
   py -3.12 -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
   ```

3. Upgrade `pip`, `wheel`, `setuptools`.
   
   ```pwsh-session
   py -3.12 -m pip install --upgrade pip wheel setuptools
   ```

4. Install EZannot via `pip`.
   
   ```pwsh-session
   py -3.12 -m pip install EZannot
   ```
 
5. Setup SAM2 models.
 
   Find the 'site-packages' folder where the EZannot is by typing:
   ```pwsh-session
   py -3.12 -m pip show EZannot
   ```
   And download the [SAM2 models](https://github.com/yujiahu415/EZannot/tree/main/SAM2%20models) ('large' is more accurate but slower than 'tiny'; if you have a good GPU, go for large). You can download the individual files and organize them in the same way as they appear in the folder, and put them (e.g., the entire 'large' folder) inside the '.../site-packages/EZannot/sam2 models/' folder for easy access within EZannot's user interface, or store them somewhere else and choose the option of 'Choose a new directory of the SAM2 model' and navigate to where you store them to access these models.

<p>&nbsp;</p>

### Mac

You need to access the terminal. Use `Cmd+Space` to enter Spotlight Search, then search for "Terminal" and hit enter to open it. Next, follow these steps.

1. Install Python>=3.10, for example, [Python 3.12](https://www.python.org/downloads/release/python-31210/).

   Scroll down to the bottom and click the `macOS 64-bit universal2 installer` option. Run the installer and select "Add python to path".

2. Upgrade `pip`, `wheel`, `setuptools`.

   ```console
   python3.12 -m pip install --upgrade pip wheel setuptools
   ```

3. Install EZannot via `pip`.
 
   ```console
   python3.12 -m pip install EZannot
   ```

4. Setup SAM2 models.
 
   Find the 'site-packages' folder where the EZannot is by typing:
   ```console
   python3.12 -m pip show EZannot
   ```
   And download the [SAM2 models](https://github.com/yujiahu415/EZannot/tree/main/SAM2%20models) ('large' is more accurate but slower than 'tiny'). You can download the individual files and organize them in the same way as they appear in the folder, and put them (e.g., the entire 'large' folder) inside the '.../site-packages/EZannot/sam2 models/' folder for easy access within EZannot's user interface, or store them somewhere else and choose the option of 'Choose a new directory of the SAM2 model' and navigate to where you store them to access these models.

&nbsp;

## Usage

1. Launch EZannot:

   ```pwsh-session
   EZannot
   ```
   
   The user interface may take a few minutes to start up during the first launch. If the user interface fails to initiate with the above method, which is typcially because the python3 is not added into the 'PATH' environmental variable, you can still make it show up by three lines of code:
   ```pwsh-session
   py -3.12
   ```
   ```pwsh-session
   from EZannot import __main__
   ```
   ```pwsh-session
   __main__.main()
   ```

2. There's a hint for each button in the user interface:

   ![alt text](https://github.com/yujiahu415/EZannot/blob/main/Examples/User_interface.png?raw=true)

3. You can specify a new folder for exporting the annotation without performing any augmentation. In this way, you will have an 'origin' dataset that stores the annotations on all original images. You may perform augmentation on the 'origin' dataset at any time later.

4. You can load the annotated images to the user interface, EZannot automatically looks for the annotation file in json format in the same folder where the images are stored, and reads the annotations and displays them in images within the user interface.

5. You can include more images to an existing annotated dataset. Simply put new images and old ones, as well as the annotation file for the old ones in the same folder, select all the images, specify a new folder to export the annotation, and start to annotate. The previous annotations will be shown in old images.
   
6. If there is no object of interest in an image (a 'null' image), simply don't do any annotation and proceed to the next image and the 'null' image will be marked as 'null' in the annotation file.


# Welcome
This is a program implementing point based mixed-level radar-camera fusion in front view application for autonomous vehicles.
In this work a driver is implemented for communication with Texas instrumets [AWR1843AOP](https://www.ti.com/tool/AWR1843AOPEVM), which was attached to an [MMWAVEICBOOST](https://www.ti.com/tool/MMWAVEICBOOST) board.
A standard webcamera feed with a resolution of 480x640 is fused with the radar point cloud using YOLOv8 and BotSort tracker.

This work is made by Kalle Paasio as part of his Masters thesis for [Tampere University](https://www.tuni.fi/fi) under supervision of [Kovilta Oy](https://kovilta.fi/)

This work contains 2 independent main files with their independent driver implementations.
It also contains files for looking at saved runs.

`main.py` is the main development branch and implements the fusion using sparse point clouds as input from radar. Point clouds are in 3D coordinates + velocity (4D radar)

`get_input.py` was a deep look into how the data comes out of the AWR device and it looks at the range-azimuth map in its inherent polar coordinate form. 
This data is the 0-doppler slice of the radar cube, from which the point cloud is made later on by peak detection using CA-CFAR.

`fitKalmanFilter_radar.py` looks at saved runs of main. With this function we can do data analysis offline without the radar sensor. 
Data from main is saved into ./measurements/ using numpy. The previous runs are in the form of _savefile_name = "../measurements/kalman_results_laptop1.npy"

`fitKalmanFilter_camera.py` also looks at the same saved data, but the camera portion. With this file it was noted that a difference quotient based
approach to camera state is as good if not better than a kalman filter based velocity estimation.

`./measurements` contains powerpoint slides on the fitting process of the Kalman filter.


The AWR1843AOP platform code is the standard mmWave Demo (out of box demo version 3.6.0)
https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/3.6.0/

# Where to start
Sparse data fusion is located in
`main.py`
![Alt text](/pictures/multiple_people.jpg?raw=true "Title")

Dense data is in 
`get_input.py`
![Alt text](/pictures/first_milestone.png?raw=true "Title")

Kalman filter design is in
`fitKalmanFilter_radar.py`
![Alt text](/pictures/mahalalobis_square.png?raw=true "Title")

Run these files as is without parameters (developed to be run in pycharm IDE by pressing play). Hyper parameter definitions are made in code.



# Installation

## install from internet: 
- Microsoft visual studio (IDE,not code) + all the c++ components (They come selectable in the same installation)
- CUDA
- USB drivers for XDS110. This can be done by running the mmWave Demo online or by standalone istallation at https://software-dl.ti.com/ccs/esd/documents/xdsdebugprobes/emu_xds_software_package_download.html

## Setup physical device
- Connect AWR1843AOP to MMWAVEICBOOST
- Set MMWAVEICBOOST to DCA1000 mode using jumpers (see picture below)
- Connect USB cable to XDS110 port (There are two ports. Do not use FTDI port)
- Give 5v power to MMWAVEICBOOST. A phone bettery pack is a portable choice.
![Alt text](/pictures/physical_setup.JPG?raw=true "Title")

## Create virtual environment and download packages
This can be done in two ways:
1. Try to get the latest libraries. Might not work if major changes in libraries have occured
2. Get the exact same library versions as used in the project.

### 1. How I installed the packages: 
This will download the latest libraries and setup the environment.
The libraries might however have conflicts as new versions come about.
```
conda create -n dippa python=3.8
conda activate dippa
pip install numpy
pip install -U lap
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install pyserial
pip install filterpy
```
### 2. Old versions of libraries: 
If the above doesnt work, you can use the previous versions of libraries and the base environment
This is done by simply running requirements.txt with pip. When running this, do not run the code in section 1 above.

```
pip install -r requirements.txt
```


# If your system hardware differs from mine:
The camera `f` needs to be measured (`f` is not directly a focal length, but a scalar relating real world meters to image pixels)
This is easiest done by using the built in switch in main.py called `_show_calibration_dot = True`. 
This compares center dot of a detection to the center dot projected with your `f`. The results should be the same and you should see a dot,
indicating that projection lands in the expected image coordinates. If you see a line, then your `f` is wrong. Try changing it until only a dot appears.

Next you want to measure your camera's Field of View. This can be done with the switch `_show_fov = True` and guessing your FoV in `_fov_azim` and `_fov_elev`
This will draw a rectangle on your image. Adjust the rectangle size with the `_fov_azim` and `_fov_elev` until it fits your image. The values in `_fov_azim` and `_fov_elev`
now represent your FoV. Give this information to the radar by writing in the numbers into the configuration file: example
You found `_fov_azim = 20` degrees,  `_fov_elev = 10` degrees. Change your `aoaFovCfg` line in .cfg file to be:

`aoaFovCfg -1 -20 20 -10 10`

If you have multiple cameras, and the right on is not recognised: try changing 0 to 1 inside main.main() ->
`cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)`

If you do not have an nvidia GPU then go to main.main() and comment out 
`model.to('cuda')`

If your serial ports definitions are not the same as mine go to main.main() and change the COM ports in 
`cli_serial_port, data_serial_port = serial_config(_config_file_name, cli_port='COM3', data_port='COM4')`
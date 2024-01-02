"""
Author: Kalle Paasio
Description: This is a program implementing point-based mixed-level projection type radar-camera fusion
in front view application for autonomous vehicles. It uses AWR1843AOP sensor and a web camera
"""

# Driver based on https://github.com/ibaiGorordo/AWR1843-Read-Data-Python-MMWAVE-SDK-3- (cited 31.5.2023)

import serial
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from dataclasses import dataclass
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_continuous_white_noise
from filterpy.stats import mahalanobis

# Radar sensor configuration file
_config_file_name = 'radar_configurations/scatter_20fps.cfg'
# The bandwidth of USB channel limits the FPS. 20fps is safe. Above that the sensor may crash.

# const Camera "focal length" which is actually a transfer function from real world coordinates to pixels
# f = 493 is from exel, f = 900 from projection. f = 600 is good when looking at radar dots in frame
_focal_length = 600
"""
f is calculated by measurement and correspondence of radar vs image points. eg. point-to-point matching
The points are brought to exel and least squares fit into linear regression to find f. For my excel fit f=493

OR

You can use the show_calibration_dot variable to tune f. In calibration we compare projected and observed center
of object. They SHOULD allign to form a dot. If f is wrong you will see a line. This is the more accurate method.
"""

# Global memory for timekeeping. This is updated every cycle
_frame_time = 0.02  # Used in fusion as the integration time (delta_t). Also used for PFS calculation

"""
You should also limit the FoV observed by radar to same FoV as the camera. This is not necessary, as we will filter
out all points not hitting the camera frame. It will however make the performance of the system better, as 
communication via USB is the main bottleneck of the AWR1843AOP system demo. How to do this:
Go to your configuration file and alter the line

aoaFovCfg -1 -25 25 -20 20

In my case my azimuth takes points from -25 to 25 degrees and elevation from -20 to 20 degrees. Other points are not
calculated or communicated.
"""

"""
You will also want to change the line
guiMonitor -1 1 0 0 0 0 0           in your config file to 
guiMonitor -1 2 0 0 0 0 0           in order to stop sending point side info that we dont need
"""

_save_data_to_file = False
_savefile_name = "../measurements/demo1.npy"
# List for saving data in order to view it in a graph
_save_vector = []

# Decide which plot to show
_show_2d = True
_show_3d = False
_plot_radar_points = True
_use_velocity_smoothing = False

# The focal length parameter can be tuned with this dot. The dot should be dot at all corners of image, not a line
_show_calibration_dot = False

# The fov of camera can be found with this projection
_show_fov = False
_fov_azim = 19 * (3.14159265358 / 180)    # angle * (angle to radian)
_fov_elev = 14 * (3.14159265358 / 180)
# For me the fov of camera is 20 degrees azimuth, 15 degrees elevation
# You want to set the configuration file so that radar fov is a little bigger than this


@dataclass
class TrackedObjectSpecs:
    kf_radar: KalmanFilter
    previous_camera_place: np.array
    previous_true_position: np.array
    previous_true_speed: np.array
    time_window: list


@dataclass
class ProjectedRPoint:
    x_proj: float
    y_proj: float
    radar_x: float
    radar_z: float
    radar_y: float  # dimension away from us, aka "depth"
    vRadial: float


@dataclass
class SaveItem:
    time: float
    deltaT: float
    all_radar_points: list
    radar_kalman_state: list
    camera_detection: np.array


def measure_execution_time(func):
    """
    This is a DECORATOR. To decorate a function with it, add @measure_execution_time before your function: ex.

    @measure_execution_time
    def foo(bar):
        print("hello")

    :param func: Function to be decorated
    :return: func wrapped in timekeeping
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        print(f"Execution of {func.__name__} took {time.time()-start} seconds")
        return ret
    return wrapper


def serial_config(config_file, cli_port='COM3', data_port='COM4', buffer_size=2 ** 16):
    """
    Configure the serial ports and send the data from the configuration file to the radar
    :param data_port:
    :param cli_port:
    :param config_file:
    :param buffer_size: In 5 seconds the system creates 2^14 bytes of data at 20 FPS, AoA limited case.
     2^16 is therefore plenty of space for any configuration when read at 0.3s intervals instead of 5s.
    :return: CLIport, Dataport
    """

    # Open the serial ports for the configuration and the data ports
    command_line_interface_serial_port = serial.Serial(cli_port, 115200)
    data_serial_port = serial.Serial(data_port, 921600)

    # This line is important, but troublesome. Without it the system won't work
    data_serial_port.set_buffer_size(rx_size=buffer_size)
    # Problem with this line is that pyserial library can only give a recommendation for the actual Windows driver
    # implementing serial. This means that some systems may not be able to have a long enough buffer to read
    # whole transmit frames. This would require a different kind of driver structure.
    # This driver requires that a whole frame fit into the buffer

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(config_file)]
    for i in config:
        command_line_interface_serial_port.write((i + '\n').encode())
        print(i)
        time.sleep(0.1)     # more than 0.05 is needed if we wish to send calibration data

    return command_line_interface_serial_port, data_serial_port


def parse_config_file(config_file):
    """
    Function to parse the data inside the configuration file
    :param config_file: String
    :return: dict of sensor aspects
    """
    # Initialize an empty dictionary to store the configuration parameters
    config_parameters = {}
    # The parameter names are copied from embedded c-code
    startFreq = idleTime = numAdcSamples = numAdcSamplesRoundTo2 = chirpStartIdx = \
        chirpEndIdx = numLoops = numFrames = digOutSampleRate = 0
    rampEndTime = freqSlopeConst = framePeriodicity = 0.0

    # Hard code the number of antennas, change if other chip than xwr18xx is used
    numRxAnt = 4
    numTxAnt = 3

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(config_file)]
    for i in config:

        # Split the line
        split_words = i.split(" ")

        # Get the information about the profile configuration
        if "profileCfg" in split_words[0]:
            startFreq = int(float(split_words[2]))
            idleTime = int(split_words[3])
            rampEndTime = float(split_words[5])
            freqSlopeConst = float(split_words[8])
            numAdcSamples = int(split_words[10])
            numAdcSamplesRoundTo2 = 1

            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2

            digOutSampleRate = int(split_words[11])

        # Get the information about the frame configuration
        elif "frameCfg" in split_words[0]:

            chirpStartIdx = int(split_words[1])
            chirpEndIdx = int(split_words[2])
            numLoops = int(split_words[3])
            numFrames = int(split_words[4])
            framePeriodicity = float(split_words[5])

    # Combine the read data to obtain the configuration parameters
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    config_parameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    config_parameters["numRangeBins"] = numAdcSamplesRoundTo2
    config_parameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (
                2 * freqSlopeConst * 1e12 * numAdcSamples)
    config_parameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (
                2 * freqSlopeConst * 1e12 * config_parameters["numRangeBins"])
    config_parameters["dopplerResolutionMps"] = 3e8 / (
                2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * config_parameters["numDopplerBins"] * numTxAnt)
    config_parameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    config_parameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)

    return config_parameters


def read_point_cloud_data_xwr18xx(data_serial_port, verbose=False):
    """
    Funtion to read and parse the incoming data from AWR1843AOP. Some data formats depend on config parameters
    :param verbose: Print cases of misbehavior
    :param data_serial_port: Serial object
    :return: dataOK, frameNumber, detObj
    """

    # Constants
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1
    MMWDEMO_UART_MSG_RANGE_PROFILE = 2
    MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP = 8

    MAGIC_WORD = [2, 1, 4, 3, 6, 5, 8, 7]

    dataOK = 0          # Checks if the data has been read correctly
    frameNumber = 0

    readBuffer = data_serial_port.read(data_serial_port.in_waiting)

    # The byteBuffer is handled by the serial object Dataport
    # The size of the buffer may be exeeded, in which case the buffer resets to 0 size
    # Maximum size of the buffer is defined in serialConfig()
    byteBuffer = np.frombuffer(readBuffer, dtype='uint8')
    byteCount = len(byteBuffer)

    numDetectedObjTotal = 0
    x = []
    y = []
    z = []
    velocity = []

    # Check that the buffer has some data
    if byteCount > 16:

        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == MAGIC_WORD[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc + 8]
            if len(check) == 8:
                if np.all(check == MAGIC_WORD):
                    startIdx.append(loc)

        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

        # Construct the frames
        for i, frameStartIdx in enumerate(startIdx):
            # Initialize the pointer index within frame
            idX = frameStartIdx

# ######################################### header ################################################

            idX += 8    # Magic number is skipped, as it was checked before
            version = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
            idX += 4
            totalPacketLen = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            platform = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
            idX += 4
            frameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            timeCpuCycles = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            numDetectedObj = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            numTLVs = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4
            subFrameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
            idX += 4

            # Check that the packet arrived in its entirety
            if i+1 < len(startIdx):
                # next - this = data we can read for THIS packet
                dataBeforeNextMagic = startIdx[i+1] - startIdx[i]
            else:
                # There is no next starting index. Read until end
                dataBeforeNextMagic = len(byteBuffer) - startIdx[i]

            if dataBeforeNextMagic < totalPacketLen:
                # The packet was cut short. Don't read this singular packet
                if verbose is True:
                    print("Radar packet cut short")
                continue

            #  The end of the packet is padded so that the total packet length is always multiple of 32 Bytes
            if dataBeforeNextMagic > totalPacketLen + 32:
                # Too much data -> There was an overwrite of data on top of this packet
                # This happens around once every 3 seconds and is often attributed to the first packet, i == 0 case
                if verbose is True:
                    print("Radar packet had too much data")
                continue

            # The frame is assumed OK after the above checks

# ##################################### TLV messages behind header ##################################
            numDetectedObjTotal += numDetectedObj

            for tlvIdx in range(numTLVs):

                # Check the header of the TLV message
                tlv_type = np.matmul(byteBuffer[idX:idX + 4], word)
                idX += 4
                tlv_length = np.matmul(byteBuffer[idX:idX + 4], word)
                idX += 4

                if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:
                    for objectNum in range(numDetectedObj):
                        # Read the data for each object
                        x.append(byteBuffer[idX:idX + 4].view(dtype=np.float32)[0])   # x is left, right
                        idX += 4
                        y.append(byteBuffer[idX:idX + 4].view(dtype=np.float32)[0])   # y is depth
                        idX += 4
                        z.append(byteBuffer[idX:idX + 4].view(dtype=np.float32)[0])   # z is top,down
                        idX += 4
                        # v positive = motion away from us. Negative = toward us
                        velocity.append(byteBuffer[idX:idX + 4].view(dtype=np.float32)[0])
                        idX += 4

                        dataOK = 1
                else:
                    if verbose is True:
                        print(f"Radar device sending {tlv_type=} instead of point cloud")

    # Store the data in the detObj dictionary
    detObj = {"numObj": numDetectedObjTotal, "x": np.array(x),
              "y": np.array(y), "z": np.array(z), "velocity": np.array(velocity)}

    return dataOK, frameNumber, detObj


def project3d_point_to_image_plane(x, y, z, shape_image):
    # Assume radar and camera origins align. This gives easy math.
    x_cam = _focal_length * np.divide(x, y)
    y_cam = _focal_length * np.divide(z, y)

    # The aligned coordinates range from negative to positive
    # To get into the pixel space we need to move the center of the image so that there are no "negative indexes"

    """ Alligned center              Image center transferred to camera coordinates
            /\                              /\ 
            |                               |
            |                               |
    --------O------->                       |       0
            |                               |
            |                               |
            |                               |----------------->

    """

    x_cam += shape_image[1]/2
    y_cam += shape_image[0]/2

    return x_cam, y_cam


def find_points_inside_mask(rPointsOnCamera, mask):
    # Picks all points within the mask

    rPointsInsideBoundingBox = []
    for rPoint in rPointsOnCamera:
        x = int(rPoint.x_proj)
        y = int(rPoint.y_proj)
        # Boolean segmentation mask based inclusion of points
        if mask[y][x]:                                     # mask dimensions are flipped
            rPointsInsideBoundingBox.append(rPoint)

    return rPointsInsideBoundingBox


def pick_single_point_using_median(rPointsInsideMask):
    # Take measurement from index, where depth == median depth

    if len(rPointsInsideMask) != 0:
        depths = [x.radar_y for x in rPointsInsideMask]
        index = np.argpartition(depths, len(depths) // 2)[len(depths) // 2]
        # The argpartition is not strictly the median. For even-length inputs the argpartition will choose the
        # higher of the middle values, whereas true median would calculate their average

        return rPointsInsideMask[index]
    else:
        # Not a single point hit this mask
        return None


def pick_single_point_using_average(rPointsInsideMask):
    # When averaging we are taking information from all points instead of just one

    if len(rPointsInsideMask) != 0:
        x = np.average([p.radar_x for p in rPointsInsideMask])
        z = np.average([p.radar_z for p in rPointsInsideMask])
        d = np.average([p.radar_y for p in rPointsInsideMask])
        v = np.average([p.vRadial for p in rPointsInsideMask])

        P = ProjectedRPoint(x_proj=np.nan, y_proj=np.nan,  # projections are not needed after fusion, so no recalculation
                            radar_x=x, radar_z=z, radar_y=d, vRadial=v)

        return P
    else:
        # Not a single point hit this mask
        return None


def pick_single_point_using_mahalanobis_gating(rPointsInsideMask, kalmanFilter, picking_function, gateDistance=4):
    # Version with no saving for discarded data
    insideGate = []
    gatedPointsVector = []

    for point in rPointsInsideMask:
        z = np.array([point.radar_y, project_vfrom_radial_to_tangential(point)])
        if mahalanobis(z, mean=kalmanFilter.x, cov=kalmanFilter.P) < gateDistance:
            insideGate.append(point)
        else:
            gatedPointsVector.append(point)

    return picking_function(rPointsInsideMask), gatedPointsVector


def pick_single_point_using_square_gating(rPointsInsideMask, kalmanFilter, picking_function, gateDistance=4):
    std_x = np.sqrt(kalmanFilter.P[0, 0])
    std_v = np.sqrt(kalmanFilter.P[1, 1])

    insideGate = []
    gatedPointsVector = []

    for point in rPointsInsideMask:
        z = np.array([[project_depth_from_tangential_to_radial(point)],
                      [point.vRadial]])

        # Calculate error between measurement and prior state of Kalman filter. eg. the "residual"
        y = kalmanFilter.residual_of(z)[:, 0]
        y = np.abs(y)
        # if the error is larger than x amount of standard deviations, do not include the measurement
        if y[0] < gateDistance * std_x and y[1] < gateDistance * std_v:
            insideGate.append(point)
        else:
            gatedPointsVector.append(point)

    return picking_function(rPointsInsideMask), gatedPointsVector


def pick_single_point_using_minimum(rPointsInsideMask):
    if len(rPointsInsideMask) != 0:
        depths = [project_depth_from_tangential_to_radial(x) for x in rPointsInsideMask]
        index = np.argmin(depths)

        return rPointsInsideMask[index]
    else:
        # Not a single point hit this mask
        return None


def pick_single_point_using_minimum_in_window(rPointsInsideMask, time_window, win_length=3):
    """

    :param rPointsInsideMask: All points in this frame
    :param time_window: a point from each frame which was chosen as the pick
    :return: point
    """

    all_picked = []
    now = pick_single_point_using_minimum(rPointsInsideMask)
    time_window.append(now)
    if len(time_window) > win_length:
        time_window.pop(0)

    if now is not None:
        all_picked.append(now)

    for pick in time_window:
        if pick is not None:
            all_picked.append(pick)

    return pick_single_point_using_minimum(all_picked), time_window


def project_vfrom_radial_to_tangential(rPoint):
    """
    Note that while the math by the author is intended to take just the component of the speed vector
    coming towards the camera, the radar physics (cosine effect) dictates that the measurement itself
    does not actually contain the true speed, which in turn leads us to perceive the speed scaled by cosine,
    the output of which is between 0 .. 1. Therefore this is a LOWEST estimate of speed

    :param rPoint:
    :return: float speed_tangential
    """
    depth_meas = rPoint.radar_y
    speed_meas = rPoint.vRadial
    radar_x = rPoint.radar_x
    radar_z = rPoint.radar_z

    # Transform velocity from radial to tangential with the centerline
    speed_tangential = speed_meas * (1 / np.sqrt(1 + np.square(radar_x / depth_meas))) \
                     * (1 / np.sqrt(1 + np.square(radar_z / depth_meas)))

    return speed_tangential


def project_depth_from_tangential_to_radial(rPoint):
    return rPoint.radar_y*np.sqrt(1 + (rPoint.radar_x / rPoint.radar_y) ** 2 + (rPoint.radar_z / rPoint.radar_y) ** 2)


def do_trash_keeping(tracked_list, tracked_object_dict):
    to_be_deleted = []
    for key in tracked_object_dict:
        tracked_ids = [botrack.track_id for botrack in tracked_list]
        if key not in tracked_ids:
            to_be_deleted.append(key)

    for key in to_be_deleted:
        del tracked_object_dict[key]


def color_based_on_distance(distance):
    # Rotating the hue around color circle seems efficient to display slowly changing variable such as distance
    hsv = np.uint8([[[distance * 20, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple([int(x) for x in bgr[0][0]])


def update(data_serial_port, cam, model, config_parameters, tracked_object_dict,
           dot_frame, line_frame, ax, stationarity_threshold=40):
    """
    Update the data and display in the plot
    :return: dataOk (radar)
    """
    dataOk = 0

    global _save_vector

    # Take picture with camera
    ret, frame = cam.read()                 # 480,640   vertical, horizontal
    if ret:
        frame = cv2.flip(frame, -1)         # flip horizontal and vertical, as camera is wrong side up

        # Run neural network detection + tracking
        classes = 0
        """
        "classes=0" chooses what we track. The labels are from COCO dataset and you maybe interested in
        0: person
        1: bicycle
        2: car
        3: motorcycle
        """
        yolo_results = model.track(source=frame, tracker="botsort.yaml", classes=classes, persist=True, verbose=False, conf=0.1)[0]
        # Default confidence is 0.1

# ############################################################ RADAR ###########################################
        rPointsOnCamera = []
        # Read and parse received radar data
        dataOk, frameNumber, detObj = read_point_cloud_data_xwr18xx(data_serial_port, verbose=False)

        if dataOk and len(detObj["x"]) > 0:
            # DataOk is asserted only for pointcloud packets. Sideinfo is not ok
            radar_x = detObj["x"]
            radar_y = detObj["y"]
            radar_z = detObj["z"]
            radar_v = detObj["velocity"]
            # z axis is defined in opposite direction in radar. Flip it.
            x_cam, y_cam = project3d_point_to_image_plane(radar_x, radar_y, -radar_z, np.shape(frame))

            # Save only points inside camera view
            for i in range(len(x_cam)):
                if 0 < x_cam[i] < np.shape(frame)[1] and 0 < y_cam[i] < np.shape(frame)[0]:

                    rPoint = ProjectedRPoint(x_proj=x_cam[i], y_proj=y_cam[i], radar_x=radar_x[i],
                                             radar_z=-radar_z[i], radar_y=radar_y[i], vRadial=radar_v[i])
                    # radar z-axis is flipped, just as in projectRadarPointsToImage

                    rPointsOnCamera.append(rPoint)

                    # Plot radar points with color=distance
                    if _show_2d is True:
                        if _plot_radar_points:
                            color = color_based_on_distance(radar_y[i])
                            cv2.circle(frame, (int(x_cam[i]), int(y_cam[i])), 10, color, -1)

# ########################################################### NEURAL NETWORK ###############################

        # Notice: For the Neural Network results we do not use the "results" variable for raw detection data
        # Instead we are interested in the tracker, which uses kalman filtering to enhance the position information

        tracker = model.predictor.trackers[0]
        tracked_list_yolo = tracker.tracked_stracks

        # The segmentations are gathered in combined_masks so that they can be
        # combined with the frame by transparency instead of by occlusion
        combined_masks = np.zeros(np.shape(frame)).astype("uint8")

        for box_index, box in enumerate(yolo_results.boxes):               # one box for each detection
            # box_index is also used for mask index by YOLOv8. The mask implementation in YOLOv8 return no ID
            #   but the ordering of box and mask in the returned result is consistent

            if box.cls.cpu().numpy().astype(int) == classes:      # type is right
                tracker_id = box.id

                if tracker_id is not None:
                    tracker_id = tracker_id.cpu().numpy().astype(int)[0]

                    # Get the tracker STrack(BaseTrack) related to this object
                    strack = 0
                    if box.is_track:
                        # Box does not contain pointer to track. Must go find it from a list
                        for track in tracked_list_yolo:
                            if track.track_id == tracker_id:
                                strack = track
                    else:
                        # Don't do anything for a detection with uninitialized track
                        continue

                    # Kalman filtered state variables for this strack
                    x_camera, y_camera, _, _, _, _, vw_camera, vh_camera = strack.mean

                    # Get segmentation mask from YOLOv8 as a True False-mask
                    mask = yolo_results.masks.data.cpu().numpy()[box_index].astype('bool')

                    if _show_2d is True:
                        # Draw ID on top left of bounding box
                        coord = box.xyxy.cpu().numpy().astype(int)[0]
                        x_start = coord[0]
                        y_start = coord[1]
                        x_end = coord[2]
                        y_end = coord[3]
                        cv2.putText(frame, str(tracker_id), (int(x_start), int(y_start)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

# #################################################### radar - camera fusion ##################################

                    rPointsInsideMask = find_points_inside_mask(rPointsOnCamera, mask)
                    cam_meas = np.array([[x_camera],
                                         [y_camera]])

                    # There is a tracker. We have seen this object before with camera. previous EXISTS
                    if tracker_id in tracked_object_dict:
                        objectInstance = tracked_object_dict[tracker_id]

                        # Speed of center of detected object in camera determined with derivative
                        camera_speed_2d = (cam_meas - objectInstance.previous_camera_place) / _frame_time

                        rPoint, objectInstance.time_window = pick_single_point_using_minimum_in_window(rPointsInsideMask,
                                                                           objectInstance.time_window, win_length=3)

                        objectInstance.kf_radar.predict()

                        # There are valid points belonging to this object
                        if rPoint is not None:
                            depth_radial = project_depth_from_tangential_to_radial(rPoint)
                            meas = np.array([[depth_radial],
                                             [rPoint.vRadial]])

                            objectInstance.kf_radar.update(meas)

                        # Object not seen by radar
                        else:
                            # We do not have real Radar data, but we can synthesize some for the radar
                            # Kalman filter based on camera observations.

                            # If we don't have new information we do NOT want to lead kalman covariance
                            # astray by giving false measurements. If the object is
                            # (1): moving, and we do not see it:
                            # -> Do NOT give a measurement. Kalman is allowed to increase its covariance to find
                            # new state. If the object is
                            # (2): stationary, and we can observe this by camera
                            # -> Tell kalman that we are looking at a stationary object,
                            # as being stationary is something static clutter removed radar does not observe

                            # Case (2): No movement -> Set velocity = 0 as measurement
                            if np.abs(vw_camera) < stationarity_threshold and \
                                    np.abs(vh_camera) < stationarity_threshold:
                                meas = objectInstance.kf_radar.x
                                meas[1] = 0
                                objectInstance.kf_radar.update(meas)

                        if _show_2d is True:
                            # Only color, no underlying image
                            combined_masks[mask] = color_based_on_distance(objectInstance.kf_radar.x[0])

# ####################################### Milestone 4: True speed #######################################
                        # define true position and speed by combining radar and camera
                        # measurements in spherical coordinates. Angle of a ray of light is given by camera.
                        # The ray is cut with the depth measurement from radar. This gives real coordinate

                        # First move to axis center and not image center
                        x_camera = x_camera - 320
                        y_camera = y_camera - 240

                        depth = objectInstance.kf_radar.x[0]
                        cos_elevation = _focal_length / (np.sqrt(_focal_length ** 2 + y_camera ** 2))
                        sin_elevation = y_camera/(np.sqrt(_focal_length ** 2 + y_camera ** 2))
                        cos_azimuth = _focal_length / (np.sqrt(_focal_length ** 2 + x_camera ** 2))
                        sin_azimuth = x_camera/(np.sqrt(_focal_length ** 2 + x_camera ** 2))

                        # Coordinates defined the same as in radar
                        true_x = depth * cos_elevation * sin_azimuth
                        true_z = depth * sin_elevation
                        true_y = np.sqrt(depth ** 2 - true_x ** 2 - true_z ** 2)

                        true_pos = np.array([true_x, true_y, true_z])

                        # Find v by difference function (derivative approach). This discards velocity measurement data.
                        #true_v = (true_pos - objectInstance.previous_true_position)/frame_time

                        range_radar = objectInstance.kf_radar.x[0]
                        speed_radar = objectInstance.kf_radar.x[1]

                        unit_radar_radial_speed_vector = true_pos/np.linalg.norm(true_pos)

                        euclidean_y_from_camera = range_radar * cos_elevation * cos_azimuth

                        camera_speed_vector = np.array([camera_speed_2d[0, 0], 0, camera_speed_2d[1, 0]]) \
                                              * euclidean_y_from_camera / _focal_length

                        tangential_speed_vector = camera_speed_vector - \
                            np.dot(camera_speed_vector, unit_radar_radial_speed_vector) * unit_radar_radial_speed_vector

                        true_vel = unit_radar_radial_speed_vector * speed_radar + tangential_speed_vector

                        if _use_velocity_smoothing:
                            # exponential smoothing. 0 < alpha < 1.
                            # Bigger alpha = faster response. Smaller alpha = more smoothing.
                            alpha = 0.9
                            true_vel = np.array([
                                alpha * true_vel[0] + (1 - alpha) * objectInstance.previous_true_speed[0],
                                alpha * true_vel[1] + (1 - alpha) * objectInstance.previous_true_speed[1],
                                alpha * true_vel[2] + (1 - alpha) * objectInstance.previous_true_speed[2]
                            ])

                        objectInstance.previous_true_position = true_pos
                        objectInstance.previous_true_speed = true_vel
                        objectInstance.previous_camera_place = cam_meas

                        if _show_2d is True:
                            # Plot the true velocity as emanating from human center point
                            # Start = human center. end = go along the vector a little in 3D. Project to 2D
                            end_point = true_pos + true_vel * np.linalg.norm(true_vel)
                            x_end, y_end = project3d_point_to_image_plane(end_point[0], end_point[1], end_point[2],
                                                                          np.shape(frame))
                            if _show_calibration_dot is True:
                                # Use calculated true_pos as end of line instead. This SHOULD be the same as cam_meas
                                # If not, then there is something wrong with projection, namely the focal_length, f
                                x_end, y_end = project3d_point_to_image_plane(true_pos[0], true_pos[1], true_pos[2],
                                                                              np.shape(frame))
                            # if moving toward the sensor -> red. If moving away -> blue
                            if true_vel[1] < 0:
                                color = (0, 0, 255)
                            else:
                                color = (255, 0, 0)

                            cv2.line(frame,
                                     (int(cam_meas[0, 0]), int(cam_meas[1, 0])),
                                     (int(x_end), int(y_end)),
                                     color,
                                     4)

                    # There is NO tracker. We haven't seen this object before
                    else:
                        # First time we have a track. Init filtering

                        rPoint = pick_single_point_using_minimum(rPointsInsideMask)

                        depth_radial = 0.001    # initial guesses for new objects with no radar detections
                        speed = 0.001
                        if rPoint is not None:
                            depth_radial = project_depth_from_tangential_to_radial(rPoint)
                            speed = rPoint.vRadial
# ################################################## kalman filter radar ##################################
                        # Initialize the Kalman filter using frame time as dt
                        # Kalman filter ASSUMES that FPS of the system keeps constant. Otherwise, the matrices
                        # would have to be recalculated with the new dt

                        radarKalman = KalmanFilter(dim_x=2, dim_z=2)
                        # initialize state mean
                        radarKalman.x = np.array([depth_radial, speed])
                        # initialize state variance to maximum expected error, and covariances(off diagonal) to 0
                        radarKalman.P = np.diag([1, 1])
                        # State transition matrix
                        radarKalman.F = np.array([
                            [1, _frame_time],
                            [0, 1]
                        ])
                        # Process noise matrix. Represents white noise in acceleration
                        radarKalman.Q = Q_continuous_white_noise(2, dt=_frame_time, spectral_density=4)
                        # Measurement matrix
                        radarKalman.H = np.array([
                            [1, 0],
                            [0, 1]
                        ])
                        # Measurement noise matrix
                        radarKalman.R = np.diag([0.2 ** 2, 0.4 ** 2])

                        tracked_object_dict[tracker_id] = \
                            TrackedObjectSpecs(kf_radar=radarKalman, previous_camera_place=np.array([[0], [0]]),
                                               previous_true_position=np.array([0.001, 0.001, 0.001]),
                                               previous_true_speed=np.array([0.001, 0.001, 0.001]),
                                               time_window=[])

                    # No matter if we had or did not have seen the object, it is now put to tracked_object_dict
                    if _save_data_to_file is True:
                        # Assumes only single object in vision field of system
                        save_of_iteration = SaveItem(time=0, deltaT=0, all_radar_points=[], radar_kalman_state=[],
                                                     camera_detection=np.array([[0], [0]]))

                        save_of_iteration.all_radar_points = rPointsInsideMask
                        save_of_iteration.camera_detection = cam_meas
                        save_of_iteration.radar_kalman_state = tracked_object_dict[tracker_id].kf_radar.x
                        save_of_iteration.time = time.time()
                        save_of_iteration.deltaT = _frame_time
                        _save_vector.append(save_of_iteration)

        # Delete old objects no longer tracked from our memory
        do_trash_keeping(tracked_list_yolo, tracked_object_dict)

        if _show_fov is True:
            # depth = 1 as choice, as this doesn't matter for angles.

            # top_left
            fov_x = np.cos(_fov_elev) * np.sin(-_fov_azim)
            fov_z = np.sin(_fov_elev)
            fov_y = np.sqrt(1 - fov_x ** 2 - fov_z ** 2)

            x1, y1 = project3d_point_to_image_plane(fov_x, fov_y, fov_z, np.shape(frame))

            # bottom_right
            fov_x = np.cos(-_fov_elev) * np.sin(_fov_azim)
            fov_z = np.sin(-_fov_elev)
            fov_y = np.sqrt(1 - fov_x ** 2 - fov_z ** 2)

            x2, y2 = project3d_point_to_image_plane(fov_x, fov_y, fov_z, np.shape(frame))

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

        if _show_2d is True:
            # Combine mask and image in transparent fashion
            frame = cv2.addWeighted(frame, 0.7, combined_masks, 0.3, 0)

            # Plot FPS
            FPS = np.divide(1.0, _frame_time)
            cv2.putText(frame, f'{FPS:.0f}', (0, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            cv2.imshow('frame', frame)
            cv2.setWindowTitle('frame', f"Press q to quit. Save file during quit = {_save_data_to_file}")

        if _show_3d is True:
            for instance in dot_frame:
                instance.remove()
            dot_frame.clear()

            for instance in line_frame:
                instance.remove()
            line_frame.clear()

            line_length = 5
            space = np.linspace(0, line_length, 5)
            for key in tracked_object_dict:
                x = tracked_object_dict[key].previous_true_position[0]
                y = tracked_object_dict[key].previous_true_position[1]
                z = tracked_object_dict[key].previous_true_position[2]
                dot_frame.append(ax.scatter(x, y, z))

                speed = tracked_object_dict[key].previous_true_speed
                x_speedline = speed[0] * space + x
                y_speedline = speed[1] * space + y
                z_speedline = speed[2] * space + z
                line_frame.extend(ax.plot3D(x_speedline, y_speedline, z_speedline, 'gray'))

    return dataOk


# -------------------------    MAIN   -----------------------------------------
def main():
    global _frame_time, _save_vector

    # Setup camera. 0 is the primary camera of your system.
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Setup YOLO
    model = YOLO("yolov8n-seg.pt")  # load a pretrained segmentation model
    #model.to('cuda')                # Move to nvidia GPU. If you dont have one, comment this line

    # Configure serial ports
    cli_serial_port, data_serial_port = serial_config(_config_file_name, cli_port='COM3', data_port='COM4')

    # Get the configuration parameters from the configuration file
    config_parameters = parse_config_file(_config_file_name)

    tracked_object_dict = {}
    dot_frame = line_frame = []

    ax = 0
    if _show_3d is True:
        plt.ion()
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim([-2, 2])
        ax.set_ylim([0, 4])
        ax.set_zlim([-2, 2])

    # Setup FPS
    # This is done by doing a couple iterations of the loop before actually going to loop.
    # This is done in order to initialize the FPS for the Kalman filter. Only way to know how long the loop takes
    # on certain hardware is to run the loop on said hardware.
    for i in range(2):
        start_time = time.time()
        update(data_serial_port, cam, model, config_parameters, tracked_object_dict, dot_frame, line_frame, ax)
        _frame_time = time.time() - start_time

    tracked_object_dict.clear()
    _save_vector.clear()

    while True:
        start_time = time.time()

        update(data_serial_port, cam, model, config_parameters, tracked_object_dict, dot_frame, line_frame, ax)

        if cv2.waitKey(1) & 0xFF == ord('q'):   # q to quit
            if _save_data_to_file is True:
                np.save(_savefile_name, _save_vector, allow_pickle=True)
            break

        if _show_3d is True:
            plt.pause(0.01)

        _frame_time = time.time() - start_time

    cli_serial_port.write(('sensorStop\n').encode())
    cli_serial_port.close()
    data_serial_port.close()

    # Close camera and video feed window
    cam.release()
    cv2.destroyAllWindows()
    plt.ioff()


if __name__ == '__main__':
    main()

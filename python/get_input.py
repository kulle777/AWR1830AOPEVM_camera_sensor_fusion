import numpy as np
import serial
import time
import matplotlib.pyplot as plt
import cv2
from scipy import interpolate

###############################################################################################
# These globals are set according to config file sent to device in configureDevice LATER ON
# These values are needed for packet interpretation
numRangeBins = 0
digOutSampleRate = 0
freqSlopeConst = 0
slope = 0
rangeStep = 0       # default: 0.04360212053571429
######################################################################################################
# These globals are ones found in the javascript and c source codes
NUM_ANGLE_BINS = 64
# In reality the angle is made of just 12 virtual antennas, but the Fourier is padded to 64.
# This means 64 is not the real resolution.
numVirtualAntennas = 12
rfFreqScaleFactor = 3.6     # 3.6 for startFreq >= 76 and if under then 2.7

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)    # if cv2.CAP_DSHOW is not specified then a windows bug will stop the camera


def commToMaster(cli_port, bytes):
    cli_port.write(bytes)
    while 1:
        resp = cli_port.readline()
        print(resp)
        if resp == b'Done\n' or resp == b'\rDone\n':
            return


def configureDevice(cli_port,config_file):
    global numRangeBins
    global digOutSampleRate
    global freqSlopeConst
    global slope
    global rangeStep

    file = open(config_file, "r")
    for line in file:
        if len(line) < 1:
            continue
        if line[0] == '%':
            continue

        # Find the numRangeBins
        if "profileCfg" in line:
            parts = line.split()
            freqSlopeConst = int(parts[8])
            numRangeBins = int(parts[10])
            digOutSampleRate = int(parts[11])


        byt = bytes(line, 'UTF-8')
        time.sleep(0.0001)              # If you send many commands too fast the hardware on MMWAVEICBOOST is not ready
        # to receive serial even though it says "Done"
        commToMaster(cli_port, byt)

    # From c code on AWR1843AOP:
    slope = float(freqSlopeConst) * ((rfFreqScaleFactor*1e3*900.0)/float(1 << 26)) * 1e12
    # From javascript code of demo visualizer:
    rangeStep = 3e11 * digOutSampleRate / (2 * freqSlopeConst * 1e12 * numRangeBins)
    # Demo says range resolution = 0.044 and this agrees.
    print("Configured device\n")


def processAzimuthHeatMap(complex_list):
    """The processing inside this function is translated by the author from
    javascript code of the mmWave demo visualizer in order to understand the real processing of data"""

    q = np.reshape(complex_list, (numVirtualAntennas, numRangeBins), order='F')    # shape = (12,256)
    # order='F' = Matlab columnwise reshape

    """In many places in dense data a sinc-like pulse was seen. This prompted a question of whether
    windowing could fix the issue. Yes it could, but the results become more blurry"""
    #window = np.blackman(numVirtualAntennas)
    #window2 = np.transpose(np.tile(window,(numRangeBins,1)))
    #q = np.multiply(q,window2)

    Q = np.fft.fft(q, axis=0, n=NUM_ANGLE_BINS)     # shape = (64,256)
    # FFT is padded with zeros to NUM_ANGLE_BINS size. Makes more bins, but no more true information. = Interpolation
    Q_abs = np.absolute(Q)

    Q_shift = np.fft.fftshift(Q_abs, axes=0)        # shape = (64,256)
    Q_trans = np.transpose(Q_shift)                 # shape = (256,64)

    Q_lr = np.fliplr(Q_trans)                       # (256,64)
    # range direction has 256 bins. Azimuth has 64 bins.
    # This is now in polar coordinates distance r first (256) and then angle a (64)
    return Q_lr     # (256,64)


def combine1Dimage(polar, image):   # polar shape (256,64), image shape (480,640,3)

    polar = polar[:, 1:]            # shape (256,63)  Drop the "highest mirror frequency" from FFT
    polar = np.fliplr(polar)        # picture left was previously right for us

    # The first bins are cluttered with RF interference noise. You can throw them out like this
    #polar[0:10, :] = 0  # first 43cm of data was thrown out
    # There also exists calibDcRangeSig calibration on the radar device to do this better.

    oneD = polar.argmax(axis=0)
    intensity_list = polar.max(axis=0)

    thetaScaleFactor = 1.6

    # The azimuth values found in javascript
    azimuth_values = thetaScaleFactor * np.linspace(-NUM_ANGLE_BINS / 2 + 1, NUM_ANGLE_BINS / 2 - 1,
                                                    num=NUM_ANGLE_BINS - 1) * 2 / NUM_ANGLE_BINS
    # The given max value for angle in javascript is (31*2/64) = 0.968 rad = 55.5 degrees max angle

    # azimuths start from minus
    assert -azimuth_values[0] < np.pi/2     # Do not let the values wrap around to other side of image

    polarIdx = 0
    width = image.shape[1]
    xScaleFactor = 300      # 220 to have full range..   tan(-0.96875)=-1.455
    out = np.zeros(width)
    maxRange = rangeStep * numRangeBins

    x_list = []
    value_list = []

    for theta in azimuth_values:
        x_fromZero = np.tan(theta) * xScaleFactor  # tan also gives negative x values indicating LEFT of the center axis

        x_fromMiddle = int(x_fromZero + width / 2)  # Pad the x coordinates right so we dont make an image with negative indices
        depth = oneD[polarIdx] * rangeStep

        value_list.append(depth)
        x_list.append(x_fromMiddle)

        if 0 < x_fromMiddle < width:  # Don't save values that are outside the image vector
            # The depth can be calculated by transforming largest reflection bin number to range, using conversion factor
            out[x_fromMiddle] = depth

            # BIGGER NUMBER = BIGGER DISTANCE

        polarIdx += 1
        # img_artist2.set_data(out)
        # plt.pause(0.01)

    # Takes in 63 and 63
    f = interpolate.interp1d(x_list, value_list)  # interpolate values from index positions to all 640 pixel values
    xnew = np.arange(0, width)
    ynew = f(xnew)

    im2 = image
    # 255 max int
    out2 = ynew/maxRange * 255  # The further the detection the brighter the detection
    im2[230:240, :, 0] = out2.astype(int)       # Red : distance bright = far
    im2[230:240, :, 1] = np.zeros(width)
    im2[230:240, :, 2] = np.zeros(width)

    f = interpolate.interp1d(x_list, intensity_list)  # interpolate values from index positions to all 640 pixel values
    xnew = np.arange(0, width)
    ynew = f(xnew)

    out3 = ynew / intensity_list.max() * 255  # The further the detection the brighter the detection
    im2[220:230, :, 0] = np.zeros(width)        # Green: value of reflection intensity. bright = high
    im2[220:230, :, 1] = out3.astype(int)
    im2[220:230, :, 2] = np.zeros(width)

    return im2


def readRadarDataFrame(data_port):

    data_port.reset_input_buffer()
    data_port.read_until(b'\x02\x01\x04\x03\x06\x05\x08\x07')
    # The serial data comes with flipped byte order for a single variable (eg. you should
    # read the 16bit or 32bit variable as bytes from right to left)
    # The bits themselves are in right order, just that the bytes come in as "last 8bits of a 16bits variable first,
    # then first 8 bits". For 32 bits, value \x04030201 becomes \x01020304

    dat = data_port.read(4)
    version = int.from_bytes(dat, byteorder='little', signed=False)

    dat = data_port.read(4)
    totalPacketLen = int.from_bytes(dat, byteorder='little', signed=False)    #Total packet length including header in Bytes.

    dat = data_port.read(4)
    platform = int.from_bytes(dat, byteorder='little', signed=False)            # platform is supposed to be 0xa1843
    # print(f"{platform=}")                                                      # This in dec is 6 211
    # print(format(platform, '#x'))   # We see that the platform is now good. The int.from_bytes is what fixes order
    # But of course as a result the output is now int, so you have to look at it as hex format to verify

    dat = data_port.read(4)
    frameNumber = int.from_bytes(dat, byteorder='little', signed=False)

    dat = data_port.read(4)
    timeCpuCycles = int.from_bytes(dat, byteorder='little', signed=False)

    dat = data_port.read(4)
    numDetectedObj = int.from_bytes(dat, byteorder='little', signed=False)

    dat = data_port.read(4)
    numTLVs = int.from_bytes(dat, byteorder='little', signed=False)

    dat = data_port.read(4)
    subFrameNumber = int.from_bytes(dat, byteorder='little', signed=False)

    # print header info
    #print(f' {totalPacketLen=},{frameNumber=},{numTLVs=}')

# ############################################# END OF HEADER ##########################################
    AoAFFTList = []

    # Using header info of TLV number we decode every single TLV packet
    while numTLVs > 0:
        dat = data_port.read(4)
        type = int.from_bytes(dat, byteorder='little', signed=False)

        dat = data_port.read(4)
        length = int.from_bytes(dat, byteorder='little', signed=False)

        if type == 1:
            print("MMWDEMO_OUTPUT_MSG_DETECTED_POINTS")
            dat = data_port.read(length)
        elif type == 2:
            print("MMWDEMO_OUTPUT_MSG_RANGE_PROFILE")
            dat = data_port.read(length)
        elif type == 3:
            print("MMWDEMO_OUTPUT_MSG_NOISE_PROFILE")
            dat = data_port.read(length)
        elif type == 4:
            print("MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP")
            dat = data_port.read(length)
        elif type == 5:
            print("MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP")
            dat = data_port.read(length)
        elif type == 6:
            print("MMWDEMO_OUTPUT_MSG_STATS")
            dat = data_port.read(length)
        elif type == 7:
            print("MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO")
            dat = data_port.read(length)
        elif type == 8:
            #print("MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP")    # Line 1331 of mss_main.c

            # range-azimuth static heat map, this is a 2D FFT array in range direction
            # (cmplx16ImRe_t x[numRangeBins][numVirtualAntAzim]), at doppler index 0

            for i in range(length//4):
                dat = data_port.read(2)
                imag = int.from_bytes(dat, byteorder='little', signed=True)

                dat = data_port.read(2)
                real = int.from_bytes(dat, byteorder='little', signed=True)

                comp = complex(real, imag)
                AoAFFTList.append(comp)
        elif type == 9:
            print("MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS")
            dat = data_port.read(length)
        elif type == 10:
            # If this was sent, then you probably have an error.
            print("MMWDEMO_OUTPUT_MSG_MAX")
            dat = data_port.read(length)

        numTLVs -= 1

    # In the end the packet is padded by the radar so that it is always multiple of 32 bits.
    # We do not care. We just flush when we start serial receive.
    return AoAFFTList


def main():
    cli_port = serial.Serial('COM3', baudrate=115200, bytesize=serial.EIGHTBITS,
                             parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)
    data_port = serial.Serial('COM4', baudrate=921600, bytesize=serial.EIGHTBITS,
                              parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)

    configureDevice(cli_port, "radar_configurations/range_azimuth.cfg")
    time.sleep(0.01)

    plt.style.use('_mpl-gallery-nogrid')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.ion()

    # Init cameraplot
    ret, frame = cam.read()
    img_artist2 = ax2.imshow(frame)

    # Init radarplot
    AoAList = readRadarDataFrame(data_port)
    heatmap = processAzimuthHeatMap(AoAList)
    img_artist1 = ax1.imshow(heatmap)

    while True:
        # Radar
        AoAList = readRadarDataFrame(data_port)
        heatmap = processAzimuthHeatMap(AoAList)

        # Camera
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Fusion
        frame = combine1Dimage(heatmap, frame)
        frame = np.flipud(frame)

        img_artist1.set_data(heatmap)

        # Some times image capture fails
        if ret == True:
            img_artist2.set_data(frame)

        # A pause is needed by the library to draw on screen
        plt.pause(0.01)


if __name__ == '__main__':
    main()


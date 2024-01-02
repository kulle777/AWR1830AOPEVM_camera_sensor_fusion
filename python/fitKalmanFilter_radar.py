import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from main import pick_single_point_using_median
from main import project_vfrom_radial_to_tangential
from main import SaveItem
from main import ProjectedRPoint
from main import project_depth_from_tangential_to_radial
from main import pick_single_point_using_minimum
from main import pick_single_point_using_average
from main import pick_single_point_using_mahalanobis_gating
from main import pick_single_point_using_minimum_in_window
from main import pick_single_point_using_square_gating
from filterpy.kalman import KalmanFilter
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_continuous_white_noise
from filterpy.common import Q_discrete_white_noise
from filterpy.stats import mahalanobis
from filterpy.kalman import MerweScaledSigmaPoints


plot_all_raw_meas = True
plot_variances = False
plot_residual = False
plot_gated_points = False
plot_kalman_filter = True
plot_chosen_points = True
use_time = False     # If you want to plot with time instead of index. Doesnt work with gating

# False = Linear Kalman filter. Linear turns out to be better fit in most cases
use_unscented_filter = False

std_scale = 1   # Plot this many STD:s away

# data = "../measurements/demo1.npy"
data = "../measurements/cars_laptop8.npy"
#"../measurements/cars_laptop13.npy"
_save_vector = np.load(data, allow_pickle=True)
plt.ioff()

# ###################################### init ##############################################
chosen_init_sample = None
i = 0
while chosen_init_sample is None:
    chosen_init_sample = pick_single_point_using_minimum(_save_vector[i].all_radar_points)

    i += 1

depth_radial = project_depth_from_tangential_to_radial(chosen_init_sample)
init_state = np.array([[chosen_init_sample.radar_y], [chosen_init_sample.vRadial]])

_frame_time = _save_vector[0].deltaT               # About 0.5 for laptop

# ###################################### Kalman ############################################

# Linear Kalman filter approach
kf = KalmanFilter(dim_x=2, dim_z=2)
# initialize state mean with first measurement
kf.x = np.array(init_state)
# initialize state variance to maximum expected error, and covariances(off diagonal) to 0
kf.P = np.diag([3**2, 3**2])
# State transition matrix
kf.F = np.array([
    [1, _frame_time],
    [0, 1]
])
# Measurement matrix
kf.H = np.array([
    [1, 0],
    [0, 1]
])
# Measurement noise matrix
kf.R = np.diag([0.2 ** 2, 2 ** 2])
# Process noise matrix. Represents Continuous White Noise Model for kinematic system
kf.Q = Q_continuous_white_noise(2, dt=_frame_time, spectral_density=0.5)
# Other representation is a Piecewise White Noise Model for kinematic system
# kf.Q = Q_discrete_white_noise(2, dt=frame_time, var=2)

if use_unscented_filter is True:
    # Replace the kf with UKF
    def f_cv(x, dt):
        F = np.array([[1, dt],
                      [0, 1]])
        return F @ x

    def h_cv(x):
        return x

    sigmas = MerweScaledSigmaPoints(2, alpha=.1, beta=2., kappa=1.)
    kf = UKF(dim_x=2, dim_z=2, fx=f_cv, hx=h_cv, dt=_frame_time, points=sigmas)
    # UKF and linear use different definition of state. Linear has shape=(2,1). UKF has shape=(2,)
    init_state = np.array([chosen_init_sample.radar_y, chosen_init_sample.vRadial])
    kf.x = np.array(init_state)
    kf.R = np.diag([0.2 ** 2, 2 ** 2])
    kf.Q = Q_continuous_white_noise(2, dt=_frame_time, spectral_density=0.5)

# ####################################### main #####################################

gatedPointsVector = []
gatedPointsIndexes = []
window = []

xs, covs, zs, ys, ts = [], [], [], [], []
for number, Item in enumerate(_save_vector):
    rPointsInsideMask = Item.all_radar_points

    """
    As part of the fusion we implemented many ways of choosing a single point from multiple
    points hitting the same object. Here you can select from all of the choices by using the variable "selection"
    """

    selection = 5

    if selection == 0:
        chosen_point = pick_single_point_using_median(rPointsInsideMask)

    elif selection == 1:
        chosen_point = pick_single_point_using_average(rPointsInsideMask)

    elif selection == 2:
        chosen_point = pick_single_point_using_minimum(rPointsInsideMask)

    elif selection == 3:
        chosen_point, gatedPoints = pick_single_point_using_square_gating(rPointsInsideMask, kf,
                                                                          pick_single_point_using_minimum,
                                                                          gateDistance=3)
        gatedPointsVector.extend(gatedPoints)
        for _ in range(len(gatedPoints)):
            gatedPointsIndexes.append(number)

    elif selection == 4:
        chosen_point, gatedPoints = pick_single_point_using_mahalanobis_gating(rPointsInsideMask, kf,
                                                                               pick_single_point_using_minimum,
                                                                               gateDistance=3)
        gatedPointsVector.extend(gatedPoints)
        for _ in range(len(gatedPoints)):
            gatedPointsIndexes.append(number)

    elif selection == 5:
        chosen_point, window = pick_single_point_using_minimum_in_window(rPointsInsideMask, window,
                                                                         win_length=3)

    else:
        raise Exception("The point selection method you tried to use is not implemented")

    # Predict is done even if no data arrives
    kf.predict()

    # Gating may have reduced number of points to 0
    if chosen_point is not None:
        z = np.array([project_depth_from_tangential_to_radial(chosen_point), chosen_point.vRadial])
        kf.update(z)
    else:
        z = np.array([np.nan, np.nan])

    ys.append(kf.y)
    zs.append(z)
    xs.append(kf.x)
    covs.append(kf.P)
    ts.append(Item.time)

# ############################################ plot ####################################################

# Size is already known, easier to use numpy as is has indexing
xs = np.asarray(xs)
covs = np.asarray(covs)
zs = np.asarray(zs)
ys = np.asarray(ys)
ts = np.asarray(ts)
ts = ts - ts[0]             # Make time start from o

if use_unscented_filter:
    length = len(xs[:, 0])
    xs = np.reshape(xs, (length, 2, 1))

# Plot standard deviations to see lines inside which results should fit

std_position = std_scale * np.sqrt(covs[:, 0, 0])
std_velocity = std_scale * np.sqrt(covs[:, 1, 1])

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8))

# ########################################### plot, place ########################################
#ax1.title.set_text(f'Depth,   {std_scale=},   {_frame_time=},    {data=}')
ax1.title.set_text(f'Depth, m')

if plot_all_raw_meas:
    # place
    index_vec = []
    time_vec = []
    raw_meas_vec = []
    basetime = _save_vector[0].time

    for i, meas in enumerate(_save_vector):
        all = meas.all_radar_points
        for k in range(len(all)):
            index_vec.append(i)
            time_vec.append(meas.time-basetime)
            depth_radial = project_depth_from_tangential_to_radial(all[k])
            raw_meas_vec.append(depth_radial)

    if use_time:
        ax1.plot(time_vec, raw_meas_vec, '.m')
    else:
        ax1.plot(index_vec, raw_meas_vec, '.m')

    # speed
    raw_meas_vec = []

    for i, meas in enumerate(_save_vector):
        all = meas.all_radar_points
        for k in range(len(all)):
            raw_meas_vec.append(all[k].vRadial)
    if use_time:
        ax2.plot(time_vec, raw_meas_vec, '.m')
    else:
        ax2.plot(index_vec, raw_meas_vec, '.m')

if plot_chosen_points:
    if use_time:
        ax1.plot(ts, zs[:, 0], 'o')
        ax2.plot(ts, zs[:, 1], 'o')
    else:
        ax1.plot(zs[:, 0], 'o')
        ax2.plot(zs[:, 1], 'o')

if plot_gated_points:
    depths = [project_depth_from_tangential_to_radial(p) for p in gatedPointsVector]
    velocities = [p.vRadial for p in gatedPointsVector]

    ax1.plot(gatedPointsIndexes, depths, ".r")
    ax2.plot(gatedPointsIndexes, velocities, ".r")

if plot_kalman_filter:
    std_top = xs[:, 0, 0] + std_position
    std_btm = xs[:, 0, 0] - std_position
    if use_time:
        ax1.plot(ts, std_top, linestyle=':', color='k', lw=1, alpha=0.4)
        ax1.plot(ts, std_btm, linestyle=':', color='k', lw=1, alpha=0.4)
        ax1.fill_between(ts, std_top, std_btm,
                         facecolor='yellow', alpha=0.2, interpolate=True)
    else:
        ax1.plot(std_top, linestyle=':', color='k', lw=1, alpha=0.4)
        ax1.plot(std_btm, linestyle=':', color='k', lw=1, alpha=0.4)
        ax1.fill_between(range(len(std_top)), std_top, std_btm,
                         facecolor='yellow', alpha=0.2, interpolate=True)
    if use_time:
        ax1.plot(ts, xs[:, 0, 0])
    else:
        ax1.plot(xs[:, 0, 0])



# ####################################### plot, speed ########################################
ax2.title.set_text('Speed, m/s')
if plot_kalman_filter:
    std_top = xs[:, 1, 0] + std_position
    std_btm = xs[:, 1, 0] - std_position
    if use_time:
        ax2.plot(ts, std_top, linestyle=':', color='k', lw=1, alpha=0.4)
        ax2.plot(ts, std_btm, linestyle=':', color='k', lw=1, alpha=0.4)
        ax2.fill_between(ts, std_top, std_btm,
                         facecolor='yellow', alpha=0.2, interpolate=True)
    else:
        ax2.plot(std_top, linestyle=':', color='k', lw=1, alpha=0.4)
        ax2.plot(std_btm, linestyle=':', color='k', lw=1, alpha=0.4)
        ax2.fill_between(range(len(std_top)), std_top, std_btm,
                         facecolor='yellow', alpha=0.2, interpolate=True)
    if use_time:
        ax2.plot(ts, xs[:, 1, 0])
    else:
        ax2.plot(xs[:, 1, 0])

if use_time:
    #ax1.set_xlabel("time (s)")
    ax2.set_xlabel("time (s)")

# ####################### variance plot #################################
if plot_variances:
    f2, (ax11, ax22, ax33) = plt.subplots(3, 1, figsize=(18, 8))

    ax11.title.set_text('Variance pos')
    ax11.plot(covs[:, 0, 0])
    ax22.title.set_text('Variance vel')
    ax22.plot(covs[:, 1, 1])

# ############################## Residual plot #####################
if plot_residual:
    f3, (ax111, ax222) = plt.subplots(2, 1, figsize=(18, 8))

    ax111.title.set_text(f'Residual pos {std_scale=}')
    ax111.plot(ys[:, 0,0])
    ax222.title.set_text('Residual vel')
    ax222.plot(ys[:, 1,0])

    ax111.plot(std_position, linestyle=':', color='k', lw=1, alpha=0.4)
    ax111.plot(-std_position, linestyle=':', color='k', lw=1, alpha=0.4)
    ax111.fill_between(range(len(std_position)), std_position, -std_position,
                       facecolor='yellow', alpha=0.2, interpolate=True)

    ax222.plot(std_velocity, linestyle=':', color='k', lw=1, alpha=0.4)
    ax222.plot(-std_velocity, linestyle=':', color='k', lw=1, alpha=0.4)
    ax222.fill_between(range(len(std_velocity)), std_velocity, -std_velocity,
                       facecolor='yellow', alpha=0.2, interpolate=True)


plt.show()

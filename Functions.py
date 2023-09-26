import  numpy as np
import cv2 as cv

# Define a function that calculates the moving average of a curve
def moving_average(curve, radius):
	window_size = 2 * radius + 1
	kernel = np.ones(window_size) / window_size
	curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
	curve_smoothed = np.convolve(curve_pad, kernel, mode='same')
	curve_smoothed = curve_smoothed[radius:-radius]
	return curve_smoothed

# Define a function that will apply the moving average on each dimension of the trajectory
def smooth_trajectory(trajectory):
	smoothed_trajectory = np.copy(trajectory)
	
	for i in range(3):
		smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius = 50)
	return smoothed_trajectory	
	
# Define function that eliminates border artifacts
def clean_border(frame):
	frame_shape = frame.shape
	
	mat = cv.getRotationMatrix2D(
		(frame_shape[1] / 2, frame_shape[0] / 2),
		0,
		1.04
		)
	frame = cv.warpAffine(frame, mat, (frame_shape[1], frame_shape[0]))
	return frame
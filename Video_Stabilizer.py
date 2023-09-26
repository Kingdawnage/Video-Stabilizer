# Import dependencies
import cv2 as cv
import numpy as np
from Functions import smooth_trajectory
from Functions import clean_border

# Read input video
capture = cv.VideoCapture('Video/field.mp4')

# Get the frame count of the video
frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

# Get the height, width and fps of the video stream
f_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
f_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
f_fps = int(capture.get(cv.CAP_PROP_FPS))

# Specify the codec to use for the video output
fourcc = cv.VideoWriter_fourcc(*'mp4v')

# Set the video output 
out_vid = cv.VideoWriter('Video/video.mp4', fourcc, f_fps, (2 * f_width, f_height))

# Read the first frame
_, prev_frame = capture.read()

# Convert the frame to greyscale
prev_frame_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY) 
cv.imshow('Grey', prev_frame_gray)

##################################################################################
# Define array that stores the transformations of the frame
transforms = np.zeros((frame_count - 1, 3), np.float32)

for i in range(frame_count - 2):
    # Calculate the optical flow between consecutive frames
    prev_points = cv.goodFeaturesToTrack(
        prev_frame_gray,
        maxCorners=400,
        qualityLevel=0.01,
        minDistance=30,
        blockSize=3
        )
    
    # Read the next frame
    istrue, current_frame = capture.read()
    if not istrue:
        break
    
    # Convert to greyscale
    current_frame_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
    
    # Track the feature points / Calculate the optical flow
    current_points, status, error = cv.calcOpticalFlowPyrLK(
        prev_frame_gray,
        current_frame_gray,
        prev_points,
        None
        )
    
    # Identity check
    assert prev_points.shape == current_points.shape
    
    # Filter the valid points
    index = np.where(status==1)[0]
    prev_points = prev_points[index]
    current_points = current_points[index]
    
    # Calculate the transformations between the points
    matrix, _ = cv.estimateAffine2D(prev_points, current_points)
    translation_x = matrix[0,2]
    translation_y = matrix[1,2]
    rotation_angle = np.arctan2(matrix[1, 0], matrix[0, 0])
    transforms[i] = [translation_x, translation_y, rotation_angle]
    prev_frame_gray = current_frame_gray
    
    print(f'Frame: {i} / {frame_count} - Tracked points: {len(prev_points)}')
    
# Calculate the trajectory using the cumulative sum oof the transformations
trajectory = np.cumsum(transforms, axis = 0)

# Smooth the trajectory using the moving average
smoothed_trajectory = smooth_trajectory(trajectory)

# Calculate the difference between the smoothed and the original trajectory
diff = smoothed_trajectory - trajectory

# Calculate the new transformation array
smooth_transforms = transforms + diff

# Reset video stream to the first frame
capture.set(cv.CAP_PROP_POS_FRAMES, 0)

# Process each frame and stabilize the video
for i in range(frame_count -2):
    istrue, frame = capture.read()
    
    if not istrue:
        break
    
    translation_x = smooth_transforms[i, 0]
    translation_y = smooth_transforms[i, 1]
    rotation_angle = smooth_transforms[i, 2]
    
    # Create the transformation matrix for stabilization
    transformation_matrix = np.zeros((2, 3), np.float32)
    transformation_matrix[0, 0] = np.cos(rotation_angle)
    transformation_matrix[0, 1] = -np.sin(rotation_angle)
    transformation_matrix[1, 0] = np.sin(rotation_angle)
    transformation_matrix[1, 1] = np.cos(rotation_angle)
    transformation_matrix[0, 2] = translation_x
    transformation_matrix[1, 2] = translation_y 
    
    # Apply the transformations to the frame
    stabilized_frame = cv.warpAffine(
        frame,
        transformation_matrix,
        (f_width, f_height)
        )
    print(f"Frame {i}: TranslationX = {translation_x}, TranslationY = {translation_y}, RotationAngle = {rotation_angle}")
    # Fix the border of the stabilized frame
    stabilized_frame = clean_border(stabilized_frame)
    
    # Add the original frame to the side of the stabilized frame
    out_frame = cv.hconcat([frame, stabilized_frame])
    
    # Resize the frame if necessary
    # if out_frame.shape[1] > 1920:
    #     # new_width = 1920
    #     # aspect_ratio = out_frame.shape[1] / out_frame.shape[0]
    #     # new_height = int(new_width / aspect_ratio)
    #     out_frame = cv.resize(out_frame, (out_frame.shape[1] // 2, out_frame.shape[0] // 2))
        
    # Display both frames
    cv.imshow("Before and After", out_frame)
    cv.waitKey(70)
    
    # Write the frame to the output video file
    out_vid.write(out_frame)
    
capture.release()
out_vid.release()
cv.destroyAllWindows()
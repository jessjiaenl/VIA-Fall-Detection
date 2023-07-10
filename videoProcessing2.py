import cv2
import numpy as np


def calculate_frame_difference(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Read the first frame
    success, prev_frame = video.read()

    # Check if the video opened successfully
    if not success:
        print("Video not opened successfully :( " + x)
        return

    # Convert the frame to grayscale
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Create an empty list to store the frame differences
    frame_diff_list = []
    first = True
    diff = None
    prev_diff = None
    # Iterate through the video frames
    while True:
        # Read the next frame
        success, curr_frame = video.read()

        # Break the loop if there are no more frames
        if not success:
            break

        # Convert the frame to grayscale
        curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        # Calculate the absolute difference between frames
        frame_diff = cv2.absdiff(prev_frame_gray, curr_frame_gray)
        # Cast as a signed 16 bit int
        frame_diff = frame_diff.astype(np.int16)

        if first is True:
            prev_diff = frame_diff
            first = False
        else:
            diff = frame_diff
            diff_of_diff = diff - prev_diff
            abs_diff = np.abs(diff_of_diff)
            # Take the absolute value
            total_sum = np.sum(abs_diff)
            frame_avg_diff = total_sum / (224*224)
            # Append the frame difference to the list
            frame_diff_list.append(frame_avg_diff)
            prev_diff = diff

        # Update the previous frame and differences
        prev_frame_gray = curr_frame_gray
            
    # Release the video
    video.release()

    # Convert the frame differences list to a NumPy array
    return np.array(frame_diff_list)


# Path to your video files
falling_paths = [
                 "./datasets/model1_data_new/Fall/resizedk_fall_lobby_cut.mp4",
                 "./datasets/model1_data_new/Fall/resizedk_fall_street_cut.mp4",
                 "./datasets/model1_data_new/Fall/resizedk_fall_streetwithtrees_cut.mp4",
                 "./datasets/model1_data_new/Fall/resized_IMG0480_falling.mp4",
                 "./datasets/model1_data_new/Fall/resized_IMG0484_falling.mp4",
                 "./datasets/model1_data_new/Fall/resized_IMG0485_falling.mp4",
                 "./datasets/model1_data_new/Fall/resized_IMG1614_falling.mp4",
                 "./datasets/model1_data_new/Fall/resized_IMG1615_falling.mp4",
                 "./datasets/model1_data_new/Fall/resized_IMG1616_falling.mp4",
                 "./datasets/model1_data_new/Fall/resized_s-fall3_cut.mp4",
                 "./datasets/model1_data_new/Fall/resized_vid1_falling.mp4",
                 "./datasets/model1_data_new/Fall/resized_vid2_falling.mp4",
                 "./datasets/model1_data_new/Fall/resized_vid4_falling.mp4",
                 "./datasets/model1_data_new/Fall/s-resized_s-fall2.mp4",
                 "./datasets/model1_data_new/Fall/resizeds-fall3.mp4",
                 "./datasets/model1_data_new/Fall/resizeds-fall4.mp4",
                 "./datasets/model1_data_new/Fall/resizeds-fall5.mp4"]

default_paths = [
                 "./datasets/model1_data_new/Still/resizedk-still1.mp4",
                 "./datasets/model1_data_new/Still/resizedk-still2.mp4",
                 "./datasets/model1_data_new/Still/resized_s-still1.mp4",
                 "./datasets/model1_data_new/Still/resizeds-still2.mp4",
                 "./datasets/model1_data_new/Still/resizeds-still3.mp4",
                 "./datasets/model1_data_new/Still/resizeds-still4.mp4"]

video_paths = falling_paths + default_paths
concatenated_diff, concatenated_diff2 = None, None
concatenated_norm = []
arr_of_diffs, diff_fall, diff_default = [], [], []
normalized_array, norm_fall, norm_default = [], [], []

# Call the function to calculate frame differences and save them into an array
# Get concatenated array
for x in video_paths:
    frame_diff_array = calculate_frame_difference(x)
    if concatenated_diff is None:
        concatenated_diff = frame_diff_array
    else:
        concatenated_diff = np.concatenate((concatenated_diff, frame_diff_array))
    arr_of_diffs.append(frame_diff_array)

# for falling vids
for x in falling_paths:
    frame_diff_array = calculate_frame_difference(x)
    diff_fall.append(frame_diff_array)

# for falling vids
for x in default_paths:
    frame_diff_array = calculate_frame_difference(x)
    diff_default.append(frame_diff_array)

# Normalize the array for falling
for x in diff_fall:
    norm = (x - np.min(concatenated_diff)) / (np.max(concatenated_diff) - np.min(concatenated_diff))
    norm_fall.append(norm)

# Normalize the array for default
for x in diff_default:
    norm = (x - np.min(concatenated_diff)) / (np.max(concatenated_diff) - np.min(concatenated_diff))
    norm_default.append(norm)

# Normalize the array of arrays
for x in arr_of_diffs:
    norm = (x - np.min(concatenated_diff)) / (np.max(concatenated_diff) - np.min(concatenated_diff))
    normalized_array.append(norm)

# Print normalized differences of differences
for elem in norm_default:
    # Iterate through the array with a sliding window
    for i in range(len(elem) - 16+1):
        window =elem[i: i+16]
        for x in window:
            print(x, end=" ")
        print("")

#normalize the concatenated array
# concatenated_norm = (concatenated_diff - np.min(concatenated_diff)) / (np.max(concatenated_diff) - np.min(concatenated_diff))


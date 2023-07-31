import cv2
import numpy as np

def normalize_frame_differences(frame_diff_array):
    # Calculate the minimum and maximum values in the frame differences array
    min_val = np.min(frame_diff_array)
    max_val = np.max(frame_diff_array)
    # Normalize the frame differences to the range [0, 1]
    normalized_frame_diff_array = (frame_diff_array - min_val) / (max_val - min_val)
    return normalized_frame_diff_array

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
        # print(frame_diff, frame_diff.shape)
        total_sum = np.sum(frame_diff)
        frame_avg_diff = total_sum / (224*224)
        #print(frame_avg_diff)
        
        # Append the frame difference to the list
        frame_diff_list.append(frame_avg_diff)

        # Update the previous frame
        prev_frame_gray = curr_frame_gray

    # Release the video
    video.release()

    # Convert the frame differences list to a NumPy array
    return np.array(frame_diff_list)


# Path to your video files
# Add your video paths here
falling_paths = [
                 "./datasets/vids/splitted/new_moving/resized_logitech-fall1.mp4",
                 "./datasets/vids/splitted/new_moving/resized_logitech-fall2.mp4",
                 "./datasets/vids/splitted/new_moving/resized_logitech-fall3.mp4",
                 "./datasets/vids/splitted/new_moving/resized_logitech-fall4.mp4",
                 "./datasets/vids/splitted/new_moving/resized_logitech-fall5.mp4",
                 "./datasets/vids/splitted/new_moving/resized_logitech-fall6.mp4",
                 "./datasets/vids/splitted/new_moving/resized_logitech-fall7.mp4",
                 "./datasets/vids/splitted/new_moving/resized_logitech-fall8.mp4",
                 "./datasets/vids/splitted/new_moving/resized_logitech-fall9.mp4",
                 "./datasets/vids/splitted/new_moving/resized_logitech-fall10.mp4",
                 "./datasets/vids/splitted/new_moving/resized_logitech-fall11.mp4",
                 "./datasets/vids/splitted/new_moving/resized_logitech-fall12.mp4",
                 "./datasets/vids/splitted/new_moving/resized_logitech-fall13.mp4",
                 "./datasets/vids/splitted/new_moving/resized_logitech-fall14.mp4",
                 "./datasets/vids/splitted/new_moving/resized_logitech-fall15.mp4"
                 ]

default_paths = [
                 "./datasets/vids/splitted/new_still/resized_logitech-default1.mp4",
                 "./datasets/vids/splitted/new_still/resized_logitech-default2.mp4",
                 "./datasets/vids/splitted/new_still/resized_logitech-default3.mp4",
                 "./datasets/vids/splitted/new_still/resized_logitech-default4.mp4",
                 "./datasets/vids/splitted/new_still/resized_logitech-default5.mp4",
                 "./datasets/vids/splitted/new_still/resized_logitech-default6.mp4",
                 "./datasets/vids/splitted/new_still/resized_logitech-default7.mp4",
                 "./datasets/vids/splitted/new_still/resized_logitech-default8.mp4",
                 "./datasets/vids/splitted/new_still/resized_logitech-default9.mp4",
                 "./datasets/vids/splitted/new_still/resized_logitech-default10.mp4",
                 "./datasets/vids/splitted/new_still/resized_logitech-default11.mp4",
                 "./datasets/vids/splitted/new_still/resized_logitech-default12.mp4",
                 "./datasets/vids/splitted/new_still/resized_logitech-default13.mp4",
                 "./datasets/vids/splitted/new_still/resized_logitech-default14.mp4",
                 "./datasets/vids/splitted/new_still/resized_logitech-default15.mp4"
                 ]


video_paths = falling_paths + default_paths
concatenated_diff = None
arr_of_diffs = []
normalized_array = []

# Call the function to calculate frame differences and save them into an array
# Get concatenated array
for x in video_paths:
    frame_diff_array = calculate_frame_difference(x)
    if concatenated_diff is None:
        concatenated_diff = frame_diff_array
    else:
        concatenated_diff = np.concatenate((concatenated_diff, frame_diff_array))

# for falling vids
for x in falling_paths:
    frame_diff_array = calculate_frame_difference(x)
    arr_of_diffs.append(frame_diff_array)

# Uncomment below
# for x in default_paths:
#     frame_diff_array = calculate_frame_difference(x)
#     arr_of_diffs.append(frame_diff_array)

# Normalize the array
for x in arr_of_diffs:
    norm = (x - np.min(concatenated_diff)) / (np.max(concatenated_diff) - np.min(concatenated_diff))
    normalized_array.append(norm)

# normalized_array = (concatenated_diff - np.min(concatenated_diff)) / (np.max(concatenated_diff) - np.min(concatenated_diff))
# for elem in normalized_array:
#     print(elem)
#     print("\n")

for elem in normalized_array:
    # Iterate through the array with a sliding window
    for i in range(len(elem) - 8+1):
        window =elem[i: i+8]
        for x in window:
            print(x, end=" ")
        print("")
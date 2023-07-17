import cv2

def get_frame_dimensions(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Read the first frame
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error reading frame.")
        return

    # Get the dimensions of the frame
    height, width, channels = frame.shape

    # Print the dimensions
    print("Frame dimensions: {}x{}".format(width, height))

    # Release the video capture object and close any windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "./datasets/vids/splitted/still/c_still_gopro_2.mp4"  # Replace this with your video file's path
    get_frame_dimensions(video_path)
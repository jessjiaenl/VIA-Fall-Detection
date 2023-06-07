import cv2
import numpy as np
from screeninfo import get_monitors

# Get the screen resolution
screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height

# Create a named window with fullscreen flag
cv2.namedWindow("My Window", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("My Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Initialize tab sizes and positions
tab_width = (screen_width-100-120) // 6
tab_height = 50
tab_positions = [(50+i * tab_width+i*20, 20) for i in range(6)]
tab_active = [False] * 6

def draw_tabs():
    # Draw tabs on the top part of the screen
    for i, position in enumerate(tab_positions):
        color = (0, 255, 0) if tab_active[i] else (0, 0, 255)
        #cv2.rectangle(image, position, (position[0] + tab_width, position[1] + tab_height), color, -1)
        rounded_rectangle(image, (position[0],20), (position[0]+tab_width,position[1]+tab_height),(0,0,0), 5, False, (236,234,225))

def enlarge_tab(event, x, y, flags, param):
    # Check if a tab is clicked and enlarge it
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, position in enumerate(tab_positions):
            if position[0] <= x <= position[0] + tab_width and position[1] <= y <= position[1] + tab_height:
                rounded_rectangle(image, position, (position[0] + tab_width, position[1] + tab_height), (0, 0, 0), 5, True, (164,158,132))  # Active tab fill color
                tab_text(textarr)
            else:
                rounded_rectangle(image, position, (position[0] + tab_width, position[1] + tab_height), (0, 0, 0), 5, True,(236,234,225))
                tab_text(textarr)
    
def rounded_rectangle(src, topLeft, bottomRight, lineColor, cornerRadius, fill, fillcolor):
    p1 = topLeft
    p2 = (bottomRight[0], topLeft[1])
    p3 = bottomRight
    p4 = (topLeft[0], bottomRight[1])

    if fill == True:
        cv2.rectangle(src, (p1[0]+cornerRadius, p1[1]), (p3[0]-cornerRadius, p3[1]), fillcolor, -1)
        cv2.rectangle(src, (p1[0], p1[1]+cornerRadius), (p3[0], p3[1]-cornerRadius), fillcolor, -1)
        cv2.ellipse(src, (p1[0] + cornerRadius, p1[1] + cornerRadius), (cornerRadius, cornerRadius), 180.0, 0, 90, fillcolor, -1)
        cv2.ellipse(src, (p2[0] - cornerRadius, p2[1] + cornerRadius), (cornerRadius, cornerRadius), 270.0, 0, 90, fillcolor, -1)
        cv2.ellipse(src, (p3[0] - cornerRadius, p3[1] - cornerRadius), (cornerRadius, cornerRadius), 0.0, 0, 90, fillcolor, -1)
        cv2.ellipse(src, (p4[0] + cornerRadius, p4[1] - cornerRadius), (cornerRadius, cornerRadius), 90.0, 0, 90, fillcolor, -1)
    
    cv2.rectangle(src, (p1[0]+cornerRadius, p1[1]), (p3[0]-cornerRadius, p3[1]), fillcolor, -1)
    cv2.rectangle(src, (p1[0], p1[1]+cornerRadius), (p3[0], p3[1]-cornerRadius), fillcolor, -1)
    
    cv2.ellipse(src, (p1[0] + cornerRadius, p1[1] + cornerRadius), (cornerRadius, cornerRadius), 180.0, 0, 90, fillcolor, -1)
    cv2.ellipse(src, (p2[0] - cornerRadius, p2[1] + cornerRadius), (cornerRadius, cornerRadius), 270.0, 0, 90, fillcolor, -1)
    cv2.ellipse(src, (p3[0] - cornerRadius, p3[1] - cornerRadius), (cornerRadius, cornerRadius), 0.0, 0, 90, fillcolor, -1)
    cv2.ellipse(src, (p4[0] + cornerRadius, p4[1] - cornerRadius), (cornerRadius, cornerRadius), 90.0, 0, 90, fillcolor, -1)
    '''
    # Draw straight lines
    cv2.line(src, (p1[0] + cornerRadius, p1[1]), (p2[0] - cornerRadius, p2[1]), lineColor, 2)
    cv2.line(src, (p2[0], p2[1] + cornerRadius), (p3[0], p3[1] - cornerRadius), lineColor, 2)
    cv2.line(src, (p4[0] + cornerRadius, p4[1]), (p3[0] - cornerRadius, p3[1]), lineColor, 2)
    cv2.line(src, (p1[0], p1[1] + cornerRadius), (p4[0], p4[1] - cornerRadius), lineColor, 2)
    '''
def tab_text(text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    color = (0, 0, 0) 
    thickness = 2
    for i, position in enumerate(tab_positions):
        cv2.putText(image, text[i], (position[0]+10, + position[1]+tab_height-15), font, scale, color, thickness)

# Create a black image of the screen resolution
image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

image[:screen_height//2, :] = (255, 255, 255)
image[screen_height//2:, :] = (211,222,224)
#211,222,224 beige 
#164,158,132 dark blue
#236,234,225 light blue
process_time = "0.01 ms"
# Draw the initial tabs
draw_tabs()
textarr = ["Falling Detection","Pose Detection","other","other2","other3","other4"]
tab_text(textarr)
offset = 7
Descriptionpos = [(2*((screen_width-100-20) // 3),20+tab_height+100),(screen_width-100,screen_height-100)]
rounded_rectangle(image, (Descriptionpos[0][0]+offset,Descriptionpos[0][1]), (Descriptionpos[1][0]+offset,Descriptionpos[1][1]+offset), (0,0,0),5,True,(128,128,128))
rounded_rectangle(image, Descriptionpos[0], Descriptionpos[1], (0,0,0),5,True,(236,234,225))

cv2.putText(image,"Model Description",(Descriptionpos[0][0]+20,20+tab_height+100+50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
cv2.putText(image,"By the power of computer vision, a person may transmit SOS signals to the rescue task forces during a fall before he or she hits the ground or crushes any communication equipments.",(Descriptionpos[0][0]+20,20+tab_height+100+50+50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),1)
cv2.putText(image,"Performance: "+process_time,(Descriptionpos[0][0]+20,20+tab_height+100+Descriptionpos[1][1]//2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
# Set the mouse event callback function   
cv2.setMouseCallback("My Window", enlarge_tab)
# Open a video capture object
cap = cv2.VideoCapture('trimmed.mp4')  

# Read the first frame of the video
ret, frame = cap.read()

# Define the coordinates for the video box
video_box_x = 80
video_box_y = 20+tab_height+50
video_box_width = (Descriptionpos[0][0]-20)-150
video_box_height = screen_height-180
cv2.rectangle(image, (video_box_x,video_box_y),(video_box_x+video_box_width,video_box_y+video_box_height),(0,0,0),2)
cnt = 0
fade_in_speed = 0.03
alpha = 0
alpha2 = 0

# Create a transparent image for the text
text_image = np.zeros_like(image, dtype=np.uint8)

while True:
    # Restart the video if it reaches the end
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    cnt+=1
     # Resize the frame to fit within the box dimensions
    resized_frame = cv2.resize(frame, (video_box_width, video_box_height))
    if cnt >= 50:
        # Add a slightly dark overlay on the video box region
        overlay = np.full(resized_frame.shape, (0, 0, 0), dtype=np.uint8)
        alpha += fade_in_speed
        if alpha >= 0.6:
            alpha = 0.6
        overlay = cv2.addWeighted(resized_frame, 1 - alpha, overlay, alpha, 0)

        # Put the overlay on the image
        image[video_box_y:video_box_y + video_box_height, video_box_x:video_box_x + video_box_width] = overlay
        #cv2.putText(image, "Falling",(video_box_x+video_box_width//2-120,video_box_y+video_box_height//2),cv2.FONT_HERSHEY_DUPLEX,3,(255,255,255),6)
        if alpha2 >= 1:
            alpha = 1
        # Update the alpha channel of the text color
        text_color = (255, 255, 255, int(alpha2 * 255))

        # Clear the text image
        text_image.fill(0)

        # Put the text on the image with the updated color
        cv2.putText(text_image, "Falling", (video_box_x+video_box_width//2-120,video_box_y+video_box_height//2),cv2.FONT_HERSHEY_DUPLEX,3,text_color,6)
        # Blend the text image with the main image using the alpha channel
        image = cv2.addWeighted(image, 1, text_image, alpha2, 0)
        alpha2 += fade_in_speed
        # Increment the alpha value
        alpha += fade_in_speed
    else:
        # Put the resized frame in the video box region of the image
        image[video_box_y:video_box_y + video_box_height, video_box_x:video_box_x + video_box_width] = resized_frame

     # Read the next frame
    ret, frame = cap.read()
    # Display the image in the window
    cv2.imshow("My Window", image)
    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF
    # Exit the loop if 'q' key is pressed
    if key == ord('q'):
        break

cap.release()
# Destroy the window
cv2.destroyAllWindows()
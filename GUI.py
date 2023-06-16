import cv2
import numpy as np
from screeninfo import get_monitors
import time

class GUI:
    switched = False
    #screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height
    screen_width, screen_height = 1920, 1080
    # Initialize tab sizes and positions
    tab_width = None
    tab_height = 50
    tab_positions = None
    tab_active = [False] * 3
    currtab = 0
    vidlist = ["_IMG_2610.mp4","vid1.mp4","trimmed.mp4","vid4.mp4","vid1.mp4","vid4.mp4"]
    vidinit = False
    # Create a black image of the screen resolution
    image = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    process_time = "0.01 ms"
    textarr = ["Falling Detection","Pose Detection","Object Detection"]
    Descriptionpos = [(2*((screen_width-100-20) // 3),20+tab_height+100),(screen_width-100,screen_height-100)]
    video_box_x = 80
    video_box_y = 20+tab_height+50
    video_box_width = (Descriptionpos[0][0]-20)-150-(1030-900)#1030->900
    video_box_height = screen_height-180#900
    cnt = 0
    fade_in_speed = 0.03
    alpha = 0
    alpha2 = 0
    # cap = cv2.VideoCapture(vidlist[currtab])
    # ret, frame = None, None
    mousex, mousey = 0, 0
    LINE_POINTS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]
    COLOR_HEAD = (180, 0, 0, 255)
    COLOR_BODY = (180, 255, 255, 0)
    COLOR_FOOT = (180, 0, 255, 0)
    COLOR_LINE = (180, 0x88, 0x88, 0x88)
    COLOR_HEAD = COLOR_HEAD[::-1]
    COLOR_BODY = COLOR_BODY[::-1]
    COLOR_FOOT = COLOR_FOOT[::-1]
    COLOR_LINE = COLOR_LINE[::-1]
    POINT_COLORS = [
    COLOR_HEAD,  # 0. nose
    COLOR_BODY,  # 1. neck
    COLOR_BODY,  # 2. R shoulder
    COLOR_BODY,  # 3. R elbow
    COLOR_BODY,  # 4. R wrist
    COLOR_BODY,  # 5. L shoulder
    COLOR_BODY,  # 6. R elbow
    COLOR_BODY,  # 7. R wrist
    COLOR_FOOT,  # 8. R hip
    COLOR_FOOT,  # 9. R knee
    COLOR_FOOT,  # 10. R ankle
    COLOR_FOOT,  # 11. L hip
    COLOR_FOOT,  # 12. L knee
    COLOR_FOOT,  # 13. L ankle
    COLOR_HEAD,  # 14. R eye
    COLOR_HEAD,  # 15. L eye
    COLOR_HEAD,  # 16. R ear
    COLOR_HEAD  # 17. L ear
    #18. background
    ]
    faceIndex = ["0", "14", "15", "16", "17"]
    OUTPUT_RATIO = 0.008926701731979847
    OUTPUT_BIAS = 126
    threshold = 0.2/OUTPUT_RATIO+OUTPUT_BIAS

    # Create a transparent image for the text
    text_image = np.zeros_like(image, dtype=np.uint8)
    framefell = False

    def __init__(self):
        self.tab_width = (self.screen_width-100-120) // 3
        self.tab_positions = [(50+i * (self.tab_width)+i*20, 20) for i in range(3)]
        # Create a named window with fullscreen flag
        cv2.namedWindow("My Window", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("My Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.image[:self.screen_height//2, :] = (255, 255, 255)
        self.image[self.screen_height//2:, :] = (211,222,224)
        #211,222,224 beige 
        #164,158,132 dark blue
        #236,234,225 light blue
        
        # Draw the initial tabs
        self.draw_tabs()
        self.tab_text(self.textarr)
        offset = 7
        self.rounded_rectangle(self.image, (self.Descriptionpos[0][0]+offset,self.Descriptionpos[0][1]), (self.Descriptionpos[1][0]+offset,self.Descriptionpos[1][1]+offset), (0,0,0),5,True,(128,128,128))
        self.rounded_rectangle(self.image, self.Descriptionpos[0], self.Descriptionpos[1], (0,0,0),5,True,(236,234,225))

        cv2.putText(self.image,"Model Description",(self.Descriptionpos[0][0]+20,20+self.tab_height+100+50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        cv2.putText(self.image,"By the power of computer vision, a person may transmit SOS signals to the rescue task forces during a fall before he or she hits the ground or crushes any communication equipments.",(self.Descriptionpos[0][0]+20,20+self.tab_height+100+50+50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),1)
        cv2.putText(self.image,"Performance: "+self.process_time,(self.Descriptionpos[0][0]+20,20+self.tab_height+100+self.Descriptionpos[1][1]//2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        # Set the mouse event callback function   
        cv2.setMouseCallback("My Window", self.mouse_callback)
        # Define the coordinates for the video box
        
        cv2.rectangle(self.image, (self.video_box_x,self.video_box_y),(self.video_box_x+self.video_box_width,self.video_box_y+self.video_box_height),(0,0,0),2)
        
        #cv2.imshow("My Window", self.image)
        #cv2.waitKey(2500)
        
    
    def rounded_rectangle(self, src, topLeft, bottomRight, lineColor, cornerRadius, fill, fillcolor):
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

    def draw_tabs(self):
        # Draw tabs on the top part of the screen
        for i, position in enumerate(self.tab_positions):
            color = (0, 255, 0) if self.tab_active[i] else (0, 0, 255)
            #cv2.rectangle(image, position, (position[0] + tab_width, position[1] + tab_height), color, -1)
            self.rounded_rectangle(self.image, (position[0], 20), (position[0]+self.tab_width, position[1]+self.tab_height), (0,0,0), 5, False, (236,234,225))

    
    def tab_text(self, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        color = (0, 0, 0) 
        thickness = 2
        for i, position in enumerate(self.tab_positions):
            cv2.putText(self.image, text[i], (position[0]+10, + position[1]+self.tab_height-15), font, scale, color, thickness)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mousex, self.mousey = x, y
            for i, position in enumerate(self.tab_positions):
                if position[0] <= self.mousex <= position[0] + self.tab_width and position[1] <= self.mousey <= position[1] + self.tab_height:
                    self.rounded_rectangle(self.image, position, (position[0] + self.tab_width, position[1] + self.tab_height), (0, 0, 0), 5, True, (164,158,132))  # Active tab fill color
                    self.tab_text(self.textarr)
                    if i != self.currtab:
                        self.currtab = i
                        cv2.rectangle(self.image, (self.video_box_x,self.video_box_y),(self.video_box_x+self.video_box_width,self.video_box_y+self.video_box_height),(0,0,0),2)
                        self.cnt, self.alpha, self.alpha2 = 0, 0, 0
                        self.switched = True
                        #return i
                else:
                    self.rounded_rectangle(self.image, position, (position[0] + self.tab_width, position[1] + self.tab_height), (0, 0, 0), 5, True,(236,234,225))
                    self.tab_text(self.textarr)
            cv2.imshow("My Window", self.image)
        return -1

    def tab_migrated(self):
        for i, position in enumerate(self.tab_positions):
            if position[0] <= self.mousex <= position[0] + self.tab_width and position[1] <= self.mousey <= position[1] + self.tab_height:
                self.rounded_rectangle(self.image, position, (position[0] + self.tab_width, position[1] + self.tab_height), (0, 0, 0), 5, True, (164,158,132))  # Active tab fill color
                self.tab_text(self.textarr)
                if i != self.currtab:
                    self.currtab = i
                    cv2.rectangle(self.image, (self.video_box_x,self.video_box_y),(self.video_box_x+self.video_box_width,self.video_box_y+self.video_box_height),(0,0,0),2)
                    self.cnt, self.alpha, self.alpha2 = 0, 0, 0
                    return i
            else:
                self.rounded_rectangle(self.image, position, (position[0] + self.tab_width, position[1] + self.tab_height), (0, 0, 0), 5, True,(236,234,225))
                self.tab_text(self.textarr)
        return -1
    
    def draw_frame(self, frame, result):
        if self.currtab == 0:
            #self.cnt+=1
            # Resize the frame to fit within the box dimensions
            resized_frame = cv2.resize(frame, (self.video_box_width, self.video_box_height))

            '''
            if result == True:
                self.framefell = True
            if self.framefell == True:
                # Add a slightly dark overlay on the video box region
                overlay = np.full(resized_frame.shape, (0, 0, 0), dtype=np.uint8)
                self.alpha += self.fade_in_speed
                if self.alpha >= 0.6:
                    self.alpha = 0.6
                overlay = cv2.addWeighted(resized_frame, 1 - self.alpha, overlay, self.alpha, 0)

                # Put the overlay on the image
                self.image[self.video_box_y:self.video_box_y + self.video_box_height, self.video_box_x:self.video_box_x + self.video_box_width] = overlay
                #cv2.putText(image, "Falling",(video_box_x+video_box_width//2-120,video_box_y+video_box_height//2),cv2.FONT_HERSHEY_DUPLEX,3,(255,255,255),6)
                if self.alpha2 >= 1:
                    self.alpha2 = 1
                # Update the alpha channel of the text color
                text_color = (255, 255, 255, int(self.alpha2 * 255))

                # Clear the text image
                self.text_image.fill(0)

                # Put the text on the image with the updated color
                cv2.putText(self.text_image, "Falling", (self.video_box_x+self.video_box_width//2-120,self.video_box_y+self.video_box_height//2),cv2.FONT_HERSHEY_DUPLEX,3,text_color,6)
                # Blend the text image with the main image using the alpha channel
                self.image = cv2.addWeighted(self.image, 1, self.text_image, self.alpha2, 0)
                self.alpha2 += self.fade_in_speed
                # Increment the alpha value
                self.alpha += self.fade_in_speed
        
            else:
                # Put the resized frame in the video box region of the image
                self.image[self.video_box_y:self.video_box_y + self.video_box_height, self.video_box_x:self.video_box_x + self.video_box_width] = resized_frame
        '''
            self.image[self.video_box_y:self.video_box_y + self.video_box_height, self.video_box_x:self.video_box_x + self.video_box_width] = resized_frame

        elif self.currtab == 1:
            '''
            cv2.rectangle(self.image, (self.video_box_x,self.video_box_y),(self.video_box_x+self.video_box_width,self.video_box_y+self.video_box_height),(0,0,0),2)
            indexlst = []
            Points = []
            for i in range(0,46):
                indexrow = []
                for j in range(0,46):
                    list = []
                    classifications = result[0][i][j]
                    for k in range(0,57):
                        #float32_value = np.float32(classifications[k]) / 255.0
                        # if float32_value < 0:
                        #     float32_value += 256
                        #list.append((float(classifications[k])- OUTPUT_BIAS) * OUTPUT_RATIO)
                        list.append(classifications[k])
                    classifications = list
                    classifications = [round(num, 2) for num in classifications]
                    #print(classifications)
                    indexrow.append(classifications.index(max(classifications)))
                    newarr = np.array(classifications)
                        #classifications = [round(num, 4) for num in classifications]
                indexlst.append(indexrow)
                    #print(classifications)
            for a in range(0,18):
                coordinates= (-1,-1)
                maxconfidence = -1
                for b in range(0,46):
                    for c in range(0,46):
                        if indexlst[b][c] == a:
                            #print(result[0][b][c][a])
                            if result[0][b][c][a] > maxconfidence and result[0][b][c][a] >= self.threshold:
                                maxconfidence = result[0][b][c][a]
                                coordinates = (b,c)
                Points.append(coordinates)
                '''
            
            # Resize the frame to fit within the box dimensions
            resized_frame = cv2.resize(frame, (self.video_box_width, self.video_box_height))
            # Put the resized frame in the video box region of the image
            self.image[self.video_box_y:self.video_box_y + self.video_box_height, self.video_box_x:self.video_box_x + self.video_box_width] = resized_frame
            # for pt in result:
            #     if pt[0] != -1:
            #         newx = pt[0]/46*self.video_box_width
            #         newy = pt[1]/46*self.video_box_width
            #         pt = (newx,newy) 
            for i in range(len(result)):
                if result[i][0] != -1:
                    if str(i) in self.faceIndex:
                        # print(self.video_box_x+self.video_box_width-result[i][0])
                        # print(self.video_box_y+result[i][1])
                        #cv2.circle(self.image,(self.video_box_x+self.video_box_width-result[i][0],self.video_box_y+result[i][1]),10,(0,0,0),4)
                        cv2.circle(self.image,(self.video_box_x+result[i][0],self.video_box_y+result[i][1]),10,self.POINT_COLORS[0],4)
                    else:
                        cv2.circle(self.image,(self.video_box_x+result[i][0],self.video_box_y+result[i][1]),5,self.POINT_COLORS[i],4)
            for linept in self.LINE_POINTS:
                startIndex = linept[0]
                endIndex = linept[1]
                startPt = result[startIndex]
                endPt = result[endIndex]
                if(startPt[0] != -1 and endPt[0] != -1):
                    cv2.line(self.image, (self.video_box_x+startPt[0],self.video_box_y+startPt[1]),(self.video_box_x+endPt[0],self.video_box_y+endPt[1]),self.COLOR_LINE,3)
            

        # elif self.currtab == 2:
        #     # Resize the frame to fit within the box dimensions
        #     resized_frame = cv2.resize(frame, (self.video_box_width, self.video_box_height))
        #     # Put the resized frame in the video box region of the image
        #     self.image[self.video_box_y:self.video_box_y + self.video_box_height, self.video_box_x:self.video_box_x + self.video_box_width] = resized_frame

        # # Display the image in the window
        cv2.imshow("My Window", self.image)
        #time.sleep(0.3)
        # #self.cap.release()
        # # Destroy the window
        # #cv2.destroyAllWindows()
    
    def check_key(self):
        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF
        # Exit the loop if 'q' key is pressed
        if key == ord('q'):
            return True
        return False

    

    
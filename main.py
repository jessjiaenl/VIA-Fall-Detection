import subprocess

def read_from_cam():
    return 0

def render():
     #Render onto the GUI
     return 0

quit = False

while(not quit):
      pic = read_from_cam()
      result = subprocess.check_output(["python", "Fall_Detection.py", pic])
      render(result)




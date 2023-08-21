# VIA-Fall-Detection 

**Problem:** Each year an estimated 684, 000 individuals die from falls (WHO). By the power of computer vision, a person may transmit SOS signals to the rescue task forces during a fall before he or she hits the ground or crushes any communication equipments. Fall detection improves the safety for people who work in dangerous environments or do extreme sports.
 
**Solution Proposal:** By leveraging computer vision techniques, we aim to build a system that transmits SOS signals to the rescue task forces during a fall. This is especially important when communication equipments are crushed after falling. To detect falling movements, two models are developed. The first model extracts features out of each video frame. The second model analyzes the outputs from from model 1 by taking in an array of 8 frames as a sliding window to detect falling movements.

# Complete System Illustration (MobileNet Object Detection, Openpose, Fall Detection)

**Overview of system features:** 

- The diagram is a complete illustration of the system that integrates object detection, openpose, and fall detection.
- Object Detection and Open pose are adapted from open source projects
- Fall Detection is custom made and trained
</br>

**Display interface and Hardware Usage:**

- OpenCV was used to create the display GUI
- Hardware usage: all system features are run and compiled on VIA's VAB-912.

![Systemillustration](mermaid-diagram-2023-07-18-134540.png)


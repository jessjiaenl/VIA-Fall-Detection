from os import walk
from PIL import Image
import sys
import numpy as np

def convertImageToBin(filename,w,h):
	image = Image.open(filename)
	image = image.resize((w,h), Image.ANTIALIAS)

	data = np.array(image, dtype=np.float32)
	data = (data / 128.0) - 1
	dataq = np.array(image, dtype=np.uint8)

	data.tofile("input_"+str(w)+str(h)+"_f.bin")
	dataq.tofile("input_"+str(w)+str(h)+"_q.bin")
	print ("Convert" , filename , "success ,Data:",data)


if __name__ == '__main__':
	print("input input_image , output_bin , w , h")
	convertImageToBin(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))


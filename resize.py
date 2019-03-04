import os, sys
import cv2
# import imread

def main():
    if(len(sys.argv)==2):
        filename = sys.argv[1]
        img = cv2.imread(filename)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        new_filename = filename[:-4] + "_resize.jpg"
        cv2.imwrite(new_filename, img)
if __name__ == "__main__":
    main()
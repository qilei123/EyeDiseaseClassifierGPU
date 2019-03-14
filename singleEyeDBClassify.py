import os
import sys
sys.path.append("/media/cql/DATA1/Development/opencv/install33/lib")
sys.path.append("/usr/local/cuda/lib64")
def classify(img_dir):
    p = os.popen('./CBD '+img_dir)
    x = p.read()
    xl = x.split('\n')
    return int(xl[-3])

def main():
    for i in range(10):
        print classify('./Data/921607893.jpg')

#main()

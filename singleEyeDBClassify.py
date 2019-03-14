import os
import sys
from os.path import abspath, join, dirname
sys.path.insert(0, abspath(dirname(__file__)))
def classify(img_dir):
    p = os.popen('CBD '+img_dir)
    x = p.read()
    xl = x.split('\n')
    return int(xl[-3])

def main():
    for i in range(10):
        print classify('/home/ubuntu/EyeDiseaseClassifierGPU/921607893.jpg')

main()

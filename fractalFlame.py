import time
from fractal import flame
import sys

def main():
    t1 = time.time()
    img = flame(xres = 2560, yres = 1600, ymin = -1.1, ymax = 1.9, xmin = -2.4, xmax = 2.4, sup = 4,
    samples = 50000, seed = 123456789, itterations = 3000, gamma = 1.5,
    threads = 7)
    img.load_file("")
    img.render()
    img.barrier.wait()
    img.save(str(sys.argv[1]))
    print("done in %.3f"%(time.time() - t1))
    
if __name__ == "__main__":
    main()

import numpy as np
import random
import functions as f
import math
from PIL import Image
from threading import (Thread, Lock, Barrier)
import sys

class variation:
    def __init__(self, weight = 1, type = f.linear, params = [0,0,0,0]):
        self.weight = weight
        self.type = type
        self.params = params

class function:
    def __init__(self, color = 1, probability = 1, variations = [variation()], coefs = [1,0,0,0,1,0], post = [1,0,0,0,1,0]):
        self.probability = probability
        self.variations = variations
        self.coefs = coefs
        self.post = post
        self.color = color

class flame:
    def __init__(self, xres = 1920, yres = 1080, sup = 1, samples = 20000,
        seed = None, xmin = -1.777, xmax = 1.777, ymin = -1, ymax = 1,
        functions = [function()],final = function(variations = []), threads = 1,
        gamma = 1, itterations = 1000, colors = np.zeros((256,3),dtype=np.uint8)):
        self.xres = xres
        self.yres = yres
        self.sup = sup
        self.samples = samples
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.itters = itterations
        self.gamma = gamma
        self.colors = colors
        self.threads = threads

        self.pixels = np.zeros((yres*sup, xres*sup,4),dtype=np.uint8)
        self.functions = functions
        self.final = final
        self.randGen = random
        self.randGen.seed(seed)

        self.mutex = Lock()
        self.barrier = Barrier(threads + 1)

        self.rand=lambda hi,lo: lo + ((hi - lo) * self.randGen.random())

    def load_file(self, path):

        #provisorisches init

        #provisorical & crude definition of the generated flames
        self.functions = [
            function(
                color = 1,
                probability = 2,
                coefs = [1,0,.1,0,1,.1],
                variations = [
                    variation(type = f.ex),
                    variation(type = f.popcorn)
                ]
            ),
            function(
                color = 0,
                probability = 1,
                coefs = [1, 0.1, 0, 0.1, 1, 0],
                post = [1.5,0,0,0,1.5,2],
                variations = [
                    variation(type = f.handkerchief),
                    variation(type = f.bubble),
                    variation(type = f.gaussian, params = [2,1,0,0])
                ]
            ),
            function(
                probability = 0.5,
                color = 2,
                variations = [
                    variation(type = f.swirl),
                    variation(type = f.bubble)
                ]
            )
        ]

        '''function(
            color = 0,
            probability = 1.5,
            coefs= [-1.95312, 0, 0, -1.95312, -1.43956, -0.186071],
            post= [0.054203, 0, 0, 0.054203, -0.041426, -0.011715],
            variations = [
                variation()
            ]
        ),'''

        '''function(
            color = 0,
            probability = 1.5,
            coefs = [0.134839, 0.269271, -0.287245, 0.124935, 1.40241, -0.169667],
            variations = [
                variation(
                    type = f.julia
                )
            ]
        ),'''

        '''function(
                color = 0,
                probability = 1.5,
                variations = [
                    variation(
                        type = f.waves,
                        params = [0.269271, -0.287245, 1.40241, -0.169667]
                    )
                ]
        ),'''
        #self.final = function(variations = [])'''

        '''self.functions = [
            function(
                color = 1,
                coefs = [-0.625568, -0.515804, 0.620578, -0.616063, 0, 0.0682234],
                post = [1, -0, 0, -1, 0.0733487, 0.00802098],
                variations = [
                    variation(type = f.julia)
                ]
            ),
            function(
                color = 2,
                coefs= [1.01797, -0.0661345, 0.071984, 0.699828, -0.588647, 0.031761],
                post = [-1.02628, 0.00211043, -0.028899, 0.999847, 0.047017, 0.0402695],
                variations = [
                    variation(
                        weight = 0.1,
                        type = f.juliaN,
                        params = [6.9,0.05,0,0]
                    ),
                    variation(
                        weight = 0.055,
                        type = f.juliaScope,
                        params = [16.9,0.069,0,0]
                    )
                ]
            )
        ]

        self.final = function(
            color = 0,
            coefs = [-0.695157, 0.00633793, 0.0475732, -0.644204, -0.270785, 0.892879],
            post = [-1.0046, 0.0974962, -0.0416929, 0.966926, -0.59723, -1.12878],
            variations = [
                variation(
                    weight = 1.2,
                    type = f.curl,
                    params = [0,0.4,0,0]
                )
            ]
        )

        self.functions = [
            function(
                color = 1,
                probability = 1.899,
                coefs = [15.4559, 15.4559, -15.4559, 15.4559, -0.1, 0],
                variations = [
                    variation(type = f.julia)
                ]
            ),
            function(
                color = 1,
                coefs= [0.093564, 0.008978, -0.008978, 0.093564, 0, 0],
                variations = [
                    variation(
                        weight = 0.1,
                        type = f.juliaN,
                        params = [6.9,0.05,0,0]
                    ),
                    variation(
                        weight = 0.055,
                        type = f.juliaScope,
                        params = [16.9,0.069,0,0]
                    )
                ]
            )
        ]

        self.final = function(
            color = 1,
            coefs = [1,0,0,1,0,0],
            variations = [
                variation(
                    weight = 1,
                    type = f.curl,
                    params = [0.020472,0.167495,0,0]
                )
            ]
        )'''

        color = self.colors
        color[0] = [0,100,127]
        color[1] = [127,6,255]
        color[2] = [0,255,0]

    def render(self):
        for _ in range(self.threads):
            t = Thread(target=self.render_t)
            t.start()

    def render_t(self):
        print("starting render")
        progress = 0
        for sample in range(int(self.samples/self.threads)):
            cur_x = self.rand(self.xmin, self.xmax)
            cur_y = self.rand(self.ymin, self.ymax)

            for step in range(self.itters):
                # Select transformation
                prob = []
                for function in self.functions:
                    prob.append(function.probability)
                function = self.functions[self.randGen.choices(range(len(self.functions)), weights=prob)[0]]
                #Apply selected transformation
                try:
                    cur_x, cur_y = itterate(cur_x, cur_y, function)
                #Apply Final transformation
                    cur_x, cur_y = itterate(cur_x, cur_y, self.final)

                except:
                    print("Stopped Sample %i in Step %i because of an %s"%(sample,step,str(sys.exc_info()[1])))
                    break

                #Plot point exept the first 20
                if step >= 20:

                    #if point is in image
                    if self.xmin < cur_x < self.xmax and self.ymin < cur_y < self.ymax:
                        x = int( (cur_x - self.xmin)/(self.xmax - self.xmin) * self.xres * self.sup )
                        y = int( (cur_y - self.ymin)/(self.ymax - self.ymin) * self.yres * self.sup )
                        if 0 < x < self.xres * self.sup and 0 < y < self.yres * self.sup:
                            self.mutex.acquire(1)
                            point = self.pixels[y][x]

                            #Blend colors
                            r, g, b = self.colors[function.color]
                            point[0] = int(r / 2.0 + point[0] / 2.0)
                            point[1] = int(g / 2.0 + point[1] / 2.0)
                            point[2] = int(b / 2.0 + point[2] / 2.0)

                            point[3] += 1 # Hitcounter
                            self.mutex.release()
            #print(sample,"/",int(self.samples/self.threads))
        self.barrier.wait()

    def reduce(self,m,pixels,image,col_bar,save_bar,mut):
        local_max = 0
        for x in range(m,self.xres,self.threads):
            #print("Thread %i does row %i"%(m,x))
            for y in range(self.yres):
                col = [0,0,0,0]
                for i in range(self.sup):
                    for j in range(self.sup):
                        col += self.pixels[y*self.sup+i][x*self.sup+j]
                pixels[y][x] = col / (self.sup * self.sup)
                local_max = max(local_max, col[3])

        mut.acquire()
        self.max_freq = max(local_max, self.max_freq)
        mut.release()

        print("%i waits"%(m))
        col_bar.wait()

        for x in range(m,self.xres,self.threads):
            #print("Thread %i does row %i"%(m,x))
            for y in range(self.yres):
                if(pixels[y][x][3] > 0):
                    image[y][x] = pixels[y][x][:3] * ((math.log(pixels[y][x][3],self.max_freq) ** (1.0/self.gamma)))
        print("%i waits"%(m))
        save_bar.wait()

    def save(self,path):
        print("reduce")

        self.max_freq = 1
        pixels = np.zeros((self.yres , self.xres ,4), dtype = np.uint8)
        image = np.zeros((self.yres , self.xres ,3), dtype = np.uint8)
        mut = Lock()
        col_bar = Barrier(self.threads + 1)
        save_bar = Barrier(self.threads + 1)
        for m in range(self.threads):
            t = Thread(target=self.reduce, args=(m,pixels,image,col_bar,save_bar,mut))
            t.start()

        col_bar.wait()
        print("apply gamma correction")
        save_bar.wait()
        Image.fromarray(image).save(path)


def itterate(cur_x,cur_y,function):
    for variation in function.variations:
        func = variation.type
        weight = variation.weight
        a, b, c, d, e, f = function.coefs
        pa1, pa2, pa3, pa4 = variation.params
        x = a * cur_x + b * cur_y + c
        y = d * cur_x + e * cur_y + f

        x, y = func(x, y, a, b, c, d, e, f, pa1, pa2, pa3, pa4)

        cur_x = weight * x
        cur_y = weight * y

    #Post transformation
    a, b, c, d, e, f = function.post
    cur_x = a * cur_x + b * cur_y + c
    cur_y = d * cur_x + e * cur_y + f
    return cur_x, cur_y

from math import (sin, cos, tan, sqrt, atan2, exp, pow, hypot, pi, isfinite)
import random as rand
import sys

def linear (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    return x, y

def sinusoidal (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    return sin(x), sin(y)

def spherical (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = 1.0 / (x * x + y * y)
    return r * x, r * y

def swirl (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = (x * x) + (y * y)
    return x * sin(r) - y * cos(r), x * cos(r) + y * sin(r)

def horseshoe (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = 1.0 / hypot(x, y)
    return r * (x - y) * (x + y), r * 2.0 * x * y

def polar (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    return atan2(y, x) / pi, hypot(x, y) - 1.0

def handkerchief (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = hypot(x, y)
    theta = atan2(y, x)
    return r * sin(theta + r), r * cos(theta - r)

def heart (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = hypot(x, y)
    theta = atan2(y, x)
    return r * sin(theta * r), -r * cos(theta * r)

def disk (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = hypot(x, y) * pi
    theta = atan2(y, x) / pi
    return theta * sin(r), theta * cos(r)

def spiral (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = hypot(x, y)
    theta = atan2(y, x)
    return (1.0 / r) * (cos(theta) + sin(r)), (1.0 / r) * (sin(theta) - cos(r))

def hyperbolic (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = hypot(x, y)
    theta = atan2(y, x)
    return sin(theta) / r, r * cos(theta)

def diamond (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = hypot(x, y)
    theta = atan2(y, x)
    return sin(theta) * cos(r), cos(theta) * sin(r)

def ex (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = hypot(x, y)
    theta = atan2(y, x)
    i = sin(theta + r)
    i = i * i * i
    j = cos(theta - r)
    j = j * j * j
    return r * (i + j), r * (i - j)

def julia (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = sqrt (hypot(x, y))
    theta = atan2(y, x) * 0.5
    if rand.getrandbits(1):
        theta += pi
    return r * cos(theta), r * sin(theta)

def bent (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    if x >= 0.0 and y >= 0.0:
        return x, y
    if x < 0.0 and y >= 0.0:
        return 2.0 *x, y
    if x >= 0.0 and y < 0.0:
        return x, y * 0.5
    return 2.0 * x, y * 0.5

def waves (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    return x + pa1 * sin(y / (pa2 * pa2)), y + pa3 * sin(x / (pa4 * pa4))

def fisheye (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = 2.0 / (1.0 + hypot(x, y))
    return r * y, r * x

def popcorn (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    return x + c * sin(tan (3.0 * y)), y + f * sin(tan (3.0 * x))

def exponential (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    return exp (x - 1.0) * cos(pi * y), exp (x - 1.0) * sin(pi * y)

def power (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = hypot(x, y)
    theta = atan2(y, x)
    return pow(r, sin(theta)) * cos(theta), pow(r, sin(theta)) * sin(theta)

def cosine (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    return cos(pi * x) * cosh (y), -sin(pi * x) * sinh (y)

def rings (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = hypot(x, y)
    theta = atan2(y, x)
    p = pa2 * pa2
    prefix =((r + p) % (2.0 * p)) - p + (r * (1.0 - p))
    return prefix * cos(theta), prefix * sin(theta)

def fan (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = hypot(x, y)
    theta = atan2(y, x)
    t = pi * c * c / 2 + sys.float_info.epsilon
    if (theta % (t * 2)) > t  :
        return r * cos(theta - t), r * sin(theta - t)
    else:
        return r * cos(theta + t), r * sin(theta + t)

def eyefish (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = 2.0 / (1.0 + hypot(x, y))
    return r * x, r * y

def bubble (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = 4 + x * x + y * y
    return (4.0 * x) / r, (4.0 * y) / r

def cylinder (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    return sin(x), y

def tangent (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    return sin(x) / cos(y), tan(y)

def cross (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = sqrt (1.0 / ((x * x - y * y) * (x * x - y * y)))
    return x * r, y * r

def collatz (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    return 0.25 * (1.0 + 4.0 * x - (1.0 + 2.0 * x) * cos(pi * x)), 0.25 * (1.0 + 4.0 * y - (1.0 + 2.0 * y) * cos(pi * y))

def mobius (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    t = (pa3 * x + pa4) * (pa3 * x + pa4) + pa3 * y * pa3 * y
    return ((pa1 * x + pa2) * (pa3 * x + pa4) + pa1 * pa3 * y * y) / t, (pa1 * y * (pa3 * x + pa4) - pa3 * y * (pa1 * x + pa2)) / t

def blob (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    r = hypot(x, y)
    theta = atan2(y, x)
    newx = r * (pa2 + 0.5 * (pa1 - pa2) * (sin(pa3 * theta) + 1)) * cos(theta)
    newy = r * (pa2 + 0.5 * (pa1 - pa2) * (sin(pa3 * theta) + 1)) * sin(theta)
    return nwx, newy

def noise (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    theta = rand.random()
    r = rand.random()
    return theta * x * cos(2 * pi * r), theta * y * sin(2 * pi * r)

def blur (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    theta = rand.random()
    r = rand.random()
    return theta * cos(2 * pi * r), theta * sin(2 * pi * r)

def square (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    return rand.random() - 0.5, rand.random() - 0.5

def notBrokenWaves (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4):
    return x + b * sin(y / pow (c, 2.0)), y + e * sin(x / pow (f, 2.0))

def juliaN (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4): #pa1 = power, pa2 = dist
    r = hypot(x, y)
    p3 = int(abs(pa1) * rand.random())
    t = (atan2(x, y) + 2 * pi * p3)/pa1
    return r**(pa2/pa1) * cos(t), r**(pa2/pa1) * sin(t)

def juliaScope (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4): #pa1 = power, pa2 = dist
    r = hypot(x, y)
    p3 = int(abs(pa1) * rand.random())
    t = (rand.choice([-1,1])*atan2(x, y) + 2 * pi * p3)/pa1
    return r**(pa2/pa1) * cos(t), r**(pa2/pa1) * sin(t)

def curl (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4): # pa1 = c1, pa2 = c2
    t1 = 1 + (pa1*x) + (pa2 * (x**2 - y**2))
    t2 = (pa1 * y) + (2 * pa2 * x * y)
    return (1/(t1**2 + t2**2)) * (x*t1 + y*t2), (1/(t1**2 + t2**2)) * (y*t1 - x*t2)

def gaussian (x,y,a,b,c,d,e,f,pa1,pa2,pa3,pa4): #pa1 = mu, pa2 = sigma
    t = rand.gauss(pa1, pa2)
    r = 2* pi * rand.random()
    return t * cos(r), t * sin(r)

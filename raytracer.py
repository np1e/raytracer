from PIL import Image
import numpy as np
import math




BACKGROUND_COLOR = np.array([255,255,255,255])

objectlist = []

fov = 45 # 45 degrees
aspect_ratio = 1 # 2:1
wRes = 400
hRes = 400

e = np.array([0, 1.8, 10])
c = np.array([0, 3, 0])
up = np.array([0, 1, 0])

class Ray(object):

    def __init__(self, origin, direction):
        self.origin = origin # point
        self.direction = normalized(direction) # vector

    def pointAtParameter(self, t):
        return self.origin + np.multiply(self.direction, t)

class Sphere(object):
    def __init__(self, center, radius):
        self.center = center # point
        self.radius = radius # scalar

    def __repr__(self):
        return "Sphere({}, {}".format(self.center, self.radius)

    def intersectionParameter(self, ray):
        co = self.center - ray.origin # vector
        v = co.dot(ray.direction) # vector
        discriminant = v**2 - co.dot(co) + self.radius**2
        if discriminant < 0:
            return None
        else:
            return v - math.sqrt(discriminant)

    def normalAt(self, p):
        return (normalized(p - self.center))

class Plane(object):

    def __init__(self, point, normal):
        self.point = point # point
        self.normal = normalized(normal) # vector

    def __repr__(self):
        return "Plane ({}, {})".format(self.point, self.normal)

    def intersectionParameter(self, ray):
        op = ray.origin - self.point
        a = op.dot(self.normal)
        b = ray.direction.dot(self.normal)
        if b:
            return -a/b
        else:
            return None

    def normalAt(self, p):
        return self.normal

class Triangle(object):
    def __init__(self, a, b, c):
        self.a = a # point
        self.b = b # point
        self.c = c # point
        self.u = self.b - self.a # vector ab
        self.v = self.c - self.a # vector ac

    def __repr__(self):
        return "Triangle({}, {}, {})".format(self.a, self.b, self.c)

    def intersectionParameter(self, ray):
        w = ray.origin - self.a
        dv = np.cross(ray.direction, self.v)
        dvu = dv.dot(self.u)
        if dvu == 0.0:
            return None
        wu = np.cross(w, self.u)
        r = dv.dot(w) / dvu
        s = wu.dot(ray.direction) / dvu
        if 0 <= r and r <= 1 and 0 <= s and s <= 1 and r+s <= 1:
            return wu.dot(self.v) / dvu
        else:
            return None

    def normalAt(self, p):
        return normalized(np.cross(self.u, self.v))

def normalized(vector):
    return vector / np.linalg.norm(vector)

def calcRay(x, y, s, u, f):
    xcomp = np.multiply(s, x * pixelWidth - width/2)
    ycomp = np.multiply(u, y * pixelHeight - height/2)
    ray = Ray(np.array(e), f + xcomp + ycomp)
    return ray

if __name__ == "__main__":
    height = 2 * np.tan(np.radians(fov))
    width = aspect_ratio * height

    pixelWidth = width / (wRes - 1)
    pixelHeight = height / (hRes - 1)

    f = normalized(c - e)
    s = normalized(np.cross(f, up))
    u = np.cross(s, f)

    sphere = Sphere(np.array([2,5,0]), 2)
    triangle1 = Triangle(np.array([7, 0, 0]), np.array([7, 2, 2]), np.array([7, 2, -2]))

    objectlist.append(sphere)
    objectlist.append(triangle1)

    image = np.zeros((wRes, hRes, 4), dtype= np.ndarray)

    for x in range(wRes):
        for y in range(hRes):
            ray = calcRay(x, y, s, u, f)
            maxdist = float('inf')
            color = BACKGROUND_COLOR
            """if x%2:
                color = np.array([255, 255, 255, 255])
            else:
                color = np.array([0, 0, 0, 255])"""
            for object in objectlist:
                hitdist = object.intersectionParameter(ray)
                if hitdist:
                    if hitdist < maxdist:
                        maxdist = hitdist
                        Y = (hitdist - 0) / (500 - 0) * (255 - 0) + 0
                        color = np.array([Y, Y, Y, 255])
                        #object.colorAt(ray)

            image[y][x] = color


    im = Image.fromarray(image.astype('uint8'), 'RGBA')
    im.save("your_file.bmp")


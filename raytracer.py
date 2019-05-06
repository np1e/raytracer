from PIL import Image
import numpy as np
import math, time, datetime
from threading import Thread


# material constants
ka = 0.1
kd = 0.6
ks = 0.1
n = 6
ca = np.array([255, 255, 255])
reflection = 0.2

BACKGROUND_COLOR = np.array([0,0,0])

objectlist = []

maxlevel = 3
threadCount = 0

# camera parameter
fov = 45 # 45 degrees
aspect_ratio = 1 # 1:1
wRes = 400
hRes = 400
light_sources = [np.array([200, -300, 300])]

e = np.array([0, 0, 0])
c = np.array([100, 0, 0])
up = np.array([0, 1, 0])

class CheckerBoardMaterial(object):
    def __init__(self):
        self.baseColor = np.array([255,255,255])
        self.otherColor = np.array([0,0,0])
        self.ambientCoefficient = 1.0
        self.diffuseCoefficient = 0.6
        self.specularCoefficient = 0.2
        self.checkSize = 1

    def baseColorAt(self, p):
        v = p
        v *= (1.0 / self.checkSize)
        if (int(abs(v[0]) + 0.5) + int(abs(v[1]) + 0.5) + int(abs(v[2]) + 0.5)) % 2:
            return self.otherColor
        return self.baseColor


class Ray(object):

    def __init__(self, origin, direction):
        self.origin = origin # point
        self.direction = normalized(direction) # vector

    def pointAtParameter(self, t):
        return self.origin + np.multiply(self.direction, t)


class Sphere(object):
    """
    creates a Sphere with a given center, radius and color
    """
    def __init__(self, center, radius, color):
        self.center = center # point
        self.radius = radius # scalar
        self.color = color
        self.material = None

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
    """creates a Plane with a given point, normal and color"""
    def __init__(self, point, normal, color):
        self.point = point # point
        self.normal = normalized(normal) # vector
        self.color = color
        self.material = None

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
    """creates a Triangle with three given points and a given color"""
    def __init__(self, a, b, c, color):
        self.a = a # point
        self.b = b # point
        self.c = c # point
        self.u = self.b - self.a # vector ab
        self.v = self.c - self.a # vector ac
        self.color = color
        self.material = None

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



def phongColor(object, ray, hitdist):

    p = ray.origin + hitdist * ray.direction # intersection point of ray and object
    normal = object.normalAt(p) # normal of the object

    if object.material:
        return object.material.baseColorAt(p)

    cin = object.color

    ambient_light = ca * ka
    cout = ambient_light

    for light in light_sources:
        l = normalized(light - p) # lichtvektor
        lr = normalized(l - 2 * normal.dot(l) * normal)#normalized(l + 2 * (s-l)) # reflexionsvektor


        if l.dot(normal) > 0:
            cos_phi = l.dot(normal)
        else:
            cos_phi = 0

        if lr.dot(-ray.direction) > 0:
            cos_theta = lr.dot(-ray.direction)
        else:
            cos_theta = 0

        diffuse = cin * kd * cos_phi # diffuser Anteil, unabhängig vom Betrachter
        specular = cin * ks * (n+2)/(2 * math.pi) * cos_theta ** n # spekularer Anteil, abhängig vom Betrachtungswinkel

        cout += diffuse + specular


        # shadow

        light_ray = Ray(p, l)
        for obj in [o for o in objectlist if o != object]:
            t = obj.intersectionParameter(light_ray)
            if t and t > 0:
                cout -= 50

    return cout


def normalized(vector):
    return vector / np.linalg.norm(vector)


def calcRay(x, y, s, u, f):
    xcomp = np.multiply(s, x * pixelWidth - width/2)
    ycomp = np.multiply(u, y * pixelHeight - height/2)
    ray = Ray(np.array(e), f + xcomp + ycomp)
    return ray

def traceRay(level, ray):
    hitPointData = intersect(level, ray, maxlevel)
    # if max recursion depth not reached yet and ray intersects with an object
    if hitPointData:
        return shade(level, hitPointData)
    return BACKGROUND_COLOR

def intersect(level, ray, maxlevel):
    # max recursion depth reached
    if level == maxlevel:
        return None

    maxdist = float('inf')
    hitPointData = None

    for object in objectlist:
        hitdist = object.intersectionParameter(ray)
        if hitdist:
            if hitdist < maxdist:
                maxdist = hitdist
                hitPointData = {
                    "hitdist": hitdist,
                    "object": object,
                    "ray": ray
                }

    return hitPointData


def shade(level, hitPointData):
    # compute phong color
    directColor = computeDirectLight(hitPointData)


    reflectedRay = computeReflectedRay(hitPointData)

    # compute the reflect color by tracing the reflected ray and incrementing the recursion level by 1
    reflectColor = traceRay(level+1, reflectedRay)

    return directColor + reflection*reflectColor

def computeDirectLight(hitPointData):
    return phongColor(hitPointData["object"], hitPointData["ray"], hitPointData["hitdist"])

def computeReflectedRay(hitPointData):
    ray = hitPointData["ray"]
    object = hitPointData["object"]
    hitdist = hitPointData["hitdist"]

    p = ray.origin + hitdist * ray.direction  # intersection point of ray and object
    normal = object.normalAt(p)
    dr = Ray(p, ray.direction - 2 * normal.dot(ray.direction) * normal)
    return dr

def calcPixel(x, y):
    ray = calcRay(x, y, s, u, f)
    color = traceRay(0, ray)
    image[y][x] = color

if __name__ == "__main__":
    height = 2 * np.tan(np.radians(fov))
    width = aspect_ratio * height

    pixelWidth = width / (wRes - 1)
    pixelHeight = height / (hRes - 1)

    # define camera
    f = normalized(c - e)
    s = normalized(np.cross(f, up))
    u = np.cross(s, f)

    sphere = Sphere(np.array([500,-150,0]), 120, np.array([255,0,0]))
    sphere1 = Sphere(np.array([500, 150, 150]), 120, np.array([0,255,0]))
    sphere2 = Sphere(np.array([500, 150, -150]), 120, np.array([0, 0, 255]))
    triangle = Triangle(np.array([600, -100, 0]), np.array([600, 100, -150]), np.array([600, 100, 150]), np.array([255,255,0]))

    plane = Plane(np.array([0,350,0]), up, np.array([180,180,180]))
    plane.material = CheckerBoardMaterial()

    objectlist = [sphere, sphere1, sphere2, triangle, plane]

    #objectlist.append(triangle1)
    #objectlist.append(plane)

    image = np.zeros((wRes, hRes, 3), dtype= np.ndarray)

    for x in range(wRes):
        for y in range(hRes):
            """
            # parallelize with threading - new thread for calcRay
            threadCount += 1
            t = Thread(target=calcPixel, args = (x,y, threadCount))
            t.start()
            
            """
            ray = calcRay(x, y, s, u, f)
            color = traceRay(0, ray)
            """
            maxdist = float('inf')
            color = BACKGROUND_COLOR
            for object in objectlist:
                hitdist = object.intersectionParameter(ray)
                if hitdist:
                    if hitdist < maxdist:
                        maxdist = hitdist
                        color = phongColor(object, ray, hitdist)
                        """
            image[y][x] = color


    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y_%H:%M:%S')

    im = Image.fromarray(image.astype('uint8'), 'RGB')
    im.save("images/scene_{}.bmp".format(timestamp))


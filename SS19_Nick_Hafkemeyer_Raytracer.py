import numpy as np
from PIL import Image
import math, time, datetime
from threading import Thread
import multiprocessing as mp
from multiprocessing import Pool


class World(object):

    """
    def __init__(self, file):
        objects = []
        with open(file) as f:
            for line in f:
                if line == "# objects":
                    objects.append(eval(line))
                elif line == "# camera":
                    if line.startswith("c"):

                elif line == "# lights":
   """

    def __init__(self, objects, camera, lights, wRes = 400, hRes = 400, aspectRatio = 1,
                 bgColor = np.array([0,0,0]), ca = np.array([255,255,255])):
        self.objects = objects # list with objects
        self.camera = camera # Camera object
        self.lights = lights # list with numpy.ndarray's
        self.wRes = wRes # int
        self.hRes = hRes # int
        self.ca = ca
        self.height = 2 * np.tan(np.radians(self.camera.fov))
        self.width = aspectRatio * self.height

        self.pixelWidth = self.width / (wRes - 1)
        self.pixelHeight = self.height / (hRes - 1)

        self.bgColor = bgColor



    def render(self):
        """"""

        image = np.zeros((self.wRes, self.hRes, 3), dtype=np.ndarray)

        with Pool() as pool:
            for y in range(self.hRes):
                for x in range(self.wRes):
                        pool.apply(self._renderPixel, args=(x,y,image))
                        #self._renderPixel(x,y,image)

        pool.close()
        return image

    def _renderPixel(self, x, y, image):

        ray = self._calcRay(x, y, self.camera)

        color = self._traceRay(0, ray)
        image[y][x] = color

    def _calcRay(self, x, y, camera):
        """ Return a ray through x and y from a camera"""

        s = camera.s
        u = camera.u
        f = camera.f

        xcomp = s * (x * self.pixelWidth - self.width / 2)
        ycomp = u * (y * self.pixelHeight - self.height / 2)
        ray = Ray(camera.e, f + xcomp + ycomp)

        return ray

    def _traceRay(self, level, ray):

        hitPointData = self._intersect(ray, level)

        if hitPointData:
            return self._shade(level, hitPointData)

        return self.bgColor

    def _intersect(self, ray, level, maxlevel = 2):

        # max recursion depth reached
        if level >= maxlevel:
            return None

        maxdist = float('inf')
        hitPointData = None

        for object in self.objects:
            hitdist = object.intersectionParameter(ray)
            if hitdist:
                if 0 < hitdist and hitdist < maxdist:
                    maxdist = hitdist
                    hitPointData = object, hitdist, ray

        return hitPointData

    def _shade(self, level, hitPointData):
        object, hitdist, ray = hitPointData
        intersection = ray.origin + hitdist * ray.direction
        normal = object.normalAt(intersection)

        # compute color using the phong model
        directColor = self._computeDirectLight(hitPointData)

        # compute the reflect color by tracing the reflected ray and incrementing the recursion level by 1
        # V + (2 * N * c1 )
        reflectedRay = Ray(intersection, ray.direction - 2 * normal * normal.dot(ray.direction))
        reflectColor = self._traceRay(level + 1, reflectedRay)

        # refractedRay = computeRefractedRay(hitPointData)
        # refractedColor = self._traceRay(level + 1, refractedRay

        return directColor + object.material.specular * reflectColor # + refraction * refractedColor


    def _computeDirectLight(self, hitPointData):
        """Compute the direct light at a given point of an object using the phong model"""

        object, hitdist, ray = hitPointData
        material = object.material

        p = ray.origin + hitdist * ray.direction  # intersection point of ray and object
        normal = object.normalAt(p)  # normal of the object

        cin = np.array([255,255,255])

        # ambient
        ambient_light = cin * material.ambient
        cout = ambient_light

        for light in self.lights:
            l = normalized(light - p)  # lichtvektor
            lr = normalized(l - 2 * normal * normal.dot(l)) # reflexionsvektor

            # angle between normal and point-to-light vector equal to or greater than 90 degrees
            if l.dot(normal) >= 0:
                cos_phi = l.dot(normal)
            else:
                cos_phi = 0

            # angle between light reflection ray and direction of ray equal to or greater than 90 degrees
            if lr.dot(-ray.direction) >= 0:
                cos_theta = lr.dot(-ray.direction)
            else:
                cos_theta = 0

            # diffuse
            cout += cin * material.diffuse * cos_phi

            # specular
            # cout += cin * material.specular * cos_theta

            # shadow
            light_ray = Ray(p, l)
            for obj in [o for o in self.objects if o != object]:
                t = obj.intersectionParameter(light_ray)
                if t and t > 0:
                    cout *= 0.6
                    break

        return cout * material.baseColorAt(p)


class Ray(object):

    def __init__(self, origin, direction):
        self.origin = origin # point
        self.direction = normalized(direction) # vector

    def pointAtParameter(self, t):
        return self.origin + np.multiply(self.direction, t)


class Material(object):

    def __init__(self, color, ambient = 0.2, specular = 0.2, diffuse = 0.5):
        self.color = color / 255.0
        self.ambient = ambient
        self.specular = specular
        self.diffuse = diffuse

    def baseColorAt(self, p):
        return self.color

class CheckerBoardMaterial(Material):
    def __init__(self):
        super().__init__(np.array([255,255,255]))
        self.otherColor = np.array([0,0,0]) / 255.0
        self.checkSize = 16

    def baseColorAt(self, p):
        v = p
        v *= (1.0 / self.checkSize)
        if (int(abs(v[0]) + 0.5) + int(abs(v[1]) + 0.5) + int(abs(v[2]) + 0.5)) % 2:
            return self.otherColor
        return self.color


class Sphere():
    """
    creates a Sphere with a given center, radius and color
    """
    def __init__(self, center, radius, material):
        self.material = material
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



class Plane():
    """creates a Plane with a given point, normal and color"""
    def __init__(self, point, normal, material):
        self.point = point # point
        self.normal = normalized(normal) # vector
        self.material = material

    def __repr__(self):
        return "Plane ({}, {})".format(self.point, self.normal)

    def intersectionParameter(self, ray):
        op = ray.origin - self.point
        a = op.dot(self.normal)
        b = ray.direction.dot(self.normal)
        if b:
            return -(a/b)
        else:
            return None

    def normalAt(self, p):
        return self.normal


class Triangle():
    """creates a Triangle with three given points and a given color"""
    def __init__(self, a, b, c, material):
        self.material = material
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
        return normalized(np.cross(self.u, self.v)) * -1

class Camera(object):

    def __init__(self, c, e, up, fov):
        self.e = e
        self.f = normalized(c - e)
        self.s = normalized(np.cross(self.f, up))
        self.u = np.cross(self.s, self.f)
        self.fov = fov


def normalized(vector):
    """Return a normalized version of the vector given as argument.
    Vector must be a numpy.ndarray.

    Arguments:
    vector -- a numpy.ndarray with any dimension
    """
    if type(vector) is not np.ndarray:
        raise Exception("Argument 'vector' must be a numpy ndarray.")
    return vector / np.linalg.norm(vector)

def array_to_image(array):
    """Create an image from an array with each array element being an RGB color."""

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y_%H:%M:%S')

    im = Image.fromarray(array.astype('uint8'), 'RGB')
    im.save("images/scene_{}.bmp".format(timestamp))


if __name__ == "__main__":

    begin = time.time()

    print("started raytracer")

    objects = [
        Sphere(np.array([0, 22, 100]), 20,
                Material(color = np.array([255, 0, 0]))),
        Sphere(np.array([-25, -20, 100]), 20,
                Material(color = np.array([0, 255, 0]))),
        Sphere(np.array([25, -20, 100]), 20,
                Material(color = np.array([0, 0, 255]))),
        Triangle(np.array([-30, -25, 100]), np.array([30, -25, 100]), np.array([0, 30, 100]),
                Material(np.array([255, 255, 0]), specular=0)),
        Plane(np.array([0,-70,0]), np.array([0, 1, 0]),
                CheckerBoardMaterial())
    ]

    camera = Camera(e = np.array([0, 0, 0]),
                    c = np.array([0, 0, 1]),
                    up = np.array([0, -1, 0]),
                    fov = 45)

    lights = [np.array([50, 150, 0])]

    world = World(objects, camera, lights, 500, 500)

    pixels = world.render()
    array_to_image(pixels)

    end = time.time()
    duration = end - begin
    print("Execution time: {:.2f}s".format(duration))

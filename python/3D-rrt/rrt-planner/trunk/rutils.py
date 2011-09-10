# -*- coding: utf-8 -*-
from math import pi, sqrt
from time import strftime

from numpy.random import rand, uniform

INTEGRATION_TIME = 1.0
DELTA_T = 100
LINEAR_TOLERANCE = 0.25
is_debug_active = False


def debug(debug_str):
    if is_debug_active:
        print "%s - DEBUG: %s" % (strftime("%H:%M:%S"), debug_str)

def dist_3d(p, q):
    if not p:
        return 100000000
    if not q:
        return 100000000 
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    dz = p[2] - q[2]
    return sqrt(dx ** 2 + dy ** 2 + dz**2)

def biased_sampling_3d(bounds, bias, qgoal):
    if rand() < bias:
        return qgoal
    q = get_random_config_3d(*bounds)
    return q

def get_random_config_3d(lower_w, upper_w, lower_h, upper_h, lower_z, upper_z):
    x = uniform(lower_w, upper_w)
    y = uniform(lower_h, upper_h)
    z = uniform(lower_z, upper_z)
    return x, y, z

def is_near_qgoal_3d(p, q, scale):
    dx2 = (p[0] - q[0]) ** 2
    dy2 = (p[1] - q[1]) ** 2
    dz2 = (p[2] - q[2]) ** 2
    d = sqrt(dx2 + dy2 + dz2)
    if d < LINEAR_TOLERANCE * scale:
        return True
    else:
        return False

def select_nearest_node_3d(g, q):
    """Returns the nearest node according to function dist."""
    dmin = float('inf')
    vertices = g.nodes()
    for configuration in vertices:
        d = dist_3d(configuration, q)
        if d < dmin:
            dmin = d
            nearest = configuration
    return nearest



import sys
sys.path.append( '../../' )
from search import *


class ConfigSpace(object):  
    def __init__(self, g):
        
        self.width, self.height, self.depth = g.size
        self.scale = g.scale
        self.bbox3= [0, self.width, 0, self.height, 0, self.depth]
        self.data3 = g.locations
        self.obs = g.obstructions
        self.g = g
               


    def is_configuration_in_colision_3d(self, p, q):
        if  q[0] >= self.width or q[1] >= self.height or q[2] >= self.depth or q[0] < 0 or q[1] < 0 or q[2] >= self.depth or q[2] < 0:
            return True
        if not self.g.lookup.has_key("x:{0}y:{1}z:{2}".format(q[0], q[1], q[2])):
            return True
        A = self.g.lookup["x:{0}y:{1}z:{2}".format(p[0], p[1], p[2])]
        B = self.g.lookup["x:{0}y:{1}z:{2}".format(q[0], q[1], q[2])]
#        print "A B get ", A, B, self.g.get(A,B)
        if (self.g.get(A,B) != None):
            return False
        else:
            return True

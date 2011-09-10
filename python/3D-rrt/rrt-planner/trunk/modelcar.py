# -*- coding: utf-8 -*-
from math import radians, tan, cos, sin, pi


class Car(object):


    def integrate_3d(self, state, u, cspace, qgoal):
        
        x0 = state[0]
        y0 = state[1]
        z0 = state[2]
        
        x = u[0] + x0
        y = u[1] + y0
        z = u[2] + z0

        points = [(x0, y0, z0)]
        
        if cspace.is_configuration_in_colision_3d((x0, y0, z0),(x, y, z)):
            return None

        points.append((x, y, z))
        
        return x,y,z,points
    

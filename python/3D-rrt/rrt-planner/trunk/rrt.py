   # Copyright [2011] [Chris McClanahan]

   # Licensed under the Apache License, Version 2.0 (the "License");
   # you may not use this file except in compliance with the License.
   # You may obtain a copy of the License at

   #     http://www.apache.org/licenses/LICENSE-2.0

   # Unless required by applicable law or agreed to in writing, software
   # distributed under the License is distributed on an "AS IS" BASIS,
   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   # See the License for the specific language governing permissions and
   # limitations under the License.


import pygraph
from pygraph.classes.graph import graph
from pygraph.algorithms.heuristics.chow import chow
from pygraph.algorithms.minmax import heuristic_search

from rutils import *

class RRT(object):
    def __init__(self, cspace, car, qinit, qgoal, qgoal_bias, n):
        self.qinit = qinit
        self.qgoal = qgoal
        self.cspace = cspace
        self.car = car
        self.bias = qgoal_bias
        self.n = n
        s = cspace.scale
        self.INPUT = [ (s, 0, 0), (0,  s, 0), (0, 0,  s), (-s, 0, 0), ( 0, -s,  0), ( 0,  0, -s),
                       (s, s, s), (s, -s, s), (s, s, -s), (-s, s, s), (-s, -s, -s), (-s, -s,  s), (-s, s, -s), (s, -s, -s) ]
        self.attempts = []
    
    def build_rrt_3d(self):
        self.path = None
        self.plot_points = []
        debug("build_rrt_3d started")
        count = 0
        g = graph()
        g.add_node(self.qinit)
        while count < self.n:
            qrand = biased_sampling_3d(self.cspace.bbox3, self.bias, self.qgoal)
            q = self.extend_RRT_3d(g, qrand)
            if q:
                count = count + 1
                if is_near_qgoal_3d(q, self.qgoal, self.cspace.scale):
                    debug("GOAL REACHED")
                    self.nearest_qgoal_node = q
                    self.build_path(g)
                    break
                #attempted
                self.attempts.append(q)
            else:
                continue
        debug("build_rrt_3d ended")
        return g
    
    def extend_RRT_3d(self, g, qrand):
        qnear = select_nearest_node_3d(g, qrand)
        qchoosed, control, points = self.select_best_input_3d(qnear, qrand,g)
        if not g.has_node(qchoosed):
            self.plot_points.append(points)
            g.add_node(qchoosed)
            g.add_edge((qnear, qchoosed))
            g.add_edge_attribute((qnear, qchoosed), ('control', control))
##        else:
##            return None
        return qchoosed

    def select_best_input_3d(self, qnear, qrand,t):
        qchoosed = None
        best_control = None
        points = None
        dmin = float('inf')
        for control in self.INPUT:
            temp = self.car.integrate_3d(qnear, control, self.cspace, self.qgoal)
            if temp and temp != self.qinit:
                qnew = temp[:-1]
                d = dist_3d(qnew, qrand)
                if d < dmin:
                    dmin = d
                    best_control = control
                    qchoosed = qnew
                    points = temp[-1]               
        return qchoosed, best_control, points

    def build_path(self, g):
        h = chow(self.qinit, self.nearest_qgoal_node)
        h.optimize(g)
        h_search = heuristic_search
        self.path = h_search(g, self.qinit, self.nearest_qgoal_node, h)


from rrt import RRT
from gui import Drawer
from configspace import ConfigSpace
from modelcar import Car
import math, random

import sys
sys.path.append( '../../' )
from search import *

# DRAW = False

# MAX_TREE_NODES = 1000
# N_ATTEMPT = 3
# QGOAL_BIAS = 0.33
# my_car = Car()
# print 'building graph...'
# dd=70
# ll= dd*10
# obs = RandomObstructions(num=range(10), draw=DRAW)
# g = RandomGraphWithObstructions(obs, width=ll, length=ll, height=ll, density=dd, drawConnections=False, draw=DRAW)
# space = ConfigSpace(g)
# print 'running rrt...'

# for i in range(N_ATTEMPT):

#     Qinit_3d = (0, 0, 0)
#     gx = random.choice(range(dd*3, ll+1, dd))
#     gy = random.choice(range(dd*3, ll+1, dd))
#     gz = random.choice(range(dd*3, ll+1, dd))
#     print 'attempt: ',gx,gy,gz
#     Qgoal_3d = (gx, gy, gz)
#     Qgoal_3d = (dd*9, dd*9, dd*9)
	
#     rrt = RRT(space, my_car, Qinit_3d, Qgoal_3d, QGOAL_BIAS, MAX_TREE_NODES)

#     tree = rrt.build_rrt_3d()

#     if rrt.path:
#         print 'path: ', rrt.path
#         # stats
#         print 'attempted: ',len(rrt.attempts)
#         print 'finalpath: ',len(rrt.path)

#         # draw expanded
#         for value in rrt.attempts:
#             curr = (value[0], value[1], value[2])
#             sphere(pos=curr, radius=11, color=color.yellow)

#         prev = (0,0,0)

#         # draw path
#         for value in rrt.path:
#             curr = (value[0], value[1], value[2])
#             sphere(pos=curr, radius=12, color=color.green)
#             theaxis = (curr[0]-prev[0],curr[1]-prev[1],curr[2]-prev[2])
#             arrow(pos=prev, axis=theaxis, shaftwidth=3, fixedwidth=true, color=color.green)
#             prev = curr

#         # don't try again
#         break

###############
#  BENCHMARK  #
###############

if __name__ == "__main__":

	#time, length or path, nodes expanded
	#vary obstructions, length, width, height, totalNodes
	draw = True
	density = 100
	
	for size in range(1000, 1300, density):

		for obstructions in range(30, 60, 10):

			for run in range(1, 3, 1):

				obs = RandomObstructions(num=range(obstructions), width=size, length=size, height=size, draw=draw)
				g = RandomGraphWithObstructions(obs, width=size, length=size, height=size, density=density, drawConnections=False, draw=draw)
			
				#print "Nodes: {0}".format(len(g.locations))
				#start = random.choice([k for k in g.locations])
				start = 0
				goal = random.choice(range(10, len(g.locations) - 1))
				problem = GraphProblem(start, goal, g)

				## =============================ASTAR==========================

							# RUN
				t1 = time.time()
				search = astar_search(problem);
				t2 = time.time()
			
				# Order: run#, number of nodes, size, number of obstructions, time, path length, nodes expanded
				print "A,%d,%d,%d,%d,%f,%d,%d" % (run, len(g.locations), size, obstructions, (100*(t2-t1)), len(search.path()), len(problem.expanded))
			
				# Draw the resulting path
				if draw:
					lastNode = Node(goal)
					for node in search.path():
				
						if node.state == start:
							sphere(pos=g.objects[g.reverseLookup[node.state]].pos, radius = 5, color=color.green)
						elif node.state == goal:
							sphere(pos=g.objects[g.reverseLookup[node.state]].pos, radius = 5, color=color.yellow)
						else:
							sphere(pos=g.objects[g.reverseLookup[node.state]].pos, radius = 5, color=color.red)
					
						theAxis = g.objects[g.reverseLookup[lastNode.state]].pos - g.objects[g.reverseLookup[node.state]].pos
						arrow(pos=g.objects[g.reverseLookup[node.state]].pos, axis=theAxis, shaftwidth=2, fixedwidth=true, color=color.green)
					
						lastNode = node

				## =============================RRT=============================
							
				MAX_TREE_NODES = 1000
				QGOAL_BIAS = 0.33
				my_car = Car()
				space = ConfigSpace(g)

				Qinit_3d = (0, 0, 0)
					# x = g.objects[g.reverseLookup[goal]].pos.x
					# y = g.objects[g.reverseLookup[goal]].pos.y
					# z = g.objects[g.reverseLookup[goal]].pos.z
							# Qgoal_3d = (x, y, z)
				Qgoal_3d =  g.objects[g.reverseLookup[goal]].pos
							#print 'GOAL',Qgoal_3d

							# RUN
				t1 = time.time()
				rrt = RRT(space, my_car, Qinit_3d, Qgoal_3d, QGOAL_BIAS, MAX_TREE_NODES)
				tree = rrt.build_rrt_3d()
				t2 = time.time()

				if rrt.path:
						# Order: run#, number of nodes, size, number of obstructions, time, path length, nodes expanded
                                        print "R,%d,%d,%d,%d,%f,%d,%d" % (run, len(g.locations), size, obstructions, (100*(t2-t1)), len(rrt.path), len(rrt.attempts))
				else:
					print "R,%d,%d,%d,%d,%f,%d,%d" % (run, len(g.locations), size, obstructions, (100*(t2-t1)), -1, len(rrt.attempts))

				# Draw the resulting path
				if draw and rrt.path:

					# draw expanded
					for value in rrt.attempts:
						curr = (value[0], value[1], value[2])
						sphere(pos=curr, radius=11, color=color.orange)

					prev = (0,0,0)
					# draw path
					for value in rrt.path:
						curr = (value[0], value[1], value[2])
						sphere(pos=curr, radius=12, color=color.magenta)
						theaxis = (curr[0]-prev[0],curr[1]-prev[1],curr[2]-prev[2])
						arrow(pos=prev, axis=theaxis, shaftwidth=3, fixedwidth=true, color=color.cyan)
						prev = curr
									

                                # end
                                if draw:
                                        #i = raw_input("press enter to die \n")
                                        exit(0)

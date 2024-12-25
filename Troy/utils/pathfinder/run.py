from solver.jpsplus import JPSPlus             #define the solver
from solver.pruning.bbox import BBoxPruning    #define pruning
from utils.distance import diagonalDistance    #define h function
from solver.base import findPathBase           #define search function
from graph.node import Node                    #define Node for start/finish
from graph.grid import GridMap                 #define occupancy grid map via string
from evaluation.test import simpleTest         #define eval function

x_start, y_start = 0, 0
x_finish, y_finish = 14, 15
height = 15
width = 30
mapstr = '''

. . . . . . . . . . . . . . . . . . . . . # # . . . . . . .
. . . . . . . . . . . . . . . . . . . . . # # . . . . . . .
. . . . . . . . . . . . . . . . . . . . . # # . . . . . . .
. . . # # . . . . . . . . . . . . . . . . # # . . . . . . .
. . . # # . . . . . . . . # # . . . . . . # # . . . . . . .
. . . # # . . . . . . . . # # . . . . . . # # # # # . . . .
. . . # # . . . . . . . . # # . . . . . . # # # # # . . . .
. . . # # . . . . . . . . # # . . . . . . . . . . . . . . .
. . . # # . . . . . . . . # # . . . . . . . . . . . . . . .
. . . # # . . . . . . . . # # . . . . . . . . . . . . . . .
. . . # # . . . . . . . . # # . . . . . . . . . . . . . . .
. . . # # . . . . . . . . # # . . . . . . . . . . . . . . .
. . . . . . . . . . . . . # # . . . . . . . . . . . . . . .
. . . . . . . . . . . . . # # . . . . . . . . . . . . . . .
. . . . . . . . . . . . . # # . . . . . . . . . . . . . . .
'''

startNode  = Node(x_start,  y_start)           #define start Node
finishNode = Node(x_finish, y_finish)          #define finish Node
grid       = GridMap()                         #define grid Map
grid.readFromString(mapstr, width, height)     #see additionals in main.ipynb

#routine run - always call solver.doPreprocess before eval

prune = BBoxPruning()
solver = JPSPlus(diagonalDistance, prune)
solver.doPreprocess(grid)
simpleTest(solver, findPathBase, grid, startNode, finishNode, visualise=True)

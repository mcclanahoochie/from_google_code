import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib import pyplot
from matplotlib.cm import gray
import Image

class Drawer(object):
    """This class is responsible to handle drawing operations"""
    def __init__(self, config_space):
        fig = pyplot.figure()
        self.axes = fig.add_subplot(111)
        im = config_space.im.transpose(Image.FLIP_TOP_BOTTOM)
        self.axes.imshow(im, origin="lower", cmap=gray)
        self.w, self.h = config_space.im.size

    def draw_vertices(self, vertices, *args, **kwargs):
        if None in vertices:
            vertices.remove(None)
        x_val = [vertex[0] for vertex in vertices]
        y_val = [vertex[1] for vertex in vertices]
        self.axes.plot(x_val, y_val, *args, **kwargs)

    def draw_points(self, edges, *args, **kwargs):
        Path = mpath.Path
        data = []
        for edge in edges:
            if not edge:
                continue
            data.append((Path.MOVETO, edge[0]))
            for point in edge[1:]:
                data.append((Path.LINETO, point))
        codes, verts = zip(*data)
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path, **kwargs)
        self.axes.add_patch(patch)

    def draw_edges(self, edges, *args, **kwargs):
        Path = mpath.Path
        data = []
        for edge in edges:
            data.append((Path.MOVETO, (edge[0][0], self.h - edge[0][1])))
            data.append((Path.LINETO, (edge[1][0], self.h - edge[1][1])))
        codes, verts = zip(*data)
        path = mpath.Path(verts, codes)
        patch = mpatches.PathPatch(path, **kwargs)
        self.axes.add_patch(patch)

    def show(self):
        pyplot.show()


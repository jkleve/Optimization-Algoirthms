import matplotlib.pyplot as plt # plotting

import sys # check which version of python is runnint

# check if running with python3 or python2
PY3 = sys.version_info[0] == 3

class PlotUtils:

    def __init__(self, num_dims, bounds):

        # you can only plot up to 2 dimensions
        if num_dims > 2:
            print("Can not plot more than 2 dimensions")
            raise ValueError

        # needed to update plot on the fly
        plt.ion()

        self.num_dims = num_dims
        self.bounds = bounds

        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'ro')
        self.ax.grid()

        xlim_l = bounds[0][0]
        xlim_u = bounds[0][1]
        ylim_l = bounds[1][0]
        ylim_u = bounds[1][1]
        self.ax.set_xlim(xlim_l, xlim_u)
        self.ax.set_ylim(ylim_l, ylim_u)

    def plot(self, points):
        x1 = [point[0] for point in points]
        x2 = [point[1] for point in points]
        self.line.set_xdata(x1)
        self.line.set_ydata(x2)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__ == "__main__":

    data = [(0,0),(1,1),(2,2),(-3,-3)]
    pu = PlotUtils(2, [(-10,10),(-10,10)])
    pu.plot(data)

if PY3:
    input("Waiting for Enter to be pressed ...")
else:
    raw_input("Waiting for Enter to be pressed ...")

import matplotlib.pyplot as plt # plotting
import matplotlib.mlab as mlab
import numpy as np

class PlotUtils:

    def __init__(self, num_dims, bounds, func):

        # you can only plot up to 2 dimensions
        if num_dims > 2:
            print("Can not plot more than 2 dimensions")
            raise ValueError

        if num_dims != 2:
            raise ValueError("Feel free to implement PlotUtils for 1 dimension")

        # needed to update plot on the fly
        plt.ion()

        self.num_dims = num_dims
        self.bounds = bounds

        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'ro')
        self.ax.grid()


        delta1 = float(bounds[0][1] - bounds[0][0]) / 100
        delta2 = float(bounds[1][1] - bounds[1][0]) / 100
        x1 = np.arange(bounds[0][0], bounds[0][1], delta1)
        x2 = np.arange(bounds[1][0], bounds[1][1], delta2)
        X, Y = np.meshgrid(x1, x2)
        Z = func([X, Y])

        # Create a simple contour plot with labels using default colors.  The
        # inline argument to clabel will control whether the labels are draw
        # over the line segments of the contour, removing the lines beneath
        # the label
        #plt.figure()
        CS = plt.contour(X, Y, Z)
        plt.clabel(CS, inline=1, fontsize=10)
        plt.title('Simplest default with labels')

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
    import sys # check which version of python is runnint
    # check if running with python3 or python2
    PY3 = sys.version_info[0] == 3

    data = [(0,0),(1,1),(2,2),(-3,-3)]
    pu = PlotUtils(2, [(-10,10),(-10,10)])
    pu.plot(data)

    if PY3:
        input("Waiting for Enter to be pressed ...")
    else:
        raw_input("Waiting for Enter to be pressed ...")

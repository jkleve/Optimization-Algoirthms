import sys # need sys for the next line
sys.path.append("../utils") # add the utils dir to the python path
# then we can import PlotUtils
from plot_utils import PlotUtils

# you could import settings from a separate file like so
from settings import settings

# plot util variable. probably make this an instance
# variable in a class
plotutils = None

if settings['plot']:
    try:
        # Create PlotUtils instance
        # 2 params. number of dimensions and an array of 2D lists with
        # the bounds for each dimension. ex [(-10,10), (-10,10)]
        plotutils = PlotUtils(settings['num_dims'], settings['bounds'])
    except ValueError:
        print("Can not plot more than 2 dimensions!")
        # set this to false so that we don't try to use
        # the plotutils variable later on
        settings['plot'] = False

# data should be an array of 2D lists with x1 and x2 data
data = [(1,1),(8,4),(-4,-9)]

# open or update plot
if settings['plot']:
    plotutils.plot(data)

# you can put plotutils.plot(data) in a loop and continually update
# the plot. Here I just wait for you to press enter, after which the
# plot will close
raw_input("Press Enter to Exit") # Python 2 user input

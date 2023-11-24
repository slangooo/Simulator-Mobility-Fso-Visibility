import visibility_polygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from obstacles import Obstacles, get_madrid_buildings

if __name__ == '__main__':
    madrid_env = get_madrid_buildings()
    bds = madrid_env.get_margin_boundary()
    segments = madrid_env.get_total_segments()
    # vis = visibility_polygon.test_visibility_default(segments, [-10.5, -10.5])
    vis = visibility_polygon.get_visibility(segments, [-10.5, -10.5])

    bldgs_rects = madrid_env.plot_obstacles(fill_color='gray')
    _, ax = plt.subplots()
    ax.add_collection(PatchCollection(bldgs_rects))
    ax.add_collection(PatchCollection([Polygon(np.array(vis), closed=True)], fc='g', ec='r'))
    ax.set_xlim(bds[0][0], bds[2][0])
    ax.set_ylim(bds[0][1], bds[2][1])
    # ax.set_xlim(-10, 20)
    # ax.set_ylim(-10, 20)
    _.show()
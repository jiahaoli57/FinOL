import pandas as pd
import numpy as np

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from finol.config import *
from finol.utils import get_variable_name


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def plot_radar_chart(final_profit_result, final_risk_result, column_names, logdir):
    final_profit_result = final_profit_result[["Metric"] + column_names].set_index("Metric")
    final_risk_result = final_risk_result[["Metric"] + column_names].set_index("Metric")
    df = pd.concat([final_profit_result, final_risk_result], ignore_index=False)
    num_columns = len(df.columns)

    data = []
    for strategy in PLOT_ALL_1:
        _ = [df[strategy]["CW"], -df[strategy]["MDD"], -df[strategy]["VR"], df[strategy]["SR"], df[strategy]["APY"]]
        data += [_]

    theta = radar_factory(num_vars=5, frame='polygon')
    labels = ['CW', '- MDD', '- VR', 'SR', 'APY']

    fig, ax = plt.subplots(subplot_kw=dict(projection='radar'))
    colors = ['black'] * (num_columns - 1) + ['red'] * 1

    # plot the four cases from the example data on separate axes
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])

    # scale the data to [0, 1]
    data_array = np.array(data)
    min_val = np.min(data_array, axis=0)
    max_val = np.max(data_array, axis=0)
    scaled_data = (data_array - min_val) / (max_val - min_val)
    case_data = scaled_data

    for d, color, marker in zip(case_data, colors, MARKERS):
        ax.plot(theta, d, color=color, alpha=0.5, marker=marker)
        ax.fill(theta, d, facecolor=color, alpha=0.15, label='_nolegend_')
    ax.set_varlabels(labels)

    # add legend relative to top-left plot
    plt.title(DATASET_NAME)
    ax.legend(df.columns, loc=(0.7, 0.75))
    plt.tight_layout()
    plt.savefig(logdir + '/' + MODEL_NAME + '_' + DATASET_NAME + '_' + get_variable_name(column_names) + '_RADAR.pdf',
                format='pdf',
                dpi=300,
                bbox_inches='tight')
    plt.show()

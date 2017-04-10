import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.colors import (Normalize, LinearSegmentedColormap)
from mpl_toolkits.basemap import Basemap
from shapely.geometry import shape
from shapely.ops import transform
from descartes import PolygonPatch
from pysal.esda.mapclassify import (Natural_Breaks, User_Defined)


EPSG_4283_APPROXIMATION = {
    'projection': 'merc',
    'ellps': 'WGS84'
}

POLYGON_ALPHA = 0.5
LIGHT_GRAY = '#bcbcbc'
GRAY = '#555555'

PLOT_DPI = 500
FIG_SIZE = (10, 10)

VALUE_BREAKES = {
    'House': [x * 10 ** 6 for x in (0.5, 0.6, 0.7, 0.8, 1, 1.5, 2)],
    'Unit': [x * 10 ** 6 for x in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)],
}

COLOURS = {
    'House': 'Blues',
    'Unit': 'Greens',
}


class Choroplether():
    def save_map(geo_df, bbox, outputs_dir,
                 suburb_values, img_title, prop_type):
        plt.clf()

        fig, ax = Choroplether.prep_plt()
        basemap = Choroplether.make_a_basemap(bbox['ll_cnr'], bbox['ru_cnr'])
        Choroplether.make_map(
            geo_df, bbox, suburb_values, img_title, prop_type, fig, ax, basemap
        )

        figure_file_name = outputs_dir + 'map-%s.png' % img_title
        print('Saving: %s' % figure_file_name)
        plt.savefig(figure_file_name, dpi=PLOT_DPI, alpha=True)

    def make_map(geo_df, bbox, suburb_values, img_title, prop_type,
                 fig, ax, basemap):
        geo_df = Choroplether.add_poly_patches(geo_df, basemap)
        geo_df['estimated_value'] = Choroplether.prep_estimated_values(
            geo_df, suburb_values
        )
        breaks = Choroplether.make_breaks(geo_df, prop_type)
        geo_df = Choroplether.add_jenkins_bins(geo_df, breaks)
        cmap = plt.get_cmap(COLOURS[prop_type])

        trans_ll_cnr = basemap(*bbox['ll_cnr'])
        trans_ru_cnr = basemap(*bbox['ru_cnr'])

        ax = Choroplether.add_patch_collections_to_ax(geo_df, ax, cmap)
        Choroplether.add_a_colour_bar(breaks, geo_df, cmap, ax)

        ax.set_xlim([trans_ll_cnr[0], trans_ru_cnr[0]])
        ax.set_ylim([trans_ll_cnr[1], trans_ru_cnr[1]])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        plt.tight_layout()
        fig.suptitle(img_title)

    def add_patch_collections_to_ax(geo_df, ax, cmap):
        norm = Normalize()
        cmap_values = cmap(norm(geo_df['jenks_bins'].values))
        cmap_values[:, 3] = POLYGON_ALPHA

        for a in ('D', 'G'):
            Choroplether.add_sub_collection(
                ax, geo_df, geo_df['act_loca_5'] == a, cmap_values
            )
        return ax

    def prep_plt():
        fig = plt.figure(figsize=FIG_SIZE)
        ax = plt.subplot('111', axisbg='w', frame_on=True)
        ax.set_aspect('equal')
        return fig, ax

    def add_poly_patches(geo_df, basemap):
        geo_df['shape'] = geo_df['geometry'].map(
            lambda x: transform(basemap, shape(x))
        )
        geo_df['patches'] = Choroplether.make_polygon_patches(geo_df['shape'])
        return geo_df

    def add_jenkins_bins(geo_df, breaks):
        jb = pd.DataFrame(
            {'jenks_bins': breaks.yb},
            index=geo_df[geo_df['estimated_value'].notnull()].index
        )
        geo_df = geo_df.join(jb)
        geo_df['jenks_bins'].fillna(-1, inplace=True)
        return geo_df

    def prep_estimated_values(df, suburb_values):
        suburbs_difference = np.setdiff1d(
            df['name'].values, suburb_values.index.values
        )
        for name in suburbs_difference:
            suburb_values[name] = np.nan

        return [suburb_values[s] for s in df['name']]

    def make_breaks(df, prop_type):
        return User_Defined(
            df[df['estimated_value'].notnull()]['estimated_value'].values,
            VALUE_BREAKES[prop_type]
        )

    def make_polygon_patches(shapes):
        return shapes.map(
            lambda x: PolygonPatch(
                x, fc=LIGHT_GRAY, ec=GRAY, lw=0.4  # , alpha=POLYGON_ALPHA
            )
        )

    def add_a_colour_bar(breaks, df, cmap, colourbar_ax):
        break_ranges_and_counts = zip(
            np.concatenate([[0], breaks.bins[0:-1]]),
            breaks.bins, breaks.counts
        )
        jenks_labels = [
            '$ {:,} - {:,}\n{} suburb(s)'.format(int(a), int(b), c)
            for a, b, c in break_ranges_and_counts
        ]
        jenks_labels.insert(
            0, 'Null\n%s suburb(s)' % len(df[df['estimated_value'].isnull()])
        )
        colour_bar = Choroplether.colorbar_index(
            len(jenks_labels), cmap, jenks_labels, colourbar_ax
        )
        colour_bar.ax.tick_params(labelsize=6)

    def calc_aspect_ratio(ll, ru):
        r = ((ru[1] - ll[1]) / (ru[0] - ll[0]))
        return r

    def make_a_basemap(ll_cnr, ru_cnr):
        return Basemap(
            projection=EPSG_4283_APPROXIMATION['projection'],
            ellps=EPSG_4283_APPROXIMATION['ellps'],
            lon_0=(ru_cnr[0] - ll_cnr[0]) / 2,
            lat_0=(ru_cnr[1] - ll_cnr[1]) / 2,
            llcrnrlon=ll_cnr[0],
            llcrnrlat=ll_cnr[1],
            urcrnrlon=ru_cnr[0],
            urcrnrlat=ru_cnr[1],
            lat_ts=0,
            resolution='i',
            suppress_ticks=True
        )

    def add_sub_collection(ax, df, df_filter, cmap_values):
        sub_df = df[df_filter]
        pc = PatchCollection(sub_df['patches'], match_original=True)
        pc.set_facecolor(cmap_values[df_filter.values, :])
        ax.add_collection(pc)

    def colorbar_index(ncolors, cmap, labels, colourbar_ax):
        """
        Source:
        http://sensitivecities.com/so-youd-like-to-make-a-map-using-python-EN.html
        """
        cmap = Choroplether.cmap_discretize(cmap, ncolors)
        mappable = cm.ScalarMappable(cmap=cmap)
        mappable.set_array([])
        mappable.set_clim(-0.5, ncolors+0.5)
        color_bar = plt.colorbar(
            mappable,
            ax=colourbar_ax, drawedges=True,
            shrink=0.5, pad=0  # , alpha=POLYGON_ALPHA
        )

        color_bar.set_ticks(np.linspace(0, ncolors, ncolors))
        color_bar.set_ticklabels(range(ncolors))
        color_bar.set_ticklabels(labels)
        color_bar.outline.set_edgecolor(GRAY)
        color_bar.outline.set_linewidth(1)

        # color_bar.patch.set_facecolor((0, 0, 0, 1.0))
        # color_bar.set_alpha(POLYGON_ALPHA)
        return color_bar

    def cmap_discretize(cmap, N):
        """
        Source:
        http://sensitivecities.com/so-youd-like-to-make-a-map-using-python-EN.html
        """
        if type(cmap) == str:
            cmap = get_cmap(cmap)
        colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
        rgba = cmap(colors_i)
        indices = np.linspace(0, 1., N + 1)
        cdict = {}
        for ki, key in enumerate(('red', 'green', 'blue')):
            cdict[key] = [
                (
                    indices[i],
                    rgba[i - 1, ki] * POLYGON_ALPHA + (1 - POLYGON_ALPHA),
                    rgba[i, ki] * POLYGON_ALPHA + (1 - POLYGON_ALPHA)
                )
                for i in range(N + 1)
            ]

        # cdict['alpha'] = [
        #     (indices[i], POLYGON_ALPHA, POLYGON_ALPHA) for i in range(N + 1)
        # ]
        return LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

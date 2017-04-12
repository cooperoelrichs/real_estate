import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.colors import (Normalize, LinearSegmentedColormap)
from mpl_toolkits.basemap import Basemap
from shapely.geometry import shape
from shapely.ops import transform
from descartes import PolygonPatch
from pysal.esda.mapclassify import (Natural_Breaks, User_Defined)


class Choroplether():
    EPSG_4283_APPROXIMATION = {
        'projection': 'merc',
        'ellps': 'WGS84'
    }

    POLYGON_ALPHA = 1  # 0.5
    LIGHT_GREY = '#bcbcbc'
    GREY = '#555555'
    LINE_WIDTH = 0.4
    PLOT_DPI = 500
    NA_JENKINS_BIN = -1

    HOUSE_BREAKES = [x * 10 ** 6 for x in (0.5, 0.6, 0.7, 0.8, 1, 1.5, 2, 3)]
    UNIT_BREAKES = [x * 10 ** 6 for x in (0.2, 0.4, 0.4, 0.5, 0.6, 0.8, 1, 1.5)]
    BASIC_BREAKES = [x * 10 ** 6 for x in (0.1, 0.3, 0.5, 0.75, 1, 2)]

    VALUE_BREAKES = {
        'House': HOUSE_BREAKES,
        'Unit': UNIT_BREAKES,

        'Town House': UNIT_BREAKES,
        'Rural': HOUSE_BREAKES,
        'Duplex': HOUSE_BREAKES,
        'Retirement Living': BASIC_BREAKES,
        'Studio': UNIT_BREAKES,
        'Land': BASIC_BREAKES,
        'Not Specified': BASIC_BREAKES,
    }

    COLOURS = {
        'House': 'Blues',
        'Unit': 'Greens',

        'Town House': 'Greens',
        'Rural': 'Blues',
        'Duplex': 'Blues',
        'Retirement Living': 'Reds',
        'Studio': 'Greens',
        'Land': 'Reds',
        'Not Specified': 'Reds',
    }

    def make_save_map(plot_thing, eo_df, bbox, outputs_dir,
                 suburb_values, img_title, prop_type, fig_size):
        fig, ax = Choroplether.prep_plt(plot_thing, fig_size)
        basemap = Choroplether.make_a_basemap(bbox['ll_cnr'], bbox['ru_cnr'])
        Choroplether.make_map(
            plot_thing, geo_df, bbox, suburb_values, img_title, prop_type,
            fig, ax, basemap
        )
        Choroplether.save_map(plot_thing, outputs_dir, img_title)

    def save_map(plot_thing, outputs_dir, img_title):
        figure_file_name = outputs_dir + 'map-%s.png' % img_title
        print('Saving: %s' % figure_file_name)
        plot_thing.savefig(
            figure_file_name, dpi=Choroplether.PLOT_DPI, alpha=True
        )

    def make_map(plot_thing, geo_df, bbox, suburb_values, img_title, prop_type,
                 fig, ax, basemap):
        Choroplether.plot_polygons(
            plot_thing, geo_df, suburb_values, prop_type, ax, basemap
        )
        Choroplether.format_plot(plot_thing, bbox, ax, basemap)
        return plot_thing, fig, ax

    def plot_polygons(plot_thing, geo_df, suburb_values, prop_type, ax, basemap):
        geo_df = Choroplether.add_poly_patches(geo_df, basemap)
        geo_df['estimated_value'] = Choroplether.prep_estimated_values(
            geo_df, suburb_values
        )
        breaks = Choroplether.make_breaks(geo_df, prop_type)
        geo_df = Choroplether.add_jenkins_bins(geo_df, breaks)
        cmap = plot_thing.get_cmap(Choroplether.COLOURS[prop_type])

        poly_colours = Choroplether.generate_colours(geo_df, cmap, breaks)
        ax = Choroplether.add_patch_collections_to_ax(geo_df, ax, poly_colours)
        Choroplether.add_a_colour_bar(plot_thing, breaks, geo_df, cmap, ax)

    def format_plot(plot_thing, bbox, ax, basemap):
        trans_ll_cnr = basemap(*bbox['ll_cnr'])
        trans_ru_cnr = basemap(*bbox['ru_cnr'])
        ax.set_xlim([trans_ll_cnr[0], trans_ru_cnr[0]])
        ax.set_ylim([trans_ll_cnr[1], trans_ru_cnr[1]])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plot_thing.tight_layout()

    def generate_colours(geo_df, cmap, breaks):
        norm = Normalize(vmin=Choroplether.NA_JENKINS_BIN, vmax=(breaks.k - 1))
        cmap_values = cmap(norm(geo_df['jenks_bins'].values))
        cmap_values[:, 3] = Choroplether.POLYGON_ALPHA
        return cmap_values

    def add_suptitle(fig, title):
        fig.suptitle(title, y=1)

    def add_patch_collections_to_ax(geo_df, ax, poly_colours):
        for a in ('G'):  # ('D', 'G')
            Choroplether.add_sub_collection(
                ax, geo_df, geo_df['act_loca_5'] == a, poly_colours
            )
        return ax

    def add_sub_collection(ax, df, df_filter, cmap_values):
        sub_df = df[df_filter]
        pc = PatchCollection(sub_df['patches'], match_original=True)
        pc.set_facecolor(cmap_values[df_filter.values, :])
        ax.add_collection(pc)

    def prep_plt(plot_thing, fig_size):
        fig = plot_thing.figure(figsize=fig_size)
        ax = plot_thing.subplot('111', axisbg='w', frame_on=True)
        ax.set_aspect('equal')
        return fig, ax

    def add_poly_patches(geo_df, basemap):
        geo_df['shape'] = Choroplether.make_shapes(geo_df, basemap)
        geo_df['patches'] = Choroplether.make_default_polygon_patches(
            geo_df['shape']
        )
        return geo_df

    def make_shapes(geo_df, bmap):
        return geo_df['geometry'].map(lambda x: transform(bmap, shape(x)))

    def add_jenkins_bins(geo_df, breaks):
        jb = pd.DataFrame(
            {'jenks_bins': breaks.yb},
            index=geo_df[geo_df['estimated_value'].notnull()].index
        )
        geo_df = geo_df.join(jb)
        geo_df['jenks_bins'].fillna(Choroplether.NA_JENKINS_BIN, inplace=True)
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
            Choroplether.VALUE_BREAKES[prop_type]
            # k=7
        )

    def make_default_polygon_patches(shapes):
        return Choroplether.make_polygon_patches(
            shapes,
            fc=Choroplether.LIGHT_GREY,
            ec=Choroplether.GREY,
            lw=Choroplether.LINE_WIDTH,
            alpha=Choroplether.POLYGON_ALPHA
        )

    def make_polygon_patches(shapes, fc, ec, lw, alpha):
        return shapes.map(
            lambda x: PolygonPatch(x, fc=fc, ec=ec, lw=lw, alpha=alpha)
        )

    def add_a_colour_bar(plot_thing, breaks, df, cmap, colourbar_ax):
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
            plot_thing, len(jenks_labels), cmap, jenks_labels, colourbar_ax
        )
        colour_bar.ax.tick_params(labelsize=5)

    def calc_aspect_ratio(ll, ru):
        r = ((ru[1] - ll[1]) / (ru[0] - ll[0]))
        return r

    def make_a_basemap(ax, ll_cnr, ru_cnr):
        bm = Basemap(
            projection=Choroplether.EPSG_4283_APPROXIMATION['projection'],
            ellps=Choroplether.EPSG_4283_APPROXIMATION['ellps'],
            lon_0=(ru_cnr[0] - ll_cnr[0]) / 2,
            lat_0=(ru_cnr[1] - ll_cnr[1]) / 2,
            llcrnrlon=ll_cnr[0],
            llcrnrlat=ll_cnr[1],
            urcrnrlon=ru_cnr[0],
            urcrnrlat=ru_cnr[1],
            lat_ts=0,
            resolution='i',
            suppress_ticks=True,
            ax=ax
        )
        return bm

    def colorbar_index(plot_thing, ncolors, cmap, labels, colourbar_ax):
        """
        Source:
        http://sensitivecities.com/so-youd-like-to-make-a-map-using-python-EN.html
        """
        cmap = Choroplether.cmap_discretize(cmap, ncolors)
        mappable = cm.ScalarMappable(cmap=cmap)
        mappable.set_array([])
        mappable.set_clim(-0.5, ncolors+0.5)
        color_bar = plot_thing.colorbar(
            mappable,
            ax=colourbar_ax,
            drawedges=True,
            shrink=0.5,
            aspect=40,
            pad=0
        )

        color_bar.set_ticks(np.linspace(0, ncolors, ncolors))
        color_bar.set_ticklabels(range(ncolors))
        color_bar.set_ticklabels(labels)
        color_bar.outline.set_edgecolor(Choroplether.GREY)
        color_bar.outline.set_linewidth(Choroplether.LINE_WIDTH)
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
                    (rgba[i - 1, ki] * Choroplether.POLYGON_ALPHA +
                     (1 - Choroplether.POLYGON_ALPHA)),
                    (rgba[i, ki] * Choroplether.POLYGON_ALPHA +
                     (1 - Choroplether.POLYGON_ALPHA))
                )
                for i in range(N + 1)
            ]

        # cdict['alpha'] = [
        #     (indices[i],
        #      Choroplether.POLYGON_ALPHA,
        #      Choroplether.POLYGON_ALPHA
        #     ) for i in range(N + 1)
        # ]
        return LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)

    def label_polygons(geo_df_with_shapes, df_filter, ax, font_size):
        geo_df_with_shapes[df_filter].apply(
            lambda x: ax.annotate(
                s=x['name'],
                xy=x['shape'].centroid.coords[0],
                # xy=x['shape'].representative_point().coords[:],
                ha='center',
                # color='#555555',
                color='k',
                fontsize=font_size,
                # weight='bold'
            ),
            axis=1
        )
        return ax

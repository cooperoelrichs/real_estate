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

    DOLLAR_LABEL = '$ {:,.0f} - {:,.0f}\n{} suburb(s)'
    COLOUR_BAR_LABELS = {
        'sales': DOLLAR_LABEL,
        'rentals': DOLLAR_LABEL,
        'returns': '{:.3f} - {:.3f}\n{} suburb(s)',
    }

    SALES_HOUSE_BREAKES = [x * 10 ** 6 for x in (0.5, 0.6, 0.7, 0.8, 1, 1.5, 2, 3)]
    SALES_UNIT_BREAKES = [x * 10 ** 6 for x in (0.2, 0.4, 0.4, 0.5, 0.6, 0.8, 1, 1.5)]
    SALES_BASIC_BREAKES = [x * 10 ** 6 for x in (0.1, 0.3, 0.5, 0.75, 1, 2)]
    SALES_VALUE_BREAKES = {
        'House': SALES_HOUSE_BREAKES,
        'Unit': SALES_UNIT_BREAKES,

        'Town House': SALES_UNIT_BREAKES,
        'Rural': SALES_HOUSE_BREAKES,
        'Duplex': SALES_HOUSE_BREAKES,
        'Retirement Living': SALES_BASIC_BREAKES,
        'Studio': SALES_UNIT_BREAKES,
        'Land': SALES_BASIC_BREAKES,
        'Not Specified': SALES_BASIC_BREAKES,
    }

    RENTALS_HOUSE_BREAKES = [x * 10 ** 2 for x in (4.5, 5, 6, 7, 10, 12, 14, 6, 20, 50)]
    RENTALS_UNIT_BREAKES = [x * 10 ** 2 for x in (3, 4, 5, 6, 7, 10, 15)]
    RENTALS_BASIC_BREAKES = RENTALS_HOUSE_BREAKES
    RENTALS_VALUE_BREAKES = {
        'House': RENTALS_HOUSE_BREAKES,
        'Unit': RENTALS_UNIT_BREAKES,

        'Town House': RENTALS_UNIT_BREAKES,
        'Rural': RENTALS_HOUSE_BREAKES,
        'Duplex': RENTALS_HOUSE_BREAKES,
        'Retirement Living': RENTALS_BASIC_BREAKES,
        'Studio': RENTALS_UNIT_BREAKES,
        'Land': RENTALS_BASIC_BREAKES,
        'Not Specified': RENTALS_BASIC_BREAKES,
    }

    RETURNS_BREAKES = [0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.15]
    RETURNS_VALUE_BREAKES = {
        'House': RETURNS_BREAKES,
        'Unit': RETURNS_BREAKES,
    }

    BREAKES = {
        'sales': SALES_VALUE_BREAKES,
        'rentals': RENTALS_VALUE_BREAKES,
        'returns': RETURNS_VALUE_BREAKES
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
                 suburb_values, img_title, prop_type, fig_size, value_breakes,
                 model_type):
        fig, ax = Choroplether.prep_plt(plot_thing, fig_size)
        basemap = Choroplether.make_a_basemap(bbox['ll_cnr'], bbox['ru_cnr'])
        Choroplether.make_map(
            plot_thing, geo_df, bbox, suburb_values, img_title, prop_type,
            fig, ax, basemap, value_breakes, model_type
        )
        Choroplether.save_map(plot_thing, outputs_dir, img_title)

    def save_map(plot_thing, outputs_dir, img_title):
        figure_file_name = outputs_dir + 'map-%s.png' % img_title
        print('Saving: %s' % figure_file_name)
        plot_thing.savefig(
            figure_file_name, dpi=Choroplether.PLOT_DPI, alpha=True
        )

    def make_map(plot_thing, geo_df, bbox, suburb_values, img_title, prop_type,
                 fig, ax, basemap, value_breakes, labels_str):
        Choroplether.plot_polygons(
            plot_thing, geo_df, suburb_values, prop_type, ax, basemap,
            value_breakes, labels_str
        )
        Choroplether.format_plot(plot_thing, bbox, ax, basemap)
        return plot_thing, fig, ax

    def plot_polygons(plot_thing, geo_df, suburb_values, prop_type,
            ax, basemap, value_breakes, labels_str):
        geo_df = Choroplether.add_poly_patches(geo_df, basemap)
        geo_df['estimated_value'] = Choroplether.prep_estimated_values(
            geo_df, suburb_values
        )
        breaks = Choroplether.make_breaks(geo_df, prop_type, value_breakes)
        geo_df = Choroplether.add_jenkins_bins(geo_df, breaks)
        cmap = plot_thing.get_cmap(Choroplether.COLOURS[prop_type])

        poly_colours = Choroplether.generate_colours(geo_df, cmap, breaks)
        ax = Choroplether.add_patch_collections_to_ax(geo_df, ax, poly_colours)
        Choroplether.add_a_colour_bar(plot_thing, breaks, geo_df, cmap, ax, labels_str)

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
        suburb_values = Choroplether.add_missing_data(
            df['name'], suburb_values, np.nan
        )
        return [suburb_values[s] for s in df['name']]

    def add_missing_data(index, data, missing_value):
        difference = np.setdiff1d(
            index.values, data.index.values
        )
        for a in difference:
            data[a] = missing_value
        return data

    def make_breaks(df, prop_type, value_breakes):
        return User_Defined(
            df[df['estimated_value'].notnull()]['estimated_value'].values,
            value_breakes[prop_type]
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

    def add_a_colour_bar(plot_thing, breaks, df, cmap, colourbar_ax, labels_str):
        break_ranges_and_counts = zip(
            np.concatenate([[0], breaks.bins[0:-1]]),
            breaks.bins, breaks.counts
        )
        jenks_labels = [
            labels_str.format(a, b, c)
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

    def label_polygons(df, df_filter, ax, font_size, field):
        Choroplether.annotate_polygons(
            df, df_filter, ax, font_size, lambda x: x[field]
        )

    def label_polygons_with_counts(df, df_filter, ax, font_size, counts):
        counts = Choroplether.add_missing_data(df['name'], counts, 0)
        df['count'] = df.apply(lambda x: counts[x['name']], axis=1)
        Choroplether.annotate_polygons(
            df, df_filter, ax, font_size,
            lambda x: '%s\n%0.0f' % (x['name'], x['count'])
        )

    def annotate_polygons(df, df_filter, ax, font_size, annotater):
        df[df_filter].apply(
            lambda x: ax.annotate(
                s=annotater(x),
                xy=x['shape'].centroid.coords[0],
                ha='center',
                va='center',
                color='k',
                fontsize=font_size
            ),
            axis=1
        )
        return ax

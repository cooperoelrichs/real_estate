import numpy as np

from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point
from shapely.ops import unary_union

from matplotlib.collections import PatchCollection
from real_estate.maps.choroplether import Choroplether


class Mapper():
    def plot_boundaries(plt, ax, geo_df, bmap, bbox):
        geo_df.geometry = Choroplether.make_shapes(geo_df, bmap)
        geo_df['patches'] = Choroplether.make_polygon_patches(
            geo_df.geometry,
            None,
            Choroplether.LIGHT_GREY,
            Choroplether.LINE_WIDTH * 2,
            0.5, False
        )

        pc = PatchCollection(
            geo_df[geo_df['act_loca_5'] == 'G']['patches'], match_original=True
        )
        ax.add_collection(pc)
        Choroplether.format_plot(plt, bbox, ax, bmap)

    def to_points(xs_and_ys):
        gdf = GeoDataFrame(
            {
                'X': xs_and_ys[0, :],
                'Y': xs_and_ys[1, :]
            }
        )
        gdf.geometry = gdf.apply(lambda p: Point(p['X'], p['Y']), axis=1)
        return gdf

    def intersecting_points(points, boundaries):
        print('Checking %i points for intersections' % len(points))
        intersects_filter = points.apply(
            lambda p: not boundaries.intersects(p.geometry).any(), axis=1
        )
        filtered_points = points[intersects_filter]
        return intersects_filter

import os
from itertools import product

import numpy as np

from real_estate.maps.choroplether import Choroplether
from real_estate.maps.mapper import Mapper
from real_estate.maps.basemapper import Basemapper


class LocationEffects():
    def gen_coord_grid(bbox, bmap, resolution):
        ll = LocationEffects.project(bmap, bbox['ll_cnr'])
        ru = LocationEffects.project(bmap, bbox['ru_cnr'])
        resolution = (
            resolution,
            int((ru[1]-ll[1]) / (ru[0]-ll[0]) * resolution)
        )
        xs = LocationEffects.interpolate(ll[0], ru[0], resolution[0])
        ys = LocationEffects.interpolate(ll[1], ru[1], resolution[1])
        X, Y = np.meshgrid(xs, ys)
        points = np.array((X.flatten(), Y.flatten()))
        return points, resolution

    def project(basemap, point):
        longitude, latitude = point
        return basemap(longitude, latitude)

    def interpolate(start, stop, resolution):
        return np.linspace(start, stop, num=resolution)

    def estimate_prices(X, Y, model, prop, resolution):
        data = LocationEffects.gen_test_data_set(X, Y, prop, resolution)
        return model.predict(data)

    def gen_test_data_set(X, Y, prop, resolution):
        data = np.empty([resolution[0]*resolution[1], 12])
        data[:, 0] = prop['bedrooms']
        data[:, 1] = prop['garage_spaces']
        data[:, 2] = prop['bedrooms']
        data[:, 3] = prop['property_type']
        data[:, 4] = X
        data[:, 5] = Y
        data[:, 6] = prop['bedrooms'] ** 2
        data[:, 7] = prop['bedrooms'] ** 3
        data[:, 8] = prop['garage_spaces'] ** 2
        data[:, 9] = prop['garage_spaces'] ** 3
        data[:, 10] = prop['bedrooms'] ** 2
        data[:, 11] = prop['bedrooms'] ** 3
        return data

    def color_map(
        name, property_type, model, prop,
        plt, bbox, bmap, boundaries,
        settings,
        limits, resolution
    ):
        bm_img = Basemapper.dl_tiles(bbox, 'carto-lite-no-labels', 13)
        Basemapper.add_img_to_basemap(bmap, bm_img)
        points, resolution = LocationEffects.gen_coord_grid(
            bbox, bmap, resolution
        )
        points = Mapper.to_points(points)
        prices = LocationEffects.estimate_prices(
            points['X'], points['Y'],
            model, prop, resolution
        )

        cmap_name = 'viridis'
        prices[prices < limits[0]] = limits[0]
        prices[prices > limits[1]] = limits[1]
        contour_plot = plt.contourf(
            points['X'].reshape((resolution[1], resolution[0])),
            points['Y'].reshape((resolution[1], resolution[0])),
            prices.reshape((resolution[1], resolution[0])),
            50,
            alpha=0.5,
            cmap=cmap_name,
            linewidths=None,
            antialiased=True,
        )

        delta = int(1e5)
        labels = list(
            "{:,}".format(int(x))
            for x in np.arange(limits[0]-delta, limits[1]+delta, delta)
        )
        colorbar = Choroplether.colorbar_index(
            plt, len(labels), plt.get_cmap(cmap_name), labels, None
        )
        colorbar.ax.tick_params(labelsize=8)
        Choroplether.save_map(
            plt, os.path.join(settings.outputs_dir, name),
            'location_effect_plot-%s' % property_type
        )
        return None

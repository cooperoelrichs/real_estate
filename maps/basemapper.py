from mpl_toolkits.basemap import Basemap
import providerless_geotiler


class Basemapper():
    DEFAULT_PROVIDER = 'carto-lite-no-labels'
    PROVIDERS = {
        'carto-lite-no-labels': {
            'name': 'Carto Light - No Labels',
            'attribution': "Map tiles by Carto, under CC BY 3.0.\nData by OpenStreetMap, under ODbL.",
            'url': "https://cartodb-basemaps-{subdomain}.global.ssl.fastly.net/light_nolabels/{z}/{x}/{y}.png",
            'subdomains': ['a', 'b', 'c']
        },
        'carto-lite': {
            'name': 'Carto Light',
            'attribution': "Map tiles by Carto, under CC BY 3.0.\nData by OpenStreetMap, under ODbL.",
            'url': "https://cartodb-basemaps-{subdomain}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
            'subdomains': ['a', 'b', 'c']
        }

    }

    def plot_default_on_map(ax, bmap, bbox):
        return Basemapper.plot_on_map(
            ax, bmap, bbox, Basemapper.DEFAULT_PROVIDER
        )

    def plot_on_map(ax, bmap, bbox, provider_name):
        img = Basemapper.dl_tiles(bbox, provider_name)
        bmap = Basemapper.add_img_to_basemap(bmap, img)
        Basemapper.add_attribution(
            ax, Basemapper.PROVIDERS[provider_name]['attribution']
        )
        return bmap

    def add_img_to_basemap(bmap, img, alpha=1):
        bmap.imshow(img, interpolation='lanczos', origin='upper', alpha=alpha)
        return bmap

    def add_attribution(ax, text):
        ax.text(
            0.99, 0.002,
            text,
            ha='right', va='bottom',
            size=4.5,
            color='#555555',
            transform=ax.transAxes
        )

    def dl_tiles(bbox, provider_name, zoom):
        bm_image = providerless_geotiler.MapPlus(
            provider_spec=Basemapper.PROVIDERS[provider_name],
            zoom=zoom,
            extent=(
                bbox['ll_cnr'][0], bbox['ll_cnr'][1],
                bbox['ru_cnr'][0], bbox['ru_cnr'][1]
            )
        )

        img = providerless_geotiler.render_map(bm_image)
        return img

from mpl_toolkits.basemap import Basemap
import providerless_geotiler


class Basemapper():
    DEFAULT_PROVIDER = 'carto-lite-no-labels'
    PROVIDERS = {
        'carto-lite-no-labels': {
            'name': 'Carto Light - No Labels',
            'attribution': "Map tiles by Carto, under CC BY 3.0. Data by OpenStreetMap, under ODbL.",
            'url': "https://cartodb-basemaps-{subdomain}.global.ssl.fastly.net/light_nolabels/{z}/{x}/{y}.png",
            'subdomains': ['a', 'b', 'c']
        },
        'carto-lite': {
            'name': 'Carto Light',
            'attribution': "Map tiles by Carto, under CC BY 3.0. Data by OpenStreetMap, under ODbL.",
            'url': "https://cartodb-basemaps-{subdomain}.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
            'subdomains': ['a', 'b', 'c']
        }

    }

    def plot_default_on_map(bmap, bbox):
        return Basemapper.plot_on_map(bmap, bbox, Basemapper.DEFAULT_PROVIDER)

    def plot_on_map(bmap, bbox, provider):
        img = Basemapper.dl_tiles(bbox, provider)
        bmap = Basemapper.add_img_to_basemap(bmap, img)
        return bmap


    def add_img_to_basemap(bmap, img):
        bmap.imshow(img, interpolation='lanczos', origin='upper')
        return bmap

    def dl_tiles(bbox, provider_name):
        bm_image = providerless_geotiler.MapPlus(
            provider_spec=Basemapper.PROVIDERS[provider_name],
            zoom=11,
            extent=(
                bbox['ll_cnr'][0], bbox['ll_cnr'][1],
                bbox['ru_cnr'][0], bbox['ru_cnr'][1]
            )
        )

        img = providerless_geotiler.render_map(bm_image)
        return img

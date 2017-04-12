from mpl_toolkits.basemap import Basemap
import geotiler


class Basemapper():
    DEFAULT_PROVIDER = 'carto-lite-no-labels'

    def plot_default_on_map(bmap, bbox):
        return Basemapper.plot_on_map(bmap, bbox, Basemapper.DEFAULT_PROVIDER)

    def plot_on_map(bmap, bbox, provider):
        img = Basemapper.dl_tiles(bbox, provider)
        bmap = Basemapper.add_img_to_basemap(bmap, img)
        return bmap


    def add_img_to_basemap(bmap, img):
        bmap.imshow(img, interpolation='lanczos', origin='upper')
        return bmap

    def dl_tiles(bbox, provider):
        bm_image = geotiler.Map(
            provider=provider,
            zoom=11,
            extent=(
                bbox['ll_cnr'][0], bbox['ll_cnr'][1],
                bbox['ru_cnr'][0], bbox['ru_cnr'][1]
            )
        )

        img = geotiler.render_map(bm_image)
        return img

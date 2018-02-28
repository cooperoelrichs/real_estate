import pandas as pd

from shapely.ops import transform
from shapely.geometry import shape


class Projector():
    def project_coordinates(df, basemap):
        projected = pd.DataFrame(
            columns=('X', 'Y'),
            data=list(df[['longitude', 'latitude']].apply(
                lambda r: basemap(r['longitude'], r['latitude']), axis=1
            ).values),
            index=df.index
        )

        df['X'] = projected['X']
        df['Y'] = projected['Y']
        return df

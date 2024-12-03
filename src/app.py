import gradio as gr
from gradio_folium import Folium
import folium
from folium.plugins import Draw
import random

# define some starting locations
locations = [[51.5360, -0.1196], [40.8093, -73.9678], [34.0069, -118.2304], [25.7617, -80.1918], [48.8488, 2.3470], [41.8840, 12.4790], [52.5078, 13.4003], [37.9964, 23.7293], [59.3265, 18.0680], [40.4172, -3.6867], [41.3940, 2.1606], [38.7253, -9.1403], [55.7476, 37.6209], [31.2230, 121.4860], [39.8958, 116.4000], [37.5467, 126.9901], [35.6777, 139.7685], [1.3294, 103.8081], [-33.8782, 151.1754], [30.0465, 31.2246], [6.5047, 3.3732], [-1.2873, 36.8198], [-33.9425, 18.5065],  [19.3916, -99.1279], [-23.5727, -46.6207], [36.1458, -115.1832], [38.8939, -77.0372], [49.2501, -123.1164], [43.6888, -79.4190]]

# initialise the map for drawing on
foliumMap = folium.Map(location=random.choice(locations))
draw = Draw(
    export=True,
    filename='region.geojson',
    position='topleft',
    draw_options={
        'polyline': False,
        'circlemarker': False,
        'polygon': {'allowIntersection': False},
        'marker': False,
        'circle': False,
        'rectangle': False,
    },
    edit_options={'poly': {'allowIntersection': False}}
)
draw.add_to(foliumMap)

# structure of the UI
with gr.Blocks() as demo:
    map = Folium(value=foliumMap)
    upload = gr.UploadButton(label='Upload the geojson file')

# launch the UI
demo.launch()
import gradio as gr
from gradio_folium import Folium
import folium
from folium.plugins import Draw

foliumMap = folium.Map(location=[25.7617, -80.1918])
draw = Draw(
    export=True,
    filename='data.geojson',
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

with gr.Blocks() as demo:
    map = Folium(value=foliumMap, height=800)
    upload = gr.UploadButton(label='Upload the geojson file')

if __name__ == "__main__":
    demo.launch()
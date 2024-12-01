import requests

# makes a request to overpass turbo to get osm data
def get_osm_data(object_type, bounding_coords):
    overpass_url = 'http://overpass-api.de/api/interpreter'
    key_type = ('building' if object_type == 'house' else 'natural' if object_type == 'tree' else 'gambling' if object_type == 'slot_machines' else 'amenity') # get the right key type
    while len(bounding_coords) >= 100000:
        print(f'truncating region with {len(bounding_coords)} vertices...') # log message
        bounding_coords = [bounding_coord for n, bounding_coord in enumerate(bounding_coords) if n % 2 == 0]
    bounding_coords = [(latitude, (longitude + 180) % 360 - 180) for latitude, longitude in bounding_coords]
    region_poly = ' '.join([' '.join([str(latlong) for latlong in coord]) for coord in bounding_coords]) # converts list of tuples to correct format for overpass turbo
    overpass_query = [
        '[out:json];',
        '(',
        f'   nwr["{key_type}"="{object_type}"](poly:"{region_poly}");',
        ');',
        'out center;'
    ]
    overpass_query = '\n'.join(overpass_query) # concatenates list items into a multi-line string
    
    response = requests.post(overpass_url, data=overpass_query)
    print(response) # log message, shows the HTTP response code
    data = response.json()
    
    coords = []
    for nwr in data['elements']:
        if nwr['type'] == 'node':
            coords.append((nwr['lat'], nwr['lon']))
        else: # it is a way or relation
            coords.append((nwr['center']['lat'], nwr['center']['lon']))
    
    return coords
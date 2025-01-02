import gradio as gr
from gradio_folium import Folium
import folium
from folium.plugins import Draw
import random
import json
from data_processing.get_osm_data import get_osm_data
from calc_factors import averageDistance, calcArea, density
import numpy as np
from neural_network import MultilayerPerceptron
import copy
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi
import bcrypt

# initialise neural network
network = MultilayerPerceptron([24, 18, 12, 6, 3, 1])
network.load_model('models/hdi.pkl')

# define global vars
currentRegion = []
all_objects = {}
currentHDI = -1
averageDistanceFactors = [['house', 'school'], ['house', 'hospital'], ['house', 'pharmacy'], ['house', 'restaurant'], ['school', 'hospital'], ['police', 'hospital'], ['house', 'place_of_worship'], ['bank', 'slot_machines'], ['fast_food', 'toilets'], ['house', 'police'], ['university', 'library'], ['house', 'library']]
densityFactors = ['school', 'hospital', 'pharmacy', 'police', 'library', 'toilets', 'restaurant', 'place_of_worship', 'post_box', 'vending_machine', 'bench', 'tree']

# connect to mongoDB
mongoURI = "mongodb+srv://lukerhodri:jQeBcC6XLyxPEwK8@predictinghdi.3nkqe.mongodb.net/?retryWrites=true&w=majority&appName=predictingHDI"
mongoClient = MongoClient(mongoURI, server_api=ServerApi('1'), tlsCAFile=certifi.where())
users = mongoClient['predictingHDI']['users']
# send a ping to confirm a successful connection
mongoClient.admin.command('ping')
print("Pinged your deployment. You successfully connected to MongoDB!")

# code to process the user uploading their geojson file
def uploadGeoJSON(filename):
    global currentRegion
    with open(filename, 'r') as file:
        data = json.load(file)
        currentRegion = [coordinate[::-1] for coordinate in data['features'][0]['geometry']['coordinates'][0]]
    return f'Successfully Uploaded {filename[filename.rindex("/")+1:]}'

# predict the HDI of the given region
def processPrediction(max_x_objects, max_y_objects, progress=gr.Progress()):
    global currentRegion
    global all_objects
    global currentHDI
    if currentRegion == []:
        return 'You have not uploaded a region!'
    progress(0, desc='Starting...')
    # retrieve all relevant osm data
    object_types = ['house', 'school', 'hospital', 'pharmacy', 'restaurant', 'place_of_worship', 'bank', 'slot_machines', 'fast_food', 'toilets', 'police', 'university', 'library', 'post_box', 'vending_machine', 'bench', 'tree']
    all_objects = {}
    for num, object_type in enumerate(object_types):
        progress(0.3*(num/len(object_types)), desc=f'Downloading {object_type} data...')
        print(f'getting {object_type} data...') # log message
        all_objects[object_type] = get_osm_data(object_type, currentRegion)
    # calculate average distances
    allFactors = []
    for num, averageDistanceFactor in enumerate(averageDistanceFactors):
        progress(0.3+0.6*(num/len(averageDistanceFactors)), desc=f'Calculating Average Distance between {averageDistanceFactor[0]} and {averageDistanceFactor[1]}...')
        print(f'finding A({averageDistanceFactor[0]}, {averageDistanceFactor[1]})...') # log message
        allFactors.append(averageDistance(all_objects[averageDistanceFactor[0]], all_objects[averageDistanceFactor[1]], max_x_objects, max_y_objects))
    # calculate area
    progress(0.9, desc='Calculating Area...')
    print('calculating area...') # log message
    area = calcArea(currentRegion)
    # calculate density factors
    for num, densityFactor in enumerate(densityFactors):
        progress(0.91+0.05*(num/len(densityFactors)), desc=f'Calculating {densityFactor} Density...')
        print(f'finding D({densityFactor})')
        allFactors.append(density(all_objects[densityFactor], area))
    # predict HDI
    progress(0.96, desc='Predicting HDI...')
    inputLayer = np.matrix([[100 if factor == None else float(factor)/10 if index <= 11 else float(factor)] for index, factor in enumerate(allFactors)])
    prediction = round(network.predict(inputLayer).item(0,0), 3)
    currentHDI = prediction
    return prediction

# helper function to calculate mean of coordinates
def calcMeanOfCoords(coords):
    return [np.mean([coord[0] for coord in coords]), np.mean([coord[1] for coord in coords])]

# calculate the optimal place for a new building to go when suggesting
def calcOptimalPlaceFor(building):
    global all_objects
    means = []
    for averageDistanceFactor in averageDistanceFactors:
        if averageDistanceFactor[1] == building:
            means.append(calcMeanOfCoords(all_objects[averageDistanceFactor[0]]))
    return calcMeanOfCoords(means)

# make suggestions from a prediction
def makeSuggestions(max_x_objects, max_y_objects, progress=gr.Progress()):
    global all_objects
    if currentHDI == -1:
        return [['You have not yet predicted an HDI!', None, None, None]]
    suggestions = ['school', 'hospital', 'place_of_worship', 'police', 'restaurant', 'slot_machines', 'library', 'pharmacy']
    returnTable = []
    for num, suggestion in enumerate(suggestions):
        progress(num/len(suggestions), desc=f'Suggesting building a {suggestion}...')
        # find optimal place
        progress((num/len(suggestions))+(0*(1/32)), desc=f'Suggesting building a {suggestion} (finding optimal position)...')
        position = calcOptimalPlaceFor(suggestion)
        # calculate average distances
        progress((num/len(suggestions))+(1*(1/32)), desc=f'Suggesting building a {suggestion} (calculating new average distances)...')
        allFactors = []
        for averageDistanceFactor in averageDistanceFactors:
            # ensure that we include the new building
            if suggestion == averageDistanceFactor[0]:
                allFactors.append(averageDistance(all_objects[averageDistanceFactor[0]], all_objects[averageDistanceFactor[1]], max_x_objects, max_y_objects, must_include_x=position))
            elif suggestion == averageDistanceFactor[1]:
                allFactors.append(averageDistance(all_objects[averageDistanceFactor[0]], all_objects[averageDistanceFactor[1]], max_x_objects, max_y_objects, must_include_y=position))
            else:
                allFactors.append(averageDistance(all_objects[averageDistanceFactor[0]], all_objects[averageDistanceFactor[1]], max_x_objects, max_y_objects))
        # calculate density factors
        progress((num/len(suggestions))+(2*(1/32)), desc=f'Suggesting building a {suggestion} (calculating new densities)...')
        area = calcArea(currentRegion)
        for densityFactor in densityFactors:
            # ensure that we include the new building
            if suggestion == densityFactor:
                allFactors.append(density(all_objects[densityFactor] + [position], area))
            else:
                allFactors.append(density(all_objects[densityFactor], area))
        # predict HDI
        progress((num/len(suggestions))+(3*(1/32)), desc=f'Suggesting building a {suggestion} (predicting new HDI)...')
        inputLayer = np.matrix([[100 if factor == None else float(factor)/10 if index <= 11 else float(factor)] for index, factor in enumerate(allFactors)])
        prediction = round(network.predict(inputLayer).item(0,0), 3)
        returnTable.append([suggestion, f'{round(position[0], 7)}°N, {round(position[1], 7)}°E', str(prediction), str(round(prediction-currentHDI, 3))])
    return returnTable

# place a pin on the map when user clicks the suggestion
def selectSuggestionsTable(evt: gr.SelectData):
    cellData = evt.value
    newMap = copy.deepcopy(foliumMap)
    if '°' in cellData: # it is a coordinate
        coordinates = [float(coord[:-2]) for coord in cellData.split(', ')] # convert it to floats, remove formatting
        folium.Marker(location=coordinates).add_to(newMap)
        newMap.location=coordinates
    return newMap

# validate sign up & add account to mongoDB
def signUp(username, password, password2):
    # validate no empty fields
    if username == '' or password == '' or password2 == '':
        return 'Please fill in all fields!'
    # validate username - does not already exist
    if users.count_documents({'username': username}) != 0:
        return 'This username is already in use! Please choose another username.'
    # validate password - double entry validation
    if password != password2:
        return 'The passwords do not match! Please ensure that you have entered the correct password.'
    # validate password - at least 8 characters long
    if len(password) < 8:
        return 'Your password must have at least 8 characters!'
    # validate password - at least 1 digit
    digits = '0123456789'
    hasDigit = False
    for digit in digits:
        if digit in password:
            hasDigit = True
    if not hasDigit:
        return 'Your password must include at least 1 digit!'
    # validate password - at least 1 special character
    specialChars = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    hasSpecialChar = False
    for specialChar in specialChars:
        if specialChar in password:
            hasSpecialChar = True
    if not hasSpecialChar:
        return f'Your password must include at least 1 special character! ({specialChars})'
    # salt & hash password
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    # add to mongoDB
    user = {
        'username': username,
        'hashedPassword': hashed_password,
        'regions': []
    }
    users.insert_one(user)
    return f'Successfully created an account, with username: {username}!'

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
with gr.Blocks() as app:
    with gr.Tab(label='Predict'):
        with gr.Row():
            with gr.Column(scale=3):
                map = Folium(value=foliumMap)
                uploadButton = gr.UploadButton(label='Upload the GeoJSON file', file_count='single')
            with gr.Column(scale=1):
                predictHDIbutton = gr.Button(value='Predict HDI', variant='primary')
                with gr.Group():
                    max_x_objectsSlider = gr.Slider(minimum=10, maximum=5000, value=500, label='Maximum x-buildings', info='When calculating the average distance between 2 types of building, it will find the average of this many pairs. Increasing this value may make the prediction take much longer, but may make the prediction more acurate.')
                    max_y_objectsSlider = gr.Slider(minimum=10, maximum=5000, value=500, label='Maximum y-buildings', info='When calculating the average distance between 2 types of building, it will find the closest of the second type of building out of this many options. Increasing this value may make the prediction take much longer, but may make the prediction more acurate.')
                with gr.Group():
                    HDIprediction = gr.Textbox(label='I believe that the HDI of this region is...', value='', interactive=False)
                    similarHDI = gr.Textbox(label='That\'s a similar HDI to...', value='', interactive=False)
        makeSuggestionsButton = gr.Button(value='Make Suggestions')
        suggestionsTable = gr.Dataframe(label='Suggestions', headers=['New Building', 'Coordinates', 'New HDI', 'Change in HDI'], type='array', interactive=False)
        log = gr.Textbox(label='Log Messages', value='', interactive=False)
    with gr.Tab(label='Log In'):
        logInUsername = gr.Textbox(label='Username', placeholder='Enter your username...')
        logInPassword = gr.Textbox(label='Password', placeholder='Enter your password...', type='password')
        logInButton = gr.Button(value='Log In', variant='primary')
        logInResult = gr.Textbox(label='Result', interactive=False)
    with gr.Tab(label='Sign Up'):
        signUpUsername = gr.Textbox(label='Username', placeholder='Enter your username...')
        signUpPassword = gr.Textbox(label='Password', placeholder='Enter your password...', type='password')
        signUpPassword2 = gr.Textbox(label='Re-Enter Password', placeholder='Re-Enter your password...', type='password')
        signUpButton = gr.Button(value='Create an Account', variant='primary')
        signUpResult = gr.Textbox(label='Result', interactive=False)
    
    # functionality
    uploadButton.upload(uploadGeoJSON, inputs=[uploadButton], outputs=[log])
    predictHDIbutton.click(processPrediction, inputs=[max_x_objectsSlider, max_y_objectsSlider], outputs=[HDIprediction])
    makeSuggestionsButton.click(makeSuggestions, inputs=[max_x_objectsSlider, max_y_objectsSlider], outputs=[suggestionsTable])
    suggestionsTable.select(selectSuggestionsTable, inputs=[], outputs=[map])
    signUpButton.click(signUp, inputs=[signUpUsername, signUpPassword, signUpPassword2], outputs=[signUpResult])

# launch the UI
app.launch()
import gradio as gr
from gradio_folium import Folium
import folium
from folium.plugins import Draw
import random
import json
from data_processing.get_osm_data import get_osm_data
from calc_factors import averageDistance, calcArea, density, distBetween2Points
import numpy as np
from neural_network import MultilayerPerceptron
import copy
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi
import bcrypt
import csv

# initialise neural network
network = MultilayerPerceptron([24, 18, 12, 6, 3, 1])
network.load_model('models/hdi.pkl')

# define global vars
currentRegion = []
all_objects = {}
allFactors = []
currentHDI = -1
allUserRegions = []
loggedInUsername = ''
averageDistanceFactors = [['house', 'school'], ['house', 'hospital'], ['house', 'pharmacy'], ['house', 'restaurant'], ['school', 'hospital'], ['police', 'hospital'], ['house', 'place_of_worship'], ['bank', 'slot_machines'], ['fast_food', 'toilets'], ['house', 'police'], ['university', 'library'], ['house', 'library']]
densityFactors = ['school', 'hospital', 'pharmacy', 'police', 'library', 'toilets', 'restaurant', 'place_of_worship', 'post_box', 'vending_machine', 'bench', 'tree']

# connect to mongoDB
mongoURI = "mongodb+srv://lukerhodri:jQeBcC6XLyxPEwK8@predictinghdi.3nkqe.mongodb.net/?retryWrites=true&w=majority&appName=predictingHDI"
mongoClient = MongoClient(mongoURI, server_api=ServerApi('1'), tlsCAFile=certifi.where())
users = mongoClient['predictingHDI']['users']
# send a ping to confirm a successful connection
mongoClient.admin.command('ping')
print("Pinged your deployment. You successfully connected to MongoDB!")

# load the training data
with open('data/training_data.csv', 'r') as file:
    reader = csv.DictReader(file)
    trainingData = []
    for record in reader:
        trainingData.append(record)

# code to process the user uploading their geojson file
def uploadGeoJSON(filename, max_x_objects, max_y_objects, currentRegionName, progress=gr.Progress()):
    global currentRegion
    global all_objects
    global allFactors
    global allUserRegions
    progress(0, desc='Starting...')
    with open(filename, 'r') as file:
        data = json.load(file)
        currentRegion = [coordinate[::-1] for coordinate in data['features'][0]['geometry']['coordinates'][0]]
    shortFilename = filename[filename.rindex("/")+1:]
    if shortFilename[:shortFilename.rindex('.')] in [region['name'] for region in allUserRegions] or shortFilename[:shortFilename.rindex('.')] == '':
        gr.Warning('There is already a region with this name! Please rename the file first.')
        return 'There is already a region with this name! Please rename the file first.', updateRegionsTable(), updateRegionsDropdown(), currentRegionName
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
        progress(0.91+0.09*(num/len(densityFactors)), desc=f'Calculating {densityFactor} Density...')
        print(f'finding D({densityFactor})') # log message
        allFactors.append(density(all_objects[densityFactor], area))
    allUserRegions.append({
        'name': shortFilename[:shortFilename.rindex('.')],
        'objects': all_objects,
        'factors': allFactors
    })
    # save regions to account, if logged in
    if loggedInUsername != '':
        users.update_one({'username': loggedInUsername}, {'$set': {'regions': allUserRegions}})
    gr.Info(f'Successfully Uploaded {shortFilename}', 5)
    return f'Successfully Uploaded {shortFilename}', updateRegionsTable(), updateRegionsDropdown(), shortFilename[:shortFilename.rindex('.')]

# predict the HDI of the given region
def processPrediction():
    global currentHDI
    if allFactors == []:
        gr.Warning('You have not uploaded a region!')
        return 'You have not uploaded a region!', ''
    # predict HDI
    inputLayer = np.matrix([[100 if factor == None else float(factor)/10 if index <= 11 else float(factor)] for index, factor in enumerate(allFactors)])
    prediction = round(network.predict(inputLayer).item(0,0), 3)
    currentHDI = prediction
    # find similar region
    diff = 0
    possible_choices = []
    while len(possible_choices) == 0:
        # search below & above (round to remove floating point errors)
        possible_choices += [region for region in trainingData if region['hdi'] == str(round(prediction-diff, 3))]
        possible_choices += [region for region in trainingData if region['hdi'] == str(round(prediction+diff, 3))]
        # increment difference (round to remove floating point errors)
        diff = round(diff + 0.001, 3)
    # choose one of the options at random
    chosenRegionRecord = random.choice(possible_choices)
    similar_region = f'{chosenRegionRecord["region"]}, {chosenRegionRecord["country"]}'
    gr.Info('Successfully Predicted HDI!', 5)
    return prediction, similar_region

# helper function to calculate mean of coordinates
def calcMeanOfCoords(coords):
    return [np.mean([coord[0] for coord in coords]), np.mean([coord[1] for coord in coords])]

# calculate the optimal place for a new building to go when suggesting
def calcOptimalPlaceFor(building, extraBuildings):
    global all_objects
    # construct all_objects_in_consideration
    all_objects_in_consideration = {}
    for key in all_objects.keys():
        if key in extraBuildings.keys():
            all_objects_in_consideration[key] = all_objects[key] + extraBuildings[key]
        else:
            all_objects_in_consideration[key] = all_objects[key]
    # iterate over the factors
    optimalPositions = []
    zeroOfEverything = True
    zeroOfThings = []
    for averageDistanceFactor in averageDistanceFactors:
        if averageDistanceFactor[1] == building:
            # initialise variables
            x_objects = all_objects_in_consideration[averageDistanceFactor[0]]
            y_objects = all_objects_in_consideration[averageDistanceFactor[1]]
            # update zero of everything flag
            if len(x_objects) != 0:
                zeroOfEverything = False
                # calculate the shortest distance from each {averageDistanceFactor[0]} to {averageDistanceFactor[1]}
                shortest_dists = []
                for x_object in x_objects:
                    dists = []
                    if len(y_objects) != 0:
                        for y_object in y_objects:
                            dists.append(distBetween2Points(x_object, y_object))
                    else:
                        dists.append(10000)
                    if min(dists) == 0:
                        shortest_dists.append(0.00001)
                    else:
                        shortest_dists.append(min(dists))
                # calculate the optimal position
                list_of_numpy_arrays = [np.array(x_object) for x_object in x_objects]
                optimalPosition = (sum([x_object/shortest_dists[index] for index, x_object in enumerate(list_of_numpy_arrays)], np.array([0,0])))/(sum([1/shortest_dist for shortest_dist in shortest_dists]))
                optimalPositions.append(optimalPosition)
            else:
                zeroOfThings.append(averageDistanceFactor[0])
    # return either something else to build, or the new position
    if zeroOfEverything:
        return random.choice(zeroOfThings)
    return calcMeanOfCoords(optimalPositions)

# make suggestions from a prediction
def makeSuggestions(max_x_objects, max_y_objects, num_new_buildings, progress=gr.Progress()):
    global all_objects
    if currentHDI == -1:
        gr.Warning('You have not yet predicted an HDI! Please predict one first.')
        return [['You have not yet predicted an HDI!', None, None, None]]
    if all_objects == {}:
        gr.Warning('This region does not have any associated buildings! Please choose a different region')
        return [['This region does not have any associated buildings!', 'Please choose a different region', None, None]]
    suggestions = ['school', 'hospital', 'place_of_worship', 'police', 'restaurant', 'slot_machines', 'library', 'pharmacy']
    returnTable = []
    for num, suggestion in enumerate(suggestions):
        extraBuildings = {suggestion: []}
        progress(num/len(suggestions), desc=f'Suggesting building a {suggestion}...')
        # find optimal place
        positions = []
        for buildingNum in range(num_new_buildings):
            # check for other suggestion
            if len(positions) > 0:
                if type(positions[0]) == type(''):
                    continue
            # calc actual position
            progress((num/len(suggestions))+(buildingNum*(1/8)*(1/(num_new_buildings+3))), desc=f'Suggesting building a {suggestion} (finding optimal position, {buildingNum+1}/{num_new_buildings})...')
            position = calcOptimalPlaceFor(suggestion, extraBuildings)
            positions.append(position)
            extraBuildings[suggestion].append(position)
        if type(positions[0]) != type(''):
            # calculate average distances
            progress((num/len(suggestions))+((num_new_buildings)*(1/8)*(1/(num_new_buildings+3))), desc=f'Suggesting building a {suggestion} (calculating new average distances)...')
            allFactors = []
            for averageDistanceFactor in averageDistanceFactors:
                # ensure that we include the new building
                if suggestion == averageDistanceFactor[0]:
                    allFactors.append(averageDistance(all_objects[averageDistanceFactor[0]], all_objects[averageDistanceFactor[1]], max_x_objects, max_y_objects, must_include_x=positions))
                elif suggestion == averageDistanceFactor[1]:
                    allFactors.append(averageDistance(all_objects[averageDistanceFactor[0]], all_objects[averageDistanceFactor[1]], max_x_objects, max_y_objects, must_include_y=positions))
                else:
                    allFactors.append(averageDistance(all_objects[averageDistanceFactor[0]], all_objects[averageDistanceFactor[1]], max_x_objects, max_y_objects))
            # calculate density factors
            progress((num/len(suggestions))+((num_new_buildings+1)*(1/8)*(1/(num_new_buildings+3))), desc=f'Suggesting building a {suggestion} (calculating new densities)...')
            area = calcArea(currentRegion)
            for densityFactor in densityFactors:
                # ensure that we include the new building
                if suggestion == densityFactor:
                    allFactors.append(density(all_objects[densityFactor] + [position], area))
                else:
                    allFactors.append(density(all_objects[densityFactor], area))
            # predict HDI
            progress((num/len(suggestions))+((num_new_buildings+2)*(1/8)*(1/(num_new_buildings+3))), desc=f'Suggesting building a {suggestion} (predicting new HDI)...')
            inputLayer = np.matrix([[100 if factor == None else float(factor)/10 if index <= 11 else float(factor)] for index, factor in enumerate(allFactors)])
            prediction = round(network.predict(inputLayer).item(0,0), 3)
            returnTable.append([suggestion, ', '.join([f'({round(position[0], 7)}°N, {round(position[1], 7)}°E)' for position in positions]), str(prediction), str(round(prediction-currentHDI, 3))])
        else:
            returnTable.append([suggestion, f'Build a new {positions[0]} first!', '--', '--'])
    gr.Info('Successfully Made Suggestions!', 5)
    return returnTable

# place a pin on the map when user clicks the suggestion
def selectSuggestionsTable(evt: gr.SelectData):
    cellData = evt.value
    newMap = copy.deepcopy(foliumMap)
    if '°' in cellData: # it is a coordinate
        splitByCommas = cellData.split(', ')
        coordinates = []
        for halfIndex in range(int(len(splitByCommas)/2)):
            coordinates.append([float(splitByCommas[2*halfIndex][1:-2]), float(splitByCommas[2*halfIndex+1][:-3])])
        for coordinate in coordinates:
            folium.Marker(location=coordinate).add_to(newMap)
        newMap.location=random.choice(coordinates)
    return newMap

# user selects an item from the dropdown menu to compare to
def compareRegions(regionName):
    # confirm the user has predicted
    if allFactors == []:
        gr.Warning('Please predict a region first!')
        return [['Please predict a region first!', None, None]]
    # undo the concatenation
    split = regionName.split(', ')
    country = split[-1]
    region = ', '.join(split[:-1])
    # find the region record & get lists of factors
    regionRecord = [record for record in trainingData if record['country'] == country and record['region'] == region][0]
    yourFactors = ['undefined' if factor == None else round(float(factor), 5) for factor in allFactors]
    selectedFactors = ['undefined' if factor == '' else round(float(factor), 5) for factor in list(regionRecord.values())[4:]]
    # construct the return list
    factorNames = ['Average Distance from House to nearest School (km)','Average Distance from House to nearest Hospital (km)','Average Distance from House to nearest Pharmacy (km)','Average Distance from House to nearest Restaurant (km)','Average Distance from School to nearest Hospital (km)','Average Distance from Police Station to nearest Hospital (km)','Average Distance from House to nearest Place of Worship (km)','Average Distance from Bank to nearest Slot Machine (km)','Average Distance from Fast-Food Place to nearest Toilet (km)','Average Distance from House to nearest Police Station (km)','Average Distance from University to nearest Library (km)','Average Distance from House to nearest Library (km)','Number of Schools per km²','Number of Hospitals per km²','Number of Pharmcies per km²','Number of Police Stations per km²','Number of Libraries per km²','Number of Toilets per km²','Number of Restaurants per km²','Number of Places of Worship per km²','Number of Post Boxes per km²','Number of Vending Machines per km²','Number of Benches per km²','Number of Trees per km²']
    returnList = [['HDI', currentHDI, regionRecord['hdi']]]
    for num, factor in enumerate(factorNames):
        returnList.append([factor, yourFactors[num], selectedFactors[num]])
    return returnList

# creates a 2d list to put in the regions table, from allUserRegions 
def updateRegionsTable():
    table = []
    for region in allUserRegions:
        table.append([region['name']] + region['factors'])
    # if allUserRegions is empty, return an empty table
    if table == []:
        table.append([''*25])
    return table

# updates regions dropdown, from allUserRegions
def updateRegionsDropdown():
    names = [region['name'] for region in allUserRegions]
    return gr.update(choices=names)

# updates the allUserRegions list, when the regions table is updated
def updateAllUserRegions(table, currentRegionName):
    global allUserRegions
    # store the index of the current region
    if currentRegionName != None:
        currentRegionIndex = [region['name'] for region in allUserRegions].index(currentRegionName)
    # validate the input - names of regions
    regionNames = [row[0] for row in table]
    if '' in regionNames or len(set(regionNames)) != len(regionNames):
        gr.Warning('Please ensure every region has a unique name!')
        return 'Please ensure every region has a unique name!', updateRegionsDropdown(), currentRegionName
    # validate the input - type check
    for rowIndex, row in enumerate(table):
        for cellIndex, cell in enumerate(row):
            if cellIndex != 0 and (cellIndex > 12 or cell != ''):
                try:
                    table[rowIndex][cellIndex] = float(cell)
                except ValueError:
                    gr.Warning(f'Please Enter the Data Correctly! Found an error at cell (row {rowIndex}, col {cellIndex})')
                    return f'Please Enter the Data Correctly! Found an error at cell (row {rowIndex}, col {cellIndex})', updateRegionsDropdown(), currentRegionName
    # update the allUserRegions list
    initialLength = len(allUserRegions)
    for index, row in enumerate(table):
        factors = [None if value == '' else value for value in row[1:]]
        # if the user has changed the factors of an existing list, the associated objects should be removed
        if index < initialLength:
            if [float(factor) if factor != '' else None for factor in row[1:]] != allUserRegions[index]['factors']:
                allUserRegions[index] = {'name': row[0], 'objects': {}, 'factors': factors}
            else:
                allUserRegions[index]['name'] = row[0]
        # if the user has added a new region, it should have no associated objects
        else:
            allUserRegions.append({'name': row[0], 'objects': {}, 'factors': factors})
    # save regions to account, if logged in
    if loggedInUsername != '':
        users.update_one({'username': loggedInUsername}, {'$set': {'regions': allUserRegions}})
    # update the name of the selected region
    if currentRegionName == None:
        newRegionName = None
    else:
        newRegionName = allUserRegions[currentRegionIndex]['name']
    gr.Info('Successfully Updated Regions', 5)
    return 'Successfully Updated Regions', updateRegionsDropdown(), newRegionName

# updates the current region, when the user switches it with the dropdown on the regions tab
def updateCurrentRegion(dropdownValue):
    global all_objects
    global allFactors
    global currentHDI
    for region in allUserRegions:
        if region['name'] == dropdownValue:
            all_objects = region['objects']
            allFactors = region['factors']
            # predict HDI - to update it internally
            inputLayer = np.matrix([[100 if factor == None else float(factor)/10 if index <= 11 else float(factor)] for index, factor in enumerate(allFactors)])
            prediction = round(network.predict(inputLayer).item(0,0), 3)
            currentHDI = prediction

# validate sign up & add account to mongoDB
def signUp(username, password, password2):
    global loggedInUsername
    # validate no empty fields
    if username == '' or password == '' or password2 == '':
        gr.Warning('A field has been left empty! Please fill in all fields.')
        return 'Please fill in all fields!', updateRegionsTable(), updateRegionsDropdown()
    # validate username - does not already exist
    if users.count_documents({'username': username}) != 0:
        gr.Warning('This username is already in use! Please choose another username.')
        return 'This username is already in use! Please choose another username.', updateRegionsTable(), updateRegionsDropdown()
    # validate password - double entry validation
    if password != password2:
        gr.Warning('The passwords do not match! Please ensure that you have entered the correct password.')
        return 'The passwords do not match! Please ensure that you have entered the correct password.', updateRegionsTable(), updateRegionsDropdown()
    # validate password - at least 8 characters long
    if len(password) < 8:
        gr.Warning('Your password must have at least 8 characters!')
        return 'Your password must have at least 8 characters!', updateRegionsTable(), updateRegionsDropdown()
    # validate password - at least 1 digit
    digits = '0123456789'
    hasDigit = False
    for digit in digits:
        if digit in password:
            hasDigit = True
    if not hasDigit:
        gr.Warning('Your password must include at least 1 digit!')
        return 'Your password must include at least 1 digit!', updateRegionsTable(), updateRegionsDropdown()
    # validate password - at least 1 special character
    specialChars = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    hasSpecialChar = False
    for specialChar in specialChars:
        if specialChar in password:
            hasSpecialChar = True
    if not hasSpecialChar:
        gr.Warning('Your password must include at least 1 special character!')
        return f'Your password must include at least 1 special character! ({specialChars})', updateRegionsTable(), updateRegionsDropdown()
    # salt & hash password
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    # add to mongoDB
    user = {
        'username': username,
        'hashedPassword': hashed_password,
        'regions': allUserRegions
    }
    users.insert_one(user)
    loggedInUsername = username
    gr.Info('Successfully created & logged into an account', 5)
    return f'Successfully created & logged into an account, with username: {username}!', updateRegionsTable(), updateRegionsDropdown()

# validate & log in the user
def logIn(username, password):
    global loggedInUsername
    global allUserRegions
    # validate no empty fields
    if username == '' or password == '':
        gr.Warning('A field has been left empty! Please fill in all fields.')
        return 'Please fill in all fields!', updateRegionsTable(), updateRegionsDropdown()
    # check username exists
    if users.count_documents({'username': username}) == 0:
        gr.Warning('This username does not exist! Please ensure that you have entered the correct username.')
        return 'This username does not exist! Please ensure that you have entered the correct username.', updateRegionsTable(), updateRegionsDropdown()
    # check password
    user = users.find_one({'username': username})
    if not bcrypt.checkpw(password.encode('utf-8'), user['hashedPassword']): # the hashed passwords do not match
        gr.Warning('Incorrect Password! Please try again.')
        return 'Incorrect Password! Please try again.', updateRegionsTable(), updateRegionsDropdown()
    # grant access to the account
    loggedInUsername = username
    allUserRegions = user['regions']
    gr.Info(f'Successfully logged in as {username}!', 5)
    return f'Successfully logged in as {username}!', updateRegionsTable(), updateRegionsDropdown()

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

# define markdown in help tab
markdown = '''
# \U0001F44B Welcome to Predicting HDI! \U0001F44B

This app allows you to predict the HDI of any region you draw on the map. From there, you can make some suggestions on how to improve it.

## Getting Started \U00002728

Head over to the \U0001F52E Predict tab, and find your region on the map. Once you have found your region, you can predict the HDI by following these steps:

1. Select the Draw Tool by Clicking the **Pentagon** \U00002B53 Icon in the Top Left.
2. Draw your Region, by creating vertices. Make sure you close the region by clicking on your first vertex at the end.
3. Press the **Export** button in the Top Right. This will download a file called `region.geojson` to your computer.
4. Press the \U0001F4E4 **Upload the GeoJSON file** \U0001F4E4 button, below the map, and choose your `region.geojson`. You can scroll down to see a progress bar for this upload. Once the upload is complete, you will get a notification in the top right corner.
5. Finally, press the orange \U0001F52E **Predict HDI** \U0001F52E button to predict the HDI of your region!

![](file/data/helpTab1.png "Diagram visually showing the above steps")

## More Detail on The \U0001F52E Predict Tab

### Help! It's Taking Too Long! \U0001F553

If you drew a very large region, it is likely that Uploading it took a **very long** time \U0001F553. This is because, to predict the HDI, we use data about the positions of buildings, such as houses, schools, hospitals and police stations. This data must be downloaded, so if you chose a region with a lot of these things, then it will take quite a while to download it all.

To predict the HDI from these positions, we calculate different different factors, such as the average distance from a house to the nearest school, or nearest hospital, which can also take a while. To decrease the time this may take, you can change the values of the **Maximum x-buildings** \U0001F3E0 and **Maximum y-buildings** \U0001F3EB sliders.

Changing **Maximum x-buildings** \U0001F3E0 will decrease the number of x-buildings we consider. For example, in the case of average distance from house to school, this slider corresponds to the number of houses we consider. Similarly, changing **Maximum y-buildings** \U0001F3EB corresponds to changing the number of schools we consider for each house (in this example). However, this will cause the calculated values for these factors to be less accurate, and therefore may cause the predicted HDI to be less accurate.

### Help! It Just Crashed! \U0000274C

If the progress bar suddenly turned into an **ERROR** when uploading a region, it is likely that you just timed out from the OverpassTurbo servers, which is what we are using to get the location of all these buildings. This is totally normal, and so if this happens, just try to upload it again.

If the issue persists, it is likely something is wrong with your region, so try a different one

### Suggestions \U0001F4A1

Once you have predicted the HDI of a region, you can make some suggestions. Do this, by simply pressing the \U0001F914 **Make Suggestions** \U0001F914 button. The suggestions will be in the format: "Build (a number of) new buildings, in these locations". You can change this value, to anything between 1 and 10, by moving the slider, labelled **Number of New Buildings** \U0001F3D9.

This may also take some time, As we must calculate the optimal position to place these new buildings, and the re-calculate the factors. Once it is finished, the Suggestions \U0001F4A1 Table will be populated with 8 different suggestions, on how to improve the HDI of your region. To view the locations of the suggestion on the map, simply select the cell that contains the coordinates. The map will update to have pins.

![](file/data/helpTab2.png "Typical Output for the Suggestions Table")

If your region is quite empty, you may have got some suggestions that did not return any locations, or a new HDI, but instead a prompt to build some other buildings first. This is because we did not have sufficient data to calculate where a new building should go, and so unfortunately, there is no real suggestion.

### What Makes a Good Region? \U0001F914

This brings me onto how to select a region, which will yield good results.

- **Don't make your region too large.** Larger regions will take longer to download, and longer to calculate the factors of. You will therefore want to decrease the **Maximum x-buildings** \U0001F3E0 and **Maximum y-buildings** \U0001F3EB sliders, leading to a less accurate prediction. You will then also not get very good suggestions, as building 5 schools in a large metropolitan area isn't going to do anything.
- **Ensure your region doesn't self-intersect.** Your region should be a simple 2D shape.
- **Use Small Towns.** I believe this works best with small towns, as the suggestions will often increase the HDI by a more significant amount. Downloading the Region and Calculating the Factors also won't take too long.
- **Don't Make Empty Regons.** Predicting the HDI of an empty field or the ocean is not going to yield any useful results.

### Drawing Regions \U0001F58B

If you make a mistake while drawing your region, you can press the "Delete last point" button, next to the Pentagon \U00002B53 icon. If you wish to delete your region, press the Bin \U0001F5D1 icon in the top left.

You can zoom in and out of the map, by using the scroll wheel, or by using the + and - buttons in the top right.

If you draw multiple regions before pressing Export, we will only consider the first one you drew. If you wish to predict the HDI of 2 disconnected regions, you can get around this by connecting them with a very thin line.

## The \U0001F19A Compare Tab

### Comparing your Region \U0001F4CA

You can use the \U0001F19A Compare tab to compare your region to an existing sub-national region. Sub-national regions are 1 level below countries, so for example, the sub-national regions of the United States are the 50 states.

To do this, simply select a region from the dropdown menu at the top of the screen. To search for a specific region, you can also type into this dropdown menu. When you select one, the table below will populate with the HDI of your region and the selected region, as well as each of the factors that contribute to it.

![](file/data/helpTab3.png "Typical Output for the Comparison Table")

\U0001F4C4 **Note:** You must predict the HDI of a region first, before you do this.

### Correlation Heatmap \U0001F4C8

At the bottom of this tab, you will also find a graph, showing the correlation between each factor, and HDI. You can use this to find out which factors contribute most towards the HDI, as well as which ones contribute towards each other.

Each cell has a colour, which corresponds to a number between -1 and 1, as shown by the key on the right. A value of 1 means that the 2 factors are very positively correlated, a value of -1 means that the 2 factors are very negatively correlated, and a value of 0 means that the 2 factors are not correlated. All values in between are as you would expect (for example, a value of -0.3 would mean that the 2 factors are slightly negatively correlated).

## The \U0001F30D Regions Tab

In the \U0001F30D Regions tab, you can switch between different regions that you have predicted the HDI of. To do this, simply select one from the dropdown menu at the top of the screen.

This tab also allows you to manually edit the factors of your regions, rename your regions, and add new ones. You can edit any cell in the table, so long as they follow these rules:

- The values in the columns for A(... ...) must either be a number or empty
- The values in the columns for D(...) must be a number
- The values in the name column must be unique

(A(x y) means "Average distance from x to nearest y", and D(x) means "Number of x per unit area")

Once you have made your changes, you can press the \U0001F504 **Submit Table Changes** \U0001F504 button, to make your changes. If any of the rules above have not been met, the changes will not go through, and you will be told which cell is causing an error.

\U00002757 **WARNING:** You will no longer be able to make suggestions from regions that you edit in the table. \U00002757 This is because the factors no longer reflect the positions of the buildings in the region, and so they are no longer accurate. We therefore cannot make any suggestions, as there are no buildings associated with the region.

## Account Management \U0001F510

You can also create an account, to save your regions.

### Creating an Account \U0001F4DD

To create an account, simply head to the \U0001F4DD Sign Up tab. Here, you get to choose a username and password. Your username must be unique, and your password must:

- Be at least 8 characters long
- Contain at least 1 digit
- Contain at least 1 special character. The characters that count as special characters are: ! " # $ % & \' ( ) * + , - . / : ; < = > ? @ [ \\ ] ^ _ ` { | } ~

When you create an account, you will automatically be logged in. Any changes you make while logged in will be saved to your account.

### Logging Into an Account \U0001F513

To log into your account, head to the \U0001F513 Log In tab, and enter your username and password.

\U00002757 **WARNING:** Logging in will override the regions in the \U0001F30D Regions Tab, and replace them with the ones in the account you log into. \U00002757
'''

# structure of the UI
with gr.Blocks() as app:
    with gr.Tab(label='\U0001F64B Help'):
        gr.Markdown(markdown)
    with gr.Tab(label='\U0001F52E Predict'):
        with gr.Row():
            with gr.Column(scale=3):
                map = Folium(value=foliumMap)
                uploadButton = gr.UploadButton(label='\U0001F4E4 Upload the GeoJSON file \U0001F4E4', file_count='single')
            with gr.Column(scale=1):
                predictHDIbutton = gr.Button(value='\U0001F52E Predict HDI \U0001F52E', variant='primary')
                with gr.Group():
                    max_x_objectsSlider = gr.Slider(minimum=10, maximum=5000, value=500, label='Maximum x-buildings \U0001F3E0', info='When calculating the average distance between 2 types of building, it will find the average of this many pairs. Increasing this value may make the prediction take much longer, but may make the prediction more acurate.')
                    max_y_objectsSlider = gr.Slider(minimum=10, maximum=5000, value=500, label='Maximum y-buildings \U0001F3EB', info='When calculating the average distance between 2 types of building, it will find the closest of the second type of building out of this many options. Increasing this value may make the prediction take much longer, but may make the prediction more acurate.')
                with gr.Group():
                    HDIprediction = gr.Textbox(label='I believe that the HDI of this region is... \U00002728', value='', interactive=False)
                    similarHDI = gr.Textbox(label='That\'s a similar HDI to... \U0001F30D', value='', interactive=False)
        with gr.Row():
            makeSuggestionsButton = gr.Button(value='\U0001F914 Make Suggestions \U0001F914')
            newBuildingsSlider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label='Number of New Buildings \U0001F3D9', interactive=True)
        suggestionsTable = gr.Dataframe(label='Suggestions \U0001F4A1', headers=['New Building', 'Coordinates', 'New HDI', 'Change in HDI'], type='array', interactive=False)
        log = gr.Textbox(label='Log Messages \U0001F4C4', value='', interactive=False)
    with gr.Tab(label='\U0001F19A Compare'):
        compareDropdown = gr.Dropdown(choices=[f'{region["region"]}, {region["country"]}' for region in trainingData], label='Select a Region \U0001F30D')
        compareTable = gr.DataFrame(label='Comparison \U0001F4CA', headers=['Factor', 'Your Region', 'Selected Region'], type='array', interactive=False)
        pmccImage = gr.Image('data/pmcc.png', label='Correlation between Factors \U0001F4C8', height=600, interactive=False)
    with gr.Tab(label='\U0001F30D Regions'):
        regionsDropdown = gr.Dropdown(choices=[], label='Current Region \U0001F30D', interactive=True)
        regionsTable = gr.DataFrame(label='Your Regions \U0001F304', headers=(['Name'] + list(trainingData[0].keys())[4:]), type='array', col_count=(25,'fixed'), interactive=True)
        submitEditsButton = gr.Button(value='\U0001F504 Submit Table Changes \U0001F504', variant='primary')
        gr.Markdown('\U00002757 **WARNING:** You will no longer be able to make suggestions from regions that you edit here. \U00002757')
        submitEditsResult = gr.Textbox(label='Result \U0001F4C4', interactive=False)
    with gr.Tab(label='\U0001F513 Log In'):
        with gr.Row():
            with gr.Column():
                logInUsername = gr.Textbox(label='Username \U0001F50F', placeholder='Enter your username...')
                logInPassword = gr.Textbox(label='Password \U0001F511', placeholder='Enter your password...', type='password')
            with gr.Column():
                logInButton = gr.Button(value='\U0001F4F2 Log In \U0001F4F2', variant='primary')
                gr.Markdown('\U00002757 **WARNING:** Logging in will override the regions in the \U0001F30D Regions Tab, and replace them with the ones in the account you log into. \U00002757')
                logInResult = gr.Textbox(label='Result \U0001F4C4', interactive=False)
    with gr.Tab(label='\U0001F4DD Sign Up'):
        with gr.Row():
            with gr.Column():
                signUpUsername = gr.Textbox(label='Username \U0001F50F', placeholder='Enter your username...')
                signUpPassword = gr.Textbox(label='Password \U0001F511', placeholder='Enter your password...', type='password')
                signUpPassword2 = gr.Textbox(label='Re-Enter Password \U0001F510', placeholder='Re-Enter your password...', type='password')
                gr.Markdown('''
                Your password must:
                - Be at least 8 characters
                - Contain at least 1 digit
                - Contain at least 1 special character
                ''')
            with gr.Column():
                signUpButton = gr.Button(value='\U0001F680 Create an Account \U0001F680', variant='primary')
                signUpResult = gr.Textbox(label='Result \U0001F4C4', interactive=False)
    
    # functionality
    uploadButton.upload(uploadGeoJSON, inputs=[uploadButton, max_x_objectsSlider, max_y_objectsSlider, regionsDropdown], outputs=[log, regionsTable, regionsDropdown, regionsDropdown])
    predictHDIbutton.click(processPrediction, inputs=[], outputs=[HDIprediction, similarHDI])
    makeSuggestionsButton.click(makeSuggestions, inputs=[max_x_objectsSlider, max_y_objectsSlider, newBuildingsSlider], outputs=[suggestionsTable])
    suggestionsTable.select(selectSuggestionsTable, inputs=[], outputs=[map])
    compareDropdown.input(compareRegions, inputs=[compareDropdown], outputs=[compareTable])
    regionsDropdown.change(updateCurrentRegion, inputs=[regionsDropdown], outputs=[])
    submitEditsButton.click(updateAllUserRegions, inputs=[regionsTable, regionsDropdown], outputs=[submitEditsResult, regionsDropdown, regionsDropdown])
    signUpButton.click(signUp, inputs=[signUpUsername, signUpPassword, signUpPassword2], outputs=[signUpResult, regionsTable, regionsDropdown])
    logInButton.click(logIn, inputs=[logInUsername, logInPassword], outputs=[logInResult, regionsTable, regionsDropdown])

# launch the UI
app.launch(allowed_paths=["/"])
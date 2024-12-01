from calc_factors import getAllFactors
import json
import csv

print('reading files...') # log message
# read json file
with open('data/region_coords.json', 'r') as file:
    regionData = json.load(file)

# read csv file
with open('data/hdi.csv', 'r') as file:
    reader = csv.DictReader(file)
    hdiData = []
    for record in reader:
        hdiData.append(record)

MOST_RECENT_REGION = 'AFGr108'
field_names = ['country', 'region', 'code', 'hdi', 'A(House School)', 'A(House Hospital)', 'A(House Pharmacy)', 'A(House Restaurant)', 'A(School Hospital)', 'A(Police Hospital)', 'A(House Place of Worship)', 'A(Bank Slot Machine)', 'A(Fast-Food Place Toilet)', 'A(House Police)', 'A(University Library)', 'A(House Library)', 'D(School)', 'D(Hospital)', 'D(Pharmacy)', 'D(Police)', 'D(Library)', 'D(Toilet)', 'D(Restaurant)', 'D(Place of Worship)', 'D(Post Box)', 'D(Vending Machine)', 'D(Bench)', 'D(Tree)']
SKIP_THE_FIRST = 0

# iterate over each region
for region_number, region in enumerate(regionData):
    if not started: 
        if region['properties']['gdlcode'] == MOST_RECENT_REGION:
            started = True
        continue
    print(f'Processing {region["properties"]["gdlcode"]}...') # log message
    # find the main bit of the region (if it has multiple)
    shapes = region['geometry']['coordinates']
    if region['geometry']['type'] == 'Polygon':
        mainShape = shapes[0]
    else:
        lengths = [len(polygon[0]) for polygon in shapes]
        mainShape = shapes[lengths.index(max(lengths))][0]
    mainShape = [coordinate[::-1] for coordinate in mainShape]
    # find all the factors
    allFactors = getAllFactors(mainShape)
    # match it with the hdi
    matchingHDI = [record for record in hdiData if record['code'] == region['properties']['gdlcode']]
    if len(matchingHDI) != 0:
        newRecord = matchingHDI[0]
        newRecord.update(allFactors)
        print('writing to file...') # log message
        with open('src/data/training_data.csv', 'a') as file:
            writer = csv.DictWriter(file, fieldnames=field_names)
            writer.writerow(newRecord)
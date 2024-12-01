import shapefile
import json

# read records of shapefile
reader = shapefile.Reader("data/shapefiles/GDL Shapefiles V6.3 large.shp")
fields = reader.fields[1:]
field_names = [field[0] for field in fields]
data = []
for shape_record in reader.shapeRecords():
    print(shape_record.record) # log message
    data.append({
        "type": "Feature",
        "geometry": shape_record.shape.__geo_interface__,
        "properties": dict(zip(field_names, shape_record.record))
    })

# write data to file
print('writing to file...') # log message
with open("data/region_coords.json", "w") as file:
    json.dump(data, file)
print('done!') # log message
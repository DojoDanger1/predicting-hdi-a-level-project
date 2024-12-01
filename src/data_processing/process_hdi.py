import csv

#retrieve data
with open('data/subnationalHDI.csv', 'r') as file:
    reader = csv.DictReader(file)
    newcsv = []
    for record in reader:
        if record['year'] == '2022': # the most recent year
            newRecord = {
                "country": record["country"],
                "region": record["region"],
                "code": record["GDLCODE"],
                "hdi": record["shdi"]
            } # only the most useful information
            newcsv.append(newRecord)
            print(newRecord)

#write to new csv
with open('data/hdi.csv', 'w') as file:
    field_names = ['country', 'region', 'code', 'hdi']
    writer = csv.DictWriter(file, fieldnames=field_names)
    writer.writeheader()
    for record in newcsv:
        writer.writerow(record)
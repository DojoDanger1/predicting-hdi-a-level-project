import csv

with open('src/data/subnationalHDI.csv', 'r') as file:
    reader = csv.DictReader(file)
    newcsv = []
    for record in reader:
        if record['year'] == '2022' and record['region'] != 'Total': # the most recent year + dont include full countries
            newRecord = {
                "country": record["country"],
                "region": record["region"],
                "hdi": record["shdi"]
            } # only the most useful information
            newcsv.append(newRecord)

with open('src/data/hdi.csv', 'w') as file:
    field_names = ['country', 'region', 'hdi']
    writer = csv.DictWriter(file, fieldnames=field_names)
    writer.writeheader()
    for record in newcsv:
        writer.writerow(record)
import csv

from os import path


def build_subset(csv_data, csv_reduced):
    count = 0
    csv.field_size_limit(500 * 1024 * 1024)

    with open(csv_data, newline='', encoding="UTF-8") as translation:
        with open(csv_reduced, 'w', newline='', encoding="UTF-8") as reduced_metadata:
            metadataReader = csv.reader(translation, delimiter=';')
            metadataList = list(metadataReader)

            first_row = True;

            for row in metadataList:
                genre = ''

                # skip titles
                if first_row:
                    first_row = False;
                    metadataWriter = csv.writer(reduced_metadata, delimiter=';')
                    metadataWriter.writerow(row)
                    continue

                id = str(row[0]).split('/')
                collection = id[1]
                file = id[2] + '.mp3'

                filepath = 'G:/EU_SOUND_CHANNEL/audio/'
                if path.isfile(path.join(filepath, collection, file)):
                    count += 1
                    metadataWriter = csv.writer(reduced_metadata, delimiter=';')
                    metadataWriter.writerow(row)
    return count


### Do Stuff
mypath = 'D:\Dropbox\Dropbox_Uni\Europena_NEW'
csv_data = path.join(mypath, 'metadata_translation_v2.csv')
csv_reduced = path.join(mypath, 'metadata_translation_v2_reduced.csv')
number_of_files = build_subset(csv_data, csv_reduced)
print('Subset of files and metadata: ', number_of_files)

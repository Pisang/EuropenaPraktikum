import csv
import logging


class HardcodedRules:
    def applyRules(self, csv_data, csv_groundTruth, csv_no_genre_found):
        logging.info('aplying hardcoded rules')
        csv.field_size_limit(500 * 1024 * 1024)

        with open(csv_data, newline='', encoding="UTF-8") as translation:
            with open(csv_groundTruth, 'w', newline='', encoding="UTF-8") as metadata_genre:
                with open(csv_no_genre_found, 'w', newline='', encoding='UTF-8') as no_genre_found:

                    metadataReader = csv.reader(translation, delimiter=';')
                    metadataList = list(metadataReader)

                    first_row = True;

                    for row in metadataList:
                        genre = ''

                        # skip titles
                        if first_row:
                            first_row = False;
                            metadataWriter = csv.writer(metadata_genre, delimiter=';')
                            metadataWriter_no_genre = csv.writer(no_genre_found, delimiter=';')
                            metadataWriter.writerow(row)
                            metadataWriter_no_genre.writerow(row)
                            continue

                        genre_found = False
                        for column in row:
                            if ('mozart' in column.lower() or 'schubert, franz' in column.lower()):
                                genre = 'classical'
                                genre_found = True

                            if ('testimony' in column.lower() and 'interview' in column.lower()):
                                genre = 'spoken'
                                genre_found = True
                            # A spoken life: [recorded autobiography between 1963 and 1994]
                            if (len(row[17].split(' ')) > 10):
                                if (row[17].split(' ')[4] == 'autobiography'):
                                    genre = 'spoken'
                                    genre_found = True
                            if (('free discussion' in row[18].lower() or 'interview' == row[18].lower())):
                                genre = 'spoken'
                                genre_found = True

                            if (row[16].lower() == 'instrumental folk music'):
                                genre = 'folklore'
                                genre_found = True
                            if ('latin american folk' in column.lower()):
                                genre = 'folklore'
                                genre_found = True

                            if ('sound effect' in column.lower()):
                                genre = 'invironment'
                                genre_found = True

                            if ('irish traditional' in column.lower()):
                                genre = 'irish'
                                genre_found = True

                            if ('popular music' in column.lower()):
                                genre = 'popular'
                                genre_found = True

                            if row[2] == 'greece' and (('dance song' in column.lower())
                                                       or ('table song' in column.lower())):
                                genre = 'folklore'
                                genre_found = True

                            if row[2] == 'france' and (
                                            '(investigator)' in column.lower() and '(informant)' in column.lower()):
                                genre = 'spoken'
                                genre_found = True

                        row.append(genre)
                        metadataWriter = csv.writer(metadata_genre, delimiter=';')
                        metadataWriter.writerow(row)

                        if genre_found:
                            pass
                            # row.append(genre)
                            # metadataWriter = csv.writer(metadata_genre, delimiter=';')
                            # metadataWriter.writerow(row)
                        else:
                            pass
                            # metadataWriter_no_genre = csv.writer(no_genre_found, delimiter=';')
                            # metadataWriter_no_genre.writerow(row)

import logging
import unicodedata
from collections import defaultdict

import pandas as pd
from bs4 import BeautifulSoup
from stop_words import get_stop_words


class DocManager:
    def reduceRows(self, csv_file, list_of_rows):
        '''
        reduce the csv file to a set of rows which are defined in list_of_rows.

        :param csv_file: path to the original matadata file
        :param list_of_rows: set of rows which are to be in the new file
        :return: pandas DataFrame containing the selection of rows
        '''
        logging.info('Reducing Rows')

        metadata_df = pd.read_csv(csv_file, sep=";", encoding="utf-8", index_col='id')

        reduced = metadata_df[list_of_rows]

        return reduced

    def remove_single_words(self, csv_file):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        logging.info('removing words that appear only once')

        # remove words that appear only once
        frequency = defaultdict(int)

        for row in csv_file.columns.values:
            for item in csv_file[row]:
                if type(item) == float:
                    item = str(int(item))
                for token in item.split():
                    frequency[token] += 1

        for row in csv_file.columns.values:
            csv_file[row] = csv_file[row].apply(lambda x: [item for item in str(x).split() if frequency[item] > 1])

        for row in csv_file.columns.values:
            csv_file[row] = csv_file[row].apply(' '.join)

        return csv_file

    def do_clinsing_csv(self, csv_file):
        '''
        Perform clinsing on given csv_file. 
         - remove stopwords
         - force lower case
         - normalize (remove umlaute)
         - remove words that appear only once
         
        :param csv_file: 
        :return: clean csv file
        '''
        logging.info('startin clinsing to a csv file')
        # put it into csv

        stop_words = []
        stop_words.extend(get_stop_words('en'))
        stop_words.extend(get_stop_words('de'))
        stop_words.extend(get_stop_words('fr'))
        stop_words.extend(get_stop_words('it'))
        stop_words.extend(get_stop_words('pt'))
        stop_words.extend(get_stop_words('ro'))
        stop_words.extend(get_stop_words('spanish'))
        soup = BeautifulSoup('html', 'lxml')
        text = soup.get_text()
        stop_words.extend(text)
        my_stopwords = '00 000 01 02 03 03t14 03t18 04 04bravicr 04volamih 04vollusm 04volverr 05 06 07 08 09 09t12 1 10 100 ' \
                       '102 104 105 108 109 11 110 111 112 114 115 116 119 11942505816 11t11 12 120 12000 121 122 123 125 126 ' \
                       '128 13 130 132 135 138 139 14 140 144 14429 14e 15 150 154 155 16 160 161294 162 163 166 168 16864 17 ' \
                       '17085 173 176 17675 17796 17t12 18 180 183 184 189 18t12 19 190 191 192 193 194 195 198 19c 2 20 200 ' \
                       '20206 204 21 215 21588 216 21672 22 22205 22389 22435 22917 22936 22t20 23 230 232 23277 235 23986 24 ' \
                       '240 24808 25 250 252 257 26 264 27 271 27267 277 27t11 28 280 285 288 28t10 29 3 30 300 3000 30t11 31 ' \
                       '318 31t13 32 32a 33 330 3373 3m2 34 340 35 35471 359998 36 360 36279 36706 37 3700 370a 37271 378345 38 ' \
                       '38379 39 3o 4 40 400 4000 4060 40714 41 42 43 4300 4375309 439b 44 4445 45 450 46 463 4673 47 48 4814 ' \
                       '486 49 492 49403 50 500 50496 50s 51 51a 52 520 52002709 527 53 530 53473 54 543 55 5506 56 57 571 58 ' \
                       '58112 588 59 5r 60 600 61 610 62 620 63 64 65 6502 66 67 68 680 69 70 700 69th 70 700 71 72 72000 7278 ' \
                       '73 74 75 75019 76 77 78 79 79108 7rl 80 800 81 82 83 84 844 84968 84bajimbs 84bonmala 84cucbrem ' \
                       '84gracham 84lacpare 84lactrum 84merjour 84robamap 84routamg 84vitallm 84vitmoul 84vitolly 85 850222 ' \
                       '86 865 86b 87 87981 88 885219 89 89147 89437 90 900 91 91344 92 93 94 94629 95 96 97 98 99 9bis a a1 ' \
                       'a2 a7 aa aaa aaarr aacid aakjr aalge aasmund ab absolutely ac across actual actually ad06 add added ' \
                       'adding adds afterwards ago agree agreed agrees ah ai aim al already also always another anymore anyone ' \
                       'anything anywhere ap apr ar are arr atd ate au aui auk av ax ay b b1 b2 bab began begging begin begins ' \
                       'begun behind besides beyond big bigger biggest bn bnf bni bo br iz eg cl11 cl50 cl5 can us mr unknown ' \
                       'thirteen cd th are may unless otherwise tuesday january unlike dr almost although anymore anyone ' \
                       'anything anywhere appropriate appropriately &quot &untitled &apos &amp &quot &lt &gt &nbsp &iexcl &cent ' \
                       '&pound &curren &yen &brvbar &sect &uml &copy &ordf &laquo &not &shy &reg &macr &deg &plusmn &sup2 ' \
                       '&sup3 tune(s) song(s) &lt;a href=&quot;http http wird übersetzt quot BNF unk bingham spart hebrew nucelli svec' \
                       'henebry kutchie ka hamills yanyor kavak hould fado clamper louys enzo angelillo yoshitomo kozlovskis '.split()
        stop_words.extend(my_stopwords)

        # convert nan-values to empty strings
        csv_file = csv_file.fillna("")

        self.remove_single_words(csv_file)

        # - remove stopwords
        # - force lower case
        # - normalize

        for row in list(csv_file.columns.values):
            csv_file[row] = csv_file[row].apply(
                lambda x: [unicodedata.normalize('NFKD', item.lower()).encode('ASCII', 'ignore').decode('utf-8') for
                           item in
                           str(x).split() if item not in stop_words])

        for row in csv_file.columns.values:
            csv_file[row] = csv_file[row].apply(' '.join)



        return csv_file

import os
import sqlite3

import pandas

mypath = os.path.dirname(__file__)
filename = os.path.join(mypath, 'db_importer.py')

df = pandas.read_csv('D:/Dropbox/Dropbox_Uni/Europena_NEW/Europena_Website-v1/test_.csv', sep=";", encoding="utf-8")
conn = sqlite3.connect("D:/Dropbox/Dropbox_Uni/Europena_NEW/Europena_Website-v1/europena/db.sqlite3")
df.to_sql('expertpage_document', conn, if_exists='append', index=False)

conn.close()

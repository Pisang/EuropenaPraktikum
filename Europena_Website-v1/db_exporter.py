import os
import sqlite3

import pandas

mypath = os.path.dirname(__file__)
filename = os.path.join(mypath, 'db_exporter.py')

conn = sqlite3.connect("D:/Dropbox/Dropbox_Uni/Europena_NEW/Europena_Website-v1/europena/db.sqlite3")
df = pandas.read_sql_query('select * from expertpage_document;', con=conn)
df.to_csv('D:/Dropbox/Dropbox_Uni/Europena_NEW/Europena_Website-v1/export_.csv', sep=";", encoding="utf-8")

conn.close()

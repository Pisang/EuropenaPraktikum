from os import path

import gensim

path_to_files = 'D:/Dropbox/Dropbox_Uni/Europena_NEW/'

model = gensim.models.doc2vec.Doc2Vec.load(path.join(path_to_files, 'dec2vec_model.d2v'))
model.

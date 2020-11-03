import os
from nltk.tokenize import word_tokenize 


#def preprocessing(dataset_dir):

    #rawdata_path = os.path.join(dataset_dir, 'metadata100.csv')

    #traindata_dir = os.path.join(dataset_dir, 'traindata_%d' % hps.sampling_rate )

rawdata_path = '/data/sba/HBJ/sr_test/data/stt_raw'
_, _, files = next(os.walk(rawdata_path))
text_num = len(files)

for i in range(text_num):
    f = open(os.path.join(rawdata_path, "chunk%d.txt" %i), 'r')
    data = f.read()
    print(data)

f.close()
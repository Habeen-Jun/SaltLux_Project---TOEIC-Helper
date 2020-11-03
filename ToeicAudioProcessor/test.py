import numpy as np

mylist = ['test1.wav','test2.wav','test3.wav','test4.wav','test5.wav','test6.wav',1]
input_ = np.array_split(mylist, 3)
print(input_)


# print(mylist // 3)
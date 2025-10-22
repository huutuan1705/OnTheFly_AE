import os
import pickle

with open("test_ChairV2.pickle", "rb") as f:
    Image_Array_Test, Sketch_Array_Test, Image_Name_Test, Sketch_Name_Test = pickle.load(f)
    
print(Image_Name_Test)
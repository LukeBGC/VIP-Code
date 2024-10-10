import pandas as pd
import numpy as np
from sklearn import metrics, model_selection, svm, inspection

IMU1_start_pos = 0

StartData1 = pd.read_csv("data/StartData1.csv", )
print(StartData1.head())


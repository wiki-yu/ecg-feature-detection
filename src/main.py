#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pathlib
from detection import Detectors
import sys
import pandas as pd
from openpyxl import load_workbook


# # MIT-BIH
# unfiltered_ecg = pd.read_csv("../data/100.csv")
# unfiltered_ecg = unfiltered_ecg.iloc[:,2][0:15000]
# unfiltered_ecg = unfiltered_ecg[:5000]
# fs = 360

# # ECG.TSV
# unfiltered_ecg_dat = np.loadtxt("../data/ECG.tsv")
# unfiltered_ecg = unfiltered_ecg_dat[:, 0]
# unfiltered_ecg = unfiltered_ecg[:5000]
# fs = 250

# # AFib
# unfiltered_ecg = pd.read_csv("../data/afib.csv")
# unfiltered_ecg = unfiltered_ecg.iloc[:, 1]
# unfiltered_ecg = unfiltered_ecg[:10000]
# fs = 1000

# 12-leads
unfiltered_ecg = pd.read_csv("../data/12-leads.csv")
unfiltered_ecg = unfiltered_ecg.iloc[:, 2][0:15000]
unfiltered_ecg = unfiltered_ecg[:5000]
print(type(unfiltered_ecg))
fs = 500

# Clinical data


detectors = Detectors(fs)
peaks = detectors.ecg_detector(unfiltered_ecg)

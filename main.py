import pandas as pd
import numpy as np

file = pd.ExcelFile('data\DataHW2.xlsx')

#parse the first/default sheet in the Excel
original_data = file.parse(file.sheet_names[0])

#make 2 dataframes of same type for testing and training
training_data = pd.DataFrame(columns=original_data.columns)
testing_data = pd.DataFrame(columns=original_data.columns)

#split the original into 1:3 testing:training
print(original_data.info)
for i, row in original_data.iterrows():
    if(i%4 == 0):
        testing_data = testing_data._append(row, ignore_index=True)
    else:
        training_data = training_data._append(row, ignore_index=True)
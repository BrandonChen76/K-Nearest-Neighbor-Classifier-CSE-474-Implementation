import pandas as pd
import numpy as np

file = pd.ExcelFile('data\DataHW2.xlsx')

#parse the first/default sheet in the Excel
file_data = file.parse(file.sheet_names[0])

#first row is not part of the data
original_data = file_data.iloc[1:]

#make 2 dataframes of same type for testing and training
training_data = pd.DataFrame(columns=original_data.columns)
testing_data = pd.DataFrame(columns=original_data.columns)

#randomize the dataframe
shuffled_original_data = original_data.sample(frac=1)

#split the original into 20:80 testing:training
for i, row in shuffled_original_data.iterrows():
    if(i%5 == 0):
        testing_data = testing_data._append(row, ignore_index=True)
    else:
        training_data = training_data._append(row, ignore_index=True)

print(testing_data.info())
print("-----------------------------------------------------------------------------------------------------------")
print(training_data.info())
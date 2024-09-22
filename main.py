import pandas as pd
import numpy as np

file = pd.ExcelFile('data\DataHW2.xlsx')

#parse the first/default sheet in the Excel
file_data = file.parse(file.sheet_names[0])

print(file_data.head())

#first row is not part of the data
original_data = file_data.iloc[1:]

#make 2 dataframes of same type for testing and training
training_data = pd.DataFrame(columns=original_data.columns)
testing_data = pd.DataFrame(columns=original_data.columns)

#confusion matrix (actual_predicted)
setosa_setosa = 0
setosa_versicolor = 0
setosa_virginica = 0
versicolor_setosa = 0
versicolor_versicolor = 0
versicolor_virginica = 0
virginica_setosa = 0
virginica_versicolor = 0
virginica_virginica = 0

#randomize the dataframe
shuffled_original_data = original_data.sample(frac=1, random_state=None)

#split the original into 20:80 testing:training
for i, row in shuffled_original_data.iterrows():
    if(i%5 == 0):
        testing_data = testing_data._append(row, ignore_index=True)
    else:
        training_data = training_data._append(row, ignore_index=True)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ CHANGE THIS TO CHANGE K in knn +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
K = 15

#start of actual methods

#distance calc between rows specific to this data set
def distance_calc(row1, row2):
    output = np.sqrt(np.square(row1['x1'] - row2['x1']) + np.square(row1['x2'] - row2['x2']) + np.square(row1['x3'] - row2['x3']) + np.square(row1['x4'] - row2['x4']))
    return output

#from array predict class based on k nn
def classify(array, k):
    setosa_count = 0
    versicolor_count = 0
    virginica_count = 0
    prediction_exist = False
    prediction = -1
    for i in range(len(array)):

        #include overflow nn if same as 10th or prediction is tied
        if(array[i][0] == array[k-1][0] or i < k or not prediction_exist):
            if(array[i][1] == 0):
                setosa_count += 1
            elif(array[i][1] == 1):
                versicolor_count += 1
            else:
                virginica_count += 1

            #get prediction when it is done getting the nn
            if(i == k - 1 or (i > k - 1 and array[i][0] != array[k-1][0])):
                if(setosa_count > versicolor_count and setosa_count > virginica_count):
                    prediction = 0
                    prediction_exist = True
                elif(versicolor_count > setosa_count and versicolor_count > virginica_count):
                    prediction = 1
                    prediction_exist = True
                elif(virginica_count > setosa_count and virginica_count > versicolor_count):
                    prediction = 2
                    prediction_exist = True
        else:
            break

    return prediction

#loop through all rows in test
for i, row1 in testing_data.iterrows():

    #record actual class
    actual_class = row1['Class']

    #create array of tuples (neighbor row, distance)
    array_neighbor = []
    for j, row2 in training_data.iterrows():
        curr_training_class = row2['Class']
        distance = distance_calc(row1, row2)
        array_neighbor.append((distance, curr_training_class))

    #sort array
    sorted_array = sorted(array_neighbor)
    
    #get the prediction
    predicted_class = classify(sorted_array, K)

    #enter the data into confustion matrix
    if(actual_class == 0 and predicted_class == 0):
        setosa_setosa += 1
    elif(actual_class == 1 and predicted_class == 1):
        versicolor_versicolor += 1
    elif(actual_class == 2 and predicted_class == 2):
        virginica_virginica += 1
    elif(actual_class == 0 and predicted_class == 1):
        setosa_versicolor += 1
    elif(actual_class == 0 and predicted_class == 2):
        setosa_virginica += 1
    elif(actual_class == 1 and predicted_class == 0):
        versicolor_setosa += 1
    elif(actual_class == 1 and predicted_class == 2):
        versicolor_virginica += 1
    elif(actual_class == 2 and predicted_class == 0):
        virginica_setosa += 1
    elif(actual_class == 2 and predicted_class == 1):
        virginica_versicolor += 1

print("confusion matrix: \n", setosa_setosa, setosa_versicolor, setosa_virginica, "\n", versicolor_setosa, versicolor_versicolor, versicolor_virginica, "\n", virginica_setosa, virginica_versicolor, virginica_virginica)

print("k:", K)

accuracy = (setosa_setosa + versicolor_versicolor + virginica_virginica) / (setosa_setosa + setosa_versicolor + setosa_virginica + versicolor_setosa + versicolor_versicolor + versicolor_virginica + virginica_setosa + virginica_versicolor + virginica_virginica)
print("accuracy:", accuracy)
sens_setosa = (setosa_setosa) / (setosa_setosa + setosa_versicolor + setosa_virginica)
print("sens_setosa:", sens_setosa)
sens_versicolor = (versicolor_versicolor) / (versicolor_setosa + versicolor_versicolor + versicolor_virginica)
print("sens_versicolor:", sens_versicolor)
sens_virginica = (virginica_virginica) / (virginica_setosa + virginica_versicolor + virginica_virginica)
print("sens_virginica:", sens_virginica)
pris_setosa = (setosa_setosa) / (setosa_setosa + versicolor_setosa + virginica_setosa)
print("pris_setosa:", pris_setosa)
pris_versicolor = (versicolor_versicolor) / (setosa_versicolor + versicolor_versicolor + virginica_versicolor)
print("pris_versicolor:", pris_versicolor)
pris_virginica = (virginica_virginica) / (setosa_virginica + versicolor_virginica + virginica_virginica)
print("pris_virginica:", pris_virginica)
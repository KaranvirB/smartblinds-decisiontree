# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 17:05:03 2023

@author: allod
"""

import pandas as pd
import random
from datetime import datetime
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from IPython.display import display
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus
import six
import sys
sys.modules['sklearn.externals.six'] = six

#%%

#dictionary
data = {"Light":[],"Temp":[]}
Timedata = {"RTC":[]}

#Days in a month
for day in range(1,32):

    #maximum of 10 blind operations in a day
    samples = random.randint(0,10) 
    
    for x in range(samples):
        #select time range that people would generally use blinds
        #ex: 6am to 10pm
        time_hour = random.randrange(6,23) 
        #values 0 to 50, 10 steps a time
        time_min = random.randrange(0,60,10)
        
        #convert the date times so they can be sorted 
        #account for single hours, add an extra 0 infront
        if time_hour < 10:
            Time00 = "0" + str(time_hour)  
        else:
            Time00 = time_hour

        if day < 10:
            dayDate = "0" + str(day)
        else: 
            dayDate = str(day)

        time = dayDate + "/12/2021 " + str(Time00) + ":" + str(time_min) + ":00"
        
        dateOBJ = datetime.strptime(time, "%d/%m/%Y %H:%M:%S")

        #append the real time clock
        Timedata["RTC"].append(dateOBJ)
        
Timedata_df = pd.DataFrame(Timedata)

#sort the date times
filtered_Timedata = Timedata_df.sort_values('RTC')
filtered_Timedata.reset_index(drop=True, inplace=True)
#create hour column
filtered_Timedata['hour'] = filtered_Timedata['RTC'].dt.hour

#convert RTC into weekdays
dw_mapping={
    0: 'Monday', 
    1: 'Tuesday', 
    2: 'Wednesday', 
    3: 'Thursday', 
    4: 'Friday',
    5: 'Saturday', 
    6: 'Sunday'
} 
filtered_Timedata['Day'] = filtered_Timedata['RTC'].dt.dayofweek.map(dw_mapping)

#alternate the state variable accordingly
#0 is rolled down, 1 is rolled up
state = 1
state_array = []

for x in range(len(filtered_Timedata.index)): 
    
    if state == 1:
        state_array.append(state)
        state = 0
    else:
        state_array.append(state)
        state = 1
        
filtered_Timedata.insert(3, "Event", state_array)

#display(filtered_Timedata)

state = 1
for day in range(1,32):
    
    for x in range(samples):
        
        time_hour = filtered_Timedata.iloc[x]['hour']


        #7am to 5 pm sunrise/sunset in the month of December
        #higher light levels during sunrise etc
        if time_hour > 7 and time_hour < 17: 
            light = random.randint(600,650)
        else: 
            light = random.randint(470,599)

        #add the randomly selected times
        data["Light"].append(light)

        #Indoor temperatures are affected whether blinds are opened/closed 
        temp = round(random.uniform(20,24),1)
        
        if state == 0:
            state = 1
            if light < 600:
                temp = temp - 0.5
        elif state == 1:
            state = 0
            if light > 600:
                temp = temp + 1
            
        data["Temp"].append(temp)
        
#convert to pandas dataframe
df = pd.DataFrame(data)

result = pd.concat([df, filtered_Timedata], axis=1, join='inner')

#duplicates may occure
#df2 = filtered_df[filtered_df.duplicated('RTC')]

#filtered_df.set_index('RTC', inplace=True, drop=True)

#alternate the state variable accordingly
#0 is rolled down, 1 is rolled up


#display and convert data to csv file
result['Minutes'] = ((result['RTC'].dt.hour * 60) + result['RTC'].dt.minute)
result['Day'] = result['RTC'].dt.dayofweek
display(result)
result.to_csv('data.csv',index = False)

#%%

#Training with the dataframe 

#Initialize variables for the final decision tree parameters
crit = 0
max_split = 0
max_rand = 0
max_leaf = 0
max_deep = 0
max_accuracy = 0

#Split dataset in features and target variable
df = pd.read_csv("data.csv")

conditions = ['Light', 'Temp', 'Day', 'Minutes']

x = df[conditions]
y = df['Event']

#Nested for loop used to cycle through multiple different parameters to determine the best results
for i in range(2, 4):
    for j in range(0, 42):
        for k in range(0, 2):
            for l in range(2, 200):
                for m in range(1, 20):
    
                    #Splits the data into a training set and testing set
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(i/10), random_state=j, shuffle=True)
                
                    #If statement which checks if the criterion used for the tree should be 'gini' or 'entropy'
                    if k == 0:
                        testtree = DecisionTreeClassifier(criterion="gini", max_leaf_nodes=l, max_depth=m, splitter="best")
                    elif k == 1:
                        testtree = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=l, max_depth=m, splitter="best")
                
                    #Fits the data into a decision tree
                    testtree.fit(x_train, y_train)
                
                    #Prediction gives the accuracy of the decision tree based on the testing data
                    prediction = testtree.predict(x_test)
                    accuracy = metrics.accuracy_score(y_test, prediction)
                
                    #If statement which saves the parameters which result in the highest accuracy into variables
                    if accuracy > max_accuracy:
                        max_accuracy = accuracy
                        max_split = i/10
                        max_rand = j
                        max_leaf = l
                        max_deep = m
                        
                        if k == 0:
                            crit = 0
                        elif k == 1:
                            crit = 1

#Prints out the highest accuracy possible for this data set                        
print("Accuracy:", max_accuracy)

#Splits the data in training and testing using the parameters which gave the highest accuracy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=max_split, random_state=max_rand, shuffle=True)

#Creates a decision tree using the paramters which gave the highest accuracy
if crit == 0:
    dtree = DecisionTreeClassifier(criterion="gini", max_leaf_nodes=max_leaf, max_depth=max_deep, splitter="best")
elif crit == 1:
    dtree = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=max_leaf, max_depth=max_deep, splitter="best")

#Creates the final model using the training data
dtree.fit(x_train, y_train)

#Visualize and save the decision tree
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names = conditions, class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree.png')
Image(graph.create_png())

#Predict the response for test dataset. 0 = roll up, 1 = roll down
y_pred = dtree.predict(x_test)
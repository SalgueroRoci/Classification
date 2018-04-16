'''
Programmers: Rocio Salguero
             Andy Nguyen
             Annie Chen

References:
    https://www.kaggle.com/startupsci/titanic-data-science-solutions
    https://www.kaggle.com/minsukheo/titanic-solution-with-sklearn-classifiers
    https://blog.sicara.com/naive-bayes-classifier-sklearn-python-example-tips-42d100429e44
    http://dataaspirant.com/2017/02/01/decision-tree-algorithm-python-with-scikit-learn/
    http://benalexkeen.com/decision-tree-classifier-in-python-using-scikit-learn/

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score
from subprocess import check_call



def bar_chart(feature):
    survived = original[original['Income'] == 1][feature].value_counts()
    dead = original[original['Income'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['>50', '<=50']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.title(feature)
    plt.show()


'''  Retrieve and Fix data  '''
original = pd.read_csv('Dataset.csv', header=None, index_col=None)
original.columns = ['Age', 'Work', 'Edu-Lvl', 'Edu-Years', 'Marriage-Status', 'Occupation', 'Relationship', 'Gender',
                    'Cap-Gain', 'Cap-Loss', 'Hours', 'Income']
X_columns = ['Age', 'Work', 'Edu-Lvl', 'Edu-Years', 'Marriage-Status', 'Occupation', 'Relationship', 'Gender',
                    'Cap-Gain', 'Cap-Loss', 'Hours']
Y_columns = ['Income']

# Number of empty values
# print(original.isnull().sum())
# print(original.describe(include="all"))

original.replace(["?", "? ", " ?", " ? "], np.nan, inplace=True)
original.replace(" <=50K", 0, inplace=True)
original.replace(" >50K", 1, inplace=True)
# creating new column instead of replace
# original["Income_cleaned"]=original["Income"].astype('category')
# original["Income_cleaned"]=original["Income_cleaned"].cat.codes

# # Drop null values
original = original[pd.notnull(original['Work'])]
original = original[pd.notnull(original['Occupation'])]


''' Certain Columns can be grouped into ranges for easier analysis 
    Grouping: Age, Edu-Years, Cap-Gain, Cap-Loss, Hours '''
# Combine Age, Cap-Gain, Cap-Loss, Hours
original.loc[original.Age <= 21, 'Age'] = 0
original.loc[(original.Age > 21) & (original.Age <= 30), 'Age'] = 1
original.loc[(original.Age > 30) & (original.Age <= 50), 'Age'] = 2
original.loc[(original.Age > 50) & (original.Age <= 70), 'Age'] = 3
original.loc[original.Age > 70, 'Age'] = 4

original.loc[original['Cap-Gain'] <= 2000, 'Cap-Gain'] = 0
original.loc[(original['Cap-Gain'] > 2000) & (original['Cap-Gain'] <= 4000), 'Cap-Gain'] = 1
original.loc[(original['Cap-Gain'] > 4000) & (original['Cap-Gain'] <= 6000), 'Cap-Gain'] = 2
original.loc[(original['Cap-Gain'] > 6000) & (original['Cap-Gain'] <= 10000), 'Cap-Gain'] = 3
original.loc[original['Cap-Gain'] > 10000, 'Cap-Gain'] = 4

original.loc[original['Cap-Loss'] <= 1300, 'Cap-Loss'] = 0
original.loc[(original['Cap-Loss'] > 1300) & (original['Cap-Loss'] <= 1600), 'Cap-Loss'] = 1
original.loc[(original['Cap-Loss'] > 1600) & (original['Cap-Loss'] <= 1900), 'Cap-Loss'] = 2
original.loc[(original['Cap-Loss'] > 1900) & (original['Cap-Loss'] <= 2200), 'Cap-Loss'] = 3
original.loc[original['Cap-Loss'] > 2200, 'Cap-Loss'] = 4

original.loc[original.Hours <= 20, 'Hours'] = 0
original.loc[(original.Hours > 20) & (original.Hours <= 40), 'Hours'] = 1
original.loc[(original.Hours > 40) & (original.Hours <= 60), 'Hours'] = 2
original.loc[(original.Hours > 60) & (original.Hours <= 80), 'Hours'] = 3
original.loc[original.Hours > 80, 'Hours'] = 4


''' Analyze Data '''
print(original.head(20))
# print(original.columns.values)
# print("Data shape", original.shape)

# #Info on numerical and categorical values
# print(original.info())
# print(original.describe(include=['O']))
# print(original.describe())

''' Look at percentage fo each category where Income >50K'''
# print(original[['Work', 'Income']].groupby(['Work'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(original[['Edu-Lvl', 'Income']].groupby(['Edu-Lvl'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(original[['Marriage-Status', 'Income']].groupby(['Marriage-Status'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(original[['Occupation', 'Income']].groupby(['Occupation'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(original[['Relationship', 'Income']].groupby(['Relationship'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(original[['Gender', 'Income']].groupby(['Gender'], as_index=False).mean().sort_values(by='Income', ascending=False) )

# print(original[['Age', 'Income']].groupby(['Age'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(original[['Edu-Years', 'Income']].groupby(['Edu-Years'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(original[['Cap-Gain', 'Income']].groupby(['Cap-Gain'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(original[['Cap-Loss', 'Income']].groupby(['Cap-Loss'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(original[['Hours', 'Income']].groupby(['Hours'], as_index=False).mean().sort_values(by='Income', ascending=False) )

# Bar chart distributions
# bar_chart('Work')
# bar_chart('Edu-Lvl')
# bar_chart('Marriage-Status')
# bar_chart('Occupation')
# bar_chart('Relationship')
# bar_chart('Gender')
# bar_chart('Age')
# bar_chart('Edu-Years')
# bar_chart('Cap-Gain')
# bar_chart('Cap-Loss')
# bar_chart('Hours')

''' Convert Category Columns to numerical '''
# See the categorical unique values
# print(original.Work.unique(), '\n', original['Edu-Lvl'].unique(), '\n', original['Marriage-Status'].unique())
# print(original.Occupation.unique(), '\n', original.Relationship.unique(), '\n', original.Gender.unique())

original["Work"] = original["Work"].astype('category')
original["Work"] = original["Work"].cat.codes

original["Edu-Lvl"] = original["Edu-Lvl"].astype('category')
original["Edu-Lvl"] = original["Edu-Lvl"].cat.codes

original["Marriage-Status"] = original["Marriage-Status"].astype('category')
original["Marriage-Status"] = original["Marriage-Status"].cat.codes

original["Occupation"] = original["Occupation"].astype('category')
original["Occupation"] = original["Occupation"].cat.codes

original["Relationship"] = original["Relationship"].astype('category')
original["Relationship"] = original["Relationship"].cat.codes

original["Gender"] = original["Gender"].astype('category')
original["Gender"] = original["Gender"].cat.codes
# See the numerical categorical values
# print(original.Work.unique(), '\n', original['Edu-Lvl'].unique(), '\n', original['Marriage-Status'].unique())
# print(original.Occupation.unique(), '\n', original.Relationship.unique(), '\n', original.Gender.unique())
# print(original.head(10))



''' Naive Bayes '''
nbModel = GaussianNB()
nbModel.fit(original[X_columns], original['Income'])
score = cross_val_score(nbModel, original[X_columns], original['Income'], cv=10)
print(score)
print(round(np.mean(score)*100, 2))


''' Decision Tree '''
X_train, X_test, y_train, y_test = train_test_split(original[X_columns], original[Y_columns], test_size=0.3, random_state=1)
dtModel = tree.DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=10, min_samples_leaf=10)
dtModel.fit(X_train, y_train)
y_predict = dtModel.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
print(round(accuracy*100, 2))
tree.export_graphviz(dtModel, out_file='tree.dot', feature_names=X_columns)
# check_call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])

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

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from subprocess import check_call


def bar_chart(feature):
    survived = data[data['Income'] == 1][feature].value_counts()
    dead = data[data['Income'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['>50', '<=50']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))
    plt.title(feature)
    plt.show()


def findBestMLPParams(grid, X, y):
    # Warning, very slow. Uses an exhausive search
    gridSearch = GridSearchCV(estimator=MLPClassifier(), cv=10, param_grid=grid, n_jobs=4)
    gridSearch.fit(X, y)
    print("Best Parameters: ", gridSearch.best_params_)
    print("Best Score: ", gridSearch.best_score_)


'''  Retrieve and Fix data  '''
data = pd.read_csv('Dataset.csv', header=None, index_col=None)
data.columns = ['Age', 'Work', 'Edu-Lvl', 'Edu-Years', 'Marriage-Status', 'Occupation', 'Relationship', 'Gender',
                'Cap-Gain', 'Cap-Loss', 'Hours', 'Income']
X_columns = ['Age', 'Work', 'Edu-Lvl', 'Edu-Years', 'Marriage-Status', 'Occupation', 'Relationship', 'Gender',
             'Cap-Gain', 'Cap-Loss', 'Hours']
Y_columns = ['Income']

# Number of empty values
# print(data.isnull().sum())
# print(data.describe(include="all"))

data.replace(["?", "? ", " ?", " ? "], np.nan, inplace=True)
data.replace(" <=50K", 0, inplace=True)
data.replace(" >50K", 1, inplace=True)
# creating new column instead of replace
# data["Income_cleaned"]=data["Income"].astype('category')
# data["Income_cleaned"]=data["Income_cleaned"].cat.codes

# # Drop null values
data = data[pd.notnull(data['Work'])]
data = data[pd.notnull(data['Occupation'])]

''' Certain Columns can be grouped into ranges for easier analysis 
    Grouping: Age, Edu-Years, Cap-Gain, Cap-Loss, Hours '''
# Combine Age, Cap-Gain, Cap-Loss, Hours
data.loc[data.Age <= 21, 'Age'] = 0
data.loc[(data.Age > 21) & (data.Age <= 30), 'Age'] = 1
data.loc[(data.Age > 30) & (data.Age <= 50), 'Age'] = 2
data.loc[(data.Age > 50) & (data.Age <= 70), 'Age'] = 3
data.loc[data.Age > 70, 'Age'] = 4

data.loc[data['Cap-Gain'] <= 2000, 'Cap-Gain'] = 0
data.loc[(data['Cap-Gain'] > 2000) & (data['Cap-Gain'] <= 4000), 'Cap-Gain'] = 1
data.loc[(data['Cap-Gain'] > 4000) & (data['Cap-Gain'] <= 6000), 'Cap-Gain'] = 2
data.loc[(data['Cap-Gain'] > 6000) & (data['Cap-Gain'] <= 10000), 'Cap-Gain'] = 3
data.loc[data['Cap-Gain'] > 10000, 'Cap-Gain'] = 4

data.loc[data['Cap-Loss'] <= 1300, 'Cap-Loss'] = 0
data.loc[(data['Cap-Loss'] > 1300) & (data['Cap-Loss'] <= 1600), 'Cap-Loss'] = 1
data.loc[(data['Cap-Loss'] > 1600) & (data['Cap-Loss'] <= 1900), 'Cap-Loss'] = 2
data.loc[(data['Cap-Loss'] > 1900) & (data['Cap-Loss'] <= 2200), 'Cap-Loss'] = 3
data.loc[data['Cap-Loss'] > 2200, 'Cap-Loss'] = 4

data.loc[data.Hours <= 20, 'Hours'] = 0
data.loc[(data.Hours > 20) & (data.Hours <= 40), 'Hours'] = 1
data.loc[(data.Hours > 40) & (data.Hours <= 60), 'Hours'] = 2
data.loc[(data.Hours > 60) & (data.Hours <= 80), 'Hours'] = 3
data.loc[data.Hours > 80, 'Hours'] = 4

''' Analyze Data '''
print(data.head(10))
# print(data.columns.values)
# print("Data shape", data.shape)

# #Info on numerical and categorical values
# print(data.info())
# print(data.describe(include=['O']))
# print(data.describe())

''' Look at percentage fo each category where Income >50K'''
# print(data[['Work', 'Income']].groupby(['Work'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(data[['Edu-Lvl', 'Income']].groupby(['Edu-Lvl'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(data[['Marriage-Status', 'Income']].groupby(['Marriage-Status'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(data[['Occupation', 'Income']].groupby(['Occupation'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(data[['Relationship', 'Income']].groupby(['Relationship'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(data[['Gender', 'Income']].groupby(['Gender'], as_index=False).mean().sort_values(by='Income', ascending=False) )

# print(data[['Age', 'Income']].groupby(['Age'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(data[['Edu-Years', 'Income']].groupby(['Edu-Years'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(data[['Cap-Gain', 'Income']].groupby(['Cap-Gain'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(data[['Cap-Loss', 'Income']].groupby(['Cap-Loss'], as_index=False).mean().sort_values(by='Income', ascending=False) )
# print(data[['Hours', 'Income']].groupby(['Hours'], as_index=False).mean().sort_values(by='Income', ascending=False) )

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
# print(data.Work.unique(), '\n', data['Edu-Lvl'].unique(), '\n', data['Marriage-Status'].unique())
# print(data.Occupation.unique(), '\n', data.Relationship.unique(), '\n', data.Gender.unique())

data["Work"] = data["Work"].astype('category')
data["Work"] = data["Work"].cat.codes

data["Edu-Lvl"] = data["Edu-Lvl"].astype('category')
data["Edu-Lvl"] = data["Edu-Lvl"].cat.codes

data["Marriage-Status"] = data["Marriage-Status"].astype('category')
data["Marriage-Status"] = data["Marriage-Status"].cat.codes

data["Occupation"] = data["Occupation"].astype('category')
data["Occupation"] = data["Occupation"].cat.codes

data["Relationship"] = data["Relationship"].astype('category')
data["Relationship"] = data["Relationship"].cat.codes

data["Gender"] = data["Gender"].astype('category')
data["Gender"] = data["Gender"].cat.codes


# See the numerical categorical values
# print(data.Work.unique(), '\n', data['Edu-Lvl'].unique(), '\n', data['Marriage-Status'].unique())
# print(data.Occupation.unique(), '\n', data.Relationship.unique(), '\n', data.Gender.unique())
# print(data.head(10))

def main():
    X = data[X_columns]
    y = data['Income']

    ''' Naive Bayes '''
    nbModel = GaussianNB()
    nbModel.fit(X, y)
    score = cross_val_score(nbModel, X, y, cv=10)
    print(score)
    print("GaussianNB Score:", round(np.mean(score) * 100, 2))
    y_pred = nbModel.predict(X)
    print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
        .format(
        data.shape[0],
        (y != y_pred).sum(),
        100 * (1 - (y != y_pred).sum() / data.shape[0])
    ))
    print("Model prior(Prior probabilities of the classes): ", nbModel.class_prior_, "\nModel class:", nbModel.classes_)
    print("Model mean:\n", nbModel.theta_, "\nModel SD: \n", nbModel.sigma_, "\n")

    # Figure out profiles/test the model
    for col in X_columns:
        nbModel.fit(data[col].values.reshape(-1, 1), data.Income)
        value = sorted(data[col].unique(), key=int)
        # print (sorted(value, key=int) )
        print("\n", col, ": ")
        for val in value:
            print(val, ":", nbModel.predict_proba(val), "class", nbModel.predict(val))



    ''' Decision Tree '''
    dtModel = tree.DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=10, min_samples_leaf=10)
    dtModel.fit(X, y)
    accuracy = cross_val_score(dtModel, X, y, cv=10)
    print("\nDecision Tree Score:", round(np.mean(accuracy) * 100, 2))
    tree.export_graphviz(dtModel, out_file='tree.dot', feature_names=X_columns)
    # check_call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png'])

    ''' Multilayer perceptron '''
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaler = scaler.transform(X)
    # grid = {'hidden_layer_sizes': [125, 150], 'alpha': [0.01, 0.005], 'max_iter': [470, 480]}
    # findBestMLPParams(grid, X_scaler, y)
    mlpModel = MLPClassifier(hidden_layer_sizes=150, alpha=0.01, max_iter=475)
    mlpModel.fit(X_scaler, y)
    mlpAccuracy = cross_val_score(mlpModel, X_scaler, y, cv=10)
    print("\nMLP Score:", round(np.mean(mlpAccuracy) * 100, 2))


if __name__ == '__main__':
    main()
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

plt.style.use('fivethirtyeight')


def prepare_data():


        df = pd.read_csv('flag.data')
        continents = {1:'N.America', 2:'S.America', 3:'Europe', 
                4:'Africa', 5:'Asia', 6:'Oceania'}

        # reaplce continent names
        names = lambda x: continents[x]
        df['Continent_name'] = df.Continent.apply(names)

        # replace colours
        cols = {'black':0, 'blue':1, 'brown':2, 'gold':3, 'green':4, 
                'orange':5, 'red':6, 'white':7}

        colours = lambda x: cols[x]

        df['Topl_col'] = df.Topleft.apply(colours)
        df['Botr_col'] = df.Botright.apply(colours)
        df['Main_col'] = df['Main-hue'].apply(colours)
	
	df = df.sample(frac=1.).reset_index(drop=True)

        # get feature columns
        column_mask = range(5,16) + range(18,28) + [31,32,33]
 
        df_features = df[df.columns[column_mask]]

        X = np.array(df_features)
        Y = np.array(df.Continent.values)

        return df_features, X, Y

def make_chart(df, X, importances, indices, std):

        plt.bar(range(X.shape[1]), importances[indices],
                yerr=std[indices], align="center")

        plt.xticks(range(X.shape[1]), df.columns[indices], rotation='vertical')
        plt.xlim([-1, X.shape[1]])
        plt.show()
        

if __name__ == '__main__':

        df, X, Y = prepare_data()

        # 90% test/train split
        split = len(Y)//10
        Y_train = Y[:9*split]
        Y_test  = Y[-split:]

        X_train = X[:9*split]
        X_test  = X[-split:]

        forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

        forest.fit(X_train, Y_train)

	acc = accuracy_score(Y_test, forest.predict(X_test))
	print("Accuracy on test set: %.2f%%" %(100*acc))

        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
        
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(X.shape[1]):
                print("%d. %s used in %.2f%% of trees" % (f, df.columns[indices[f]], 
                        100*importances[indices[f]]))

        make_chart(df, X, importances, indices, std)

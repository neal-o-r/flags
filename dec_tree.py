import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score


def prepare_data():

        df = pd.read_csv('flag.data')
        continents = {1:'N.America', 2:'S.America', 3:'Europe', 
                4:'Africa', 5:'Asia', 6:'Oceania'}

        # replace continent names with ints
        names = lambda x: continents[x]

        df['Continent_name'] = df.Continent.apply(names)

        cols = {'black':0, 'blue':1, 'brown':2, 'gold':3, 'green':4, 
                'orange':5, 'red':6, 'white':7}

        # replace colour strings with ints
        colours = lambda x: cols[x]

        df['Topl_col'] = df.Topleft.apply(colours)
        df['Botr_col'] = df.Botright.apply(colours)
        df['Main_col'] = df['Main-hue'].apply(colours)

	df = df.sample(frac=1.).reset_index(drop=True)

        # the columns we want to train on
        column_mask = range(7,16) + range(18,28) + [31,32,33]
        df_features = df[df.columns[column_mask]]

        X = np.array(df_features)
        Y = np.array(df.Continent.values)

        return df_features, X, Y



if __name__ == '__main__':

        df, X, Y = prepare_data()

        # 90% test/train split
        split = len(Y)//10
        Y_train = Y[:9*split]
        Y_test  = Y[-split:]

        X_train = X[:9*split]
        X_test  = X[-split:]
        
        
        model = tree.DecisionTreeClassifier()

        model.fit(X_train, Y_train)

        acc = accuracy_score(Y_test, model.predict(X_test))

        print("Prediction Accuracy on test set: %.2f%%" %(100*acc))

        with open("tree.dot", 'w') as f:
                f = tree.export_graphviz(model, out_file=f,
                        feature_names=df.columns)

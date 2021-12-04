# clustering
# https://archive.ics.uci.edu/ml/datasets/wine+quality
# Wine Quality Data Set
# for a quick approach, kmeans and agglomerative_clustering are the right candidates
# i will do a future toy project to plot data into 2d/3d spaces using PCA or t-SNE
# maybe we are able to visualize clusters

import pandas as pd
from sklearn.cluster import estimate_bandwidth
from utils import extract
from pipelines import buildpipe
from paramsearch import grid_matrix, customsearch

# import data
df = pd.read_csv('Wine_Quality_Data.csv')
# define target feature
target = 'color'
# remove irrelevant features
df.drop(columns=["quality"], inplace=True)
# keep where target not NaN
df = df[df[target].notna()]
# extract data columns per type
numeric, categorical, ordinal, binary, label = extract(df.copy(), ordinal_features=None, target=target)
# define x and y
X, y = df.drop(columns=target), df[target]
# list initialization
best_param_res = []
models = []
names = []
# loop over all evaluated models
for name, model, param_grid in grid_matrix():
    # build main pipe
    mainpipe = buildpipe(name, model, numeric, categorical, ordinal, binary)
    # custom quick search of hyperparameters
    best_param = customsearch(X, name, model, param_grid, mainpipe, n_iter=100)
    # store variables
    best_param_res.append(best_param)
    names.append(name)
    models.append(model)
# in real-world industry, true labels are likely to be unknown
# in our case, we know the real labels so we check the performances below
for name, model, best_param in list(zip(names, models, best_param_res)):
    mainpipe = buildpipe(name, model, numeric, categorical, ordinal, binary)
    model.set_params(**best_param)
    print(name + " fitting for best parameters : " + str(best_param))
    model = mainpipe.fit(X)
    try:
        labels = model.predict(X)
    except:
        labels = model[name].labels_
    df[name] = labels
    # compare clusters to original target
    print(df[[target, name]]
     .groupby([name, target])
     .size()
     .to_frame()
     .rename(columns={0: 'number'}))






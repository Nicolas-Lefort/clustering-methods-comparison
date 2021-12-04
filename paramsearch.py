from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

from random import randrange
import numpy as np

def customsearch(X, name, model, param_grid, mainpipe, n_iter):
    '''
    the customsearch builds n_iter different param_grid to test.
    if a param_grid is already tested, it is skipped.
    if after n={thresold_try} iterations, no new combination is found, the search ends.
    the models are evaluated using silhouette score.
    the parameters with the highest silhouette score are returned.
    :param:
        X -> input feature matrix (pd.Dataframe or np.array)
        name -> given name of the evaluated model (string)
        model -> the clustering model (object from sklearn.cluster)
        param_grid -> param grid as we would give it to a grid search (dict)
        mainpipe -> main data pipeline preprocessing+ML model (sklearn.pipeline)
        n_iter -> number of max iterations (int)
    :return:
        best_param -> best hyperparameters (dict)
        example: best_param = {
                    'n_clusters': 10,
                    'linkage': "ward"}
    '''

    best_param = {}
    tested_param_grid = []
    best_silhouette = -np.inf
    best_calinski = - np.inf
    i = 0
    k = 0
    thresold_try = n_iter
    flag = False
    print("Searching param for "+name+'...')
    while i <= n_iter and flag==False:
        model_iter = model
        param_grid_random = random_param_grid(param_grid)
        while param_grid_random in tested_param_grid:
            if k > thresold_try:
                flag = True
                break
            param_grid_random = random_param_grid(param_grid)
            k += 1
        model_iter.set_params(**param_grid_random)
        model_iter = mainpipe.fit(X)
        try:
            labels = model_iter.predict(X)
        except:
            labels = model_iter[name].labels_
        try:
            silhouette_sc = silhouette_score(X, labels)
        except:
            silhouette_sc = -np.inf
        try:
            calinski_sc = calinski_harabasz_score(X, labels)
        except:
            calinski_sc = -np.inf
        if silhouette_sc > best_silhouette and calinski_sc > best_calinski:
            best_silhouette = silhouette_sc
            best_param = param_grid_random.copy()
        tested_param_grid.append(param_grid_random)
        i = i + 1
    return best_param

def random_param_grid(param_grid) -> dict:
    '''
    :param:
        param_grid -> param grid as we would give it to a grid search (dict)
    :return:
        param_grid_random -> randomly built grid search from the given choices/lists (dict)

    example:
    input: param_grid_agg = {
                'n_clusters':range(2, 10),
                'linkage': ["ward", "single", "complete"]}
    output: param_grid_agg = {
                'n_clusters': 3,
                'linkage': "ward"}
    '''
    param_grid_random = {}
    for param in param_grid:
        values = param_grid[param]
        randomparam = values[randrange(len(values))]
        param_grid_random.update({param: randomparam})
    return param_grid_random

def grid_matrix():

    param_grid_km = {
                'n_clusters': range(2,10),
    }
    model_km = KMeans()

    param_grid_agg = {
                'n_clusters':range(2, 10),
                'linkage': ["ward", "single"],
    }
    model_agg = AgglomerativeClustering(compute_full_tree=True)

    param_grid_dbscan = {
                'eps': [0.5, 2, 5, 10, 20],
                'min_samples': [100, 150, 200, 250, 300, 350],
                'metric': ["manhattan"],
    }
    model_dbscan = DBSCAN()

    param_mshift = {
                "bandwidth": [10, 40, 50, 100, 200, 300, 500],
    }
    model_mshift = MeanShift(bin_seeding=False)

    param_grid = [param_grid_km, param_grid_agg, param_grid_dbscan, param_mshift]
    model = [model_km, model_agg, model_dbscan, model_mshift]
    names = ["k_means", "agglomerative_clustering", "dbscan", "mean_shift"]

    matrix = list(zip(names, model, param_grid))

    return matrix



import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer

def buildpipe(name, model, numeric, categorical, ordinal, binary):
    '''
    Pipe builder for data preprocessing till model
    :param:
        name -> given name of the evaluated model (string)
        model -> the clustering model (object from sklearn.cluster)
        numeric -> list of numeric feature names (list)
        categorical -> list of categorical feature names (list)
        ordinal -> list of ordinal feature names (list)
        binary -> list of binary feature names (list)
    :return:
        mainpipe -> main data pipeline preprocessing+ML model (sklearn.pipeline)
    '''
    # log transform on skewed features
    Logtransformer = FunctionTransformer(np.log1p)
    # sub pipeline numeric
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('log1p', Logtransformer),
        ('scaler', StandardScaler())])
    # sub pipeline categorical
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # sub pipeline ordinal
    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('int', OrdinalEncoder())])
    # sub pipeline binary
    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    # map data types to their pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
            ("ord", ordinal_transformer, ordinal),
            ("bin", binary_transformer, binary)])
    # define main pipe main pipe
    mainpipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        (name, model)])

    return  mainpipe
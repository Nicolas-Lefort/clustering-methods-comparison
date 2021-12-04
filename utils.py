import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def metrics(y_train, y_train_pred, y_test, y_test_pred, delta_t) -> dict:
    '''
    :param:
        y_train -> array of true labels (np.array)
        y_train_pred -> array of predicted labels (np.array)
        y_test -> array of true labels (np.array)
        y_test_pred -> array of predicted labels (np.array)
        delta_t -> execution time (float)
    :return:
        data with metrics (dict)
    '''
    data = {}

    if len(set(y_test)) == 2:
        data = {'acc. train': accuracy_score(y_train, y_train_pred),
                'acc. test': accuracy_score(y_test, y_test_pred),
                'prec. train': precision_score(y_train, y_train_pred),
                'prec. test': precision_score(y_test, y_test_pred),
                'recall train': recall_score(y_train, y_train_pred),
                'recall test': recall_score(y_test, y_test_pred),
                'f1 train': f1_score(y_train, y_train_pred),
                'f1 test': f1_score(y_test, y_test_pred),
                'exec. time': delta_t,
                }

    if len(set(y_test)) > 2:
        data = {'acc. train': accuracy_score(y_train, y_train_pred),
                'acc. test': accuracy_score(y_test, y_test_pred),
                'exec. time': delta_t,
                }

    return data

def measure_error(y_true, y_pred, label) -> pd.Series:
    '''
    :param:
        y_true -> array of true labels (np.array)
        y_pred -> array of predicted labels (np.array)
        label -> name of return serie (str)

    :return:
        pd.series containing metrics
    '''
    if len(set(y_true)) == 2:
        return pd.Series({'accuracy':accuracy_score(y_true, y_pred),
                          'precision': precision_score(y_true, y_pred),
                          'recall': recall_score(y_true, y_pred),
                          'f1': f1_score(y_true, y_pred)},
                          name=label)
    if len(set(y_true)) > 2:
        return pd.Series({'accuracy':accuracy_score(y_true, y_pred)},
                          name=label)

def extract(df, ordinal_features, target) -> list:
    '''
    :param:
        df -> dataframe (pd.dataframe)
        ordinal_features -> ordinal_features (str or list)
        target -> label column to predict (str)

    :return:
        list of :
        numeric -> list of numeric feature names (list)
        categorical -> list of categorical feature names (list)
        ordinal -> list of ordinal feature names (list)
        binary -> list of binary feature names (list)
    '''
    target = [target]
    df.drop(columns=target, inplace=True)
    # initialize ordinal
    ordinal = []
    # retrieve ordinal variables/features
    if ordinal_features is not None:
        ordinal.append(ordinal_features)
    # count number of different unique values for each feature
    df_uniques = pd.DataFrame([[i, len(df[i].unique())] for i in df.columns],
                              columns=['Variable', 'Unique Values']).set_index('Variable')
    # retrieve binary variables/features
    binary = list(df_uniques[df_uniques['Unique Values'] == 2].index)
    # retrieve categorical variables/features
    categorical = list(df.select_dtypes(exclude="number").columns)
    categorical_ = list(df_uniques[(df_uniques['Unique Values'] <= 10) & (df_uniques['Unique Values'] > 2)].index)
    categorical = list(set(categorical + categorical_) - set(ordinal) - set(binary))
    # retrieve numeric variables/features
    numeric = list(set(df.columns) - set(ordinal) - set(categorical) - set(binary))

    print("extracted numerical features :" , numeric)
    print("extracted categorical features :" , categorical)
    print("extracted ordinal features :" , ordinal)
    print("extracted binary features :" , binary)
    print("extracted label :" , target)

    return [
        numeric,
        categorical,
        ordinal,
        binary,
        target
    ]

def encode(df, ordinal_features, target) -> list:
    '''
    :param:
        df -> dataframe (pd.dataframe)
        ordinal_features -> ordinal_features (str or list)
        target -> label column to predict (str)

    :return:
        list of :
        df -> dataframe with encoded features and labels columns (pd.dataframe)
        df[numeric] -> dataframe with encoded numeric features (pd.dataframe)
        df[categorical_dumm] -> dataframe with encoded categorical features (pd.dataframe)
        df[ordinal] -> dataframe with encoded ordinal features (pd.dataframe)
        df[binary] -> dataframe with encoded binary features (pd.dataframe)
    '''
    # exctract features
    numeric, categorical, ordinal, binary, label = extract(df.copy(), ordinal_features, target)
    # encode categories
    df = pd.get_dummies(df, columns=categorical, drop_first=False)
    # get categorical dummies
    categorical = list(set(df.columns) - set(ordinal) - set(numeric) - set(binary))
    # encode ordinal features
    Oe = OrdinalEncoder()
    df[ordinal] = Oe.fit_transform(df[ordinal])
    # encode binary features
    lb = LabelBinarizer()
    for column in binary:
        df[column] = lb.fit_transform(df[column])
    # encode ordinal and numeric features
    mm = MinMaxScaler()
    for column in [ordinal + numeric]:
        df[column] = mm.fit_transform(df[column])
    # encode labels
    Le = LabelEncoder()
    df[target] = Le.fit_transform(df[target])
    # recover low features and fille NaN values
    df_numerical = clean_numerical(df[numeric])
    # update main df
    df.update(df_numerical)
    # drop low features
    corr = get_correlation(df[numeric].join(df[target]), target)
    low_features = get_low_features(corr, threshold=0.1).index.tolist()
    df.drop(columns=low_features, inplace=True)
    # update numercial features
    numeric = list(set(numeric) - set(low_features))

    return [df,
            df[numeric],
            df[categorical],
            df[ordinal],
            df[binary]
            ]

def clean_numerical(df) -> pd.DataFrame:
    '''
    :param:
        df -> dataframe of numeric features (pd.dataframe)
    :return:
        df -> dataframe with imputed values (pd.dataframe)
    '''
    # impute residual missing values
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(df)
    # rebuild dataframe from numpy array
    df = pd.DataFrame(imputer.transform(df), index=df.index, columns=df.columns)

    return df

def get_correlation(df, target) -> pd.Series:
    '''
    :param:
        df -> dataframe input (pd.dataframe)
        target -> label column (str)
    :return:
        corr -> sorted correlation serie in respect with target (pd.series)
    '''
    corr = df.corrwith(df[target]).abs()
    corr.sort_values(ascending=False, inplace=True)

    return corr

def get_low_features(corr, threshold=0.1) -> pd.Series:
    '''
    :param:
        corr -> serie of correlations (pd.Series)
        threshold -> min correlation factor (float)
    :return:
        low_feat -> correlation serie in respect with target (pd.series)
    '''
    low_feat = corr[corr < threshold]

    return low_feat

def get_top_features(corr, top_n=20) -> pd.Series:
    '''
    :param:
        corr -> serie of correlations (pd.Series)
        top_n -> number of features (int)
    :return:
        top_feat -> correlation serie in respect with target (pd.series)
    '''
    top_feat = corr.head(top_n)

    return top_feat

def plot_correlation(df, target, top_n=20)  -> None:
    # get correlation serie
    corr = get_correlation(df, target)
    # retain first 30 with highest correlation
    top_feat = get_top_features(corr=corr, top_n=top_n)
    # plot correlation factors
    sns.heatmap(top_feat.to_frame(),cmap='rainbow',annot=True,annot_kws={"size": 10},vmin=0)
    plt.title("correlation matrix")
    plt.show()
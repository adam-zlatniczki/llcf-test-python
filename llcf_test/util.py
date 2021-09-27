import pandas as pd


def normalize(X, Y, scale="minmax"):
    if scale == "minmax":
        X_prep = ((pd.DataFrame(X) - pd.DataFrame(X).min()) / (pd.DataFrame(X).max() - pd.DataFrame(X).min())).values
        Y_prep = ((pd.DataFrame(Y) - pd.DataFrame(Y).min()) / (pd.DataFrame(Y).max() - pd.DataFrame(Y).min())).values
    elif scale == "rank":
        X_prep = pd.DataFrame(X).rank().values
        Y_prep = pd.DataFrame(Y).rank().values
    else:
        X_prep = X
        Y_prep = Y

    return X_prep, Y_prep

from sklearn.linear_model import LogisticRegression


def build_model(**kwargs):
    params = {"max_iter": 1000, "n_jobs": -1}
    params.update(kwargs)
    return LogisticRegression(**params)

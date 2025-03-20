from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import joblib

class HyperparameterTuner:
    def __init__(self, model, param_grid, scoring='accuracy', cv=5, n_iter=None, random_state=None):
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.n_iter = n_iter
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None

    def tune(self, X, y, method='grid'):
        if method == 'grid':
            search = GridSearchCV(estimator=self.model, param_grid=self.param_grid,
                                  scoring=self.scoring, cv=self.cv)
        elif method == 'random':
            search = RandomizedSearchCV(estimator=self.model, param_distributions=self.param_grid,
                                         n_iter=self.n_iter, scoring=self.scoring, cv=self.cv,
                                         random_state=self.random_state)
        else:
            raise ValueError("Method must be 'grid' or 'random'.")

        search.fit(X, y)
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_
        return self.best_params_, self.best_score_

    def save_best_model(self, filepath):
        joblib.dump(self.model, filepath)

    def load_best_model(self, filepath):
        self.model = joblib.load(filepath)

    def get_best_params(self):
        return self.best_params_

    def get_best_score(self):
        return self.best_score_
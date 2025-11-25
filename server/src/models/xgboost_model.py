import xgboost as xgb

class XGBoostModel:
    def __init__(self, scale_pos_weight=1):
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model.load_model(path)

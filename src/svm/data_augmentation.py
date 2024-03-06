

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


def data_augmentation(data):
    noise_bias = 0.05
    scale = 0.3
    # ノイズの追加
    noise = np.random.randn(*data.shape) * noise_bias
    data = data + noise
    # データのスケーリング(1-self.scale〜1+self.scaleの範囲でスケーリング)
    scale = np.random.rand() * scale*2 + (1-scale)
    data = data * scale
    return data


class CustomAugmentation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([data_augmentation(x) for x in X])

class CustomPipeline(Pipeline):
    def __init__(self, steps, **kwargs):
        super().__init__(steps, **kwargs)

    @property
    def best_estimator_(self):
        return self.named_steps['classifier'].best_estimator_

def apply(classifier):
    pipeline = CustomPipeline([
        ('augmentation', CustomAugmentation()),
        ('classifier', classifier)
    ])
    return pipeline


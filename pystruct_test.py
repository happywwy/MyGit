# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pystruct
from pystruct.datasets import load_letters
import numpy as np

from pystruct.models import ChainCRF
from pystruct.learners import StructuredPerceptron
letters = load_letters()

X, y, folds = letters['data'], letters['labels'], letters['folds']
X, y = np.array(X), np.array(y)

X_train, X_test = X[folds == 1], X[folds != 1]
y_train, y_test = y[folds == 1], y[folds != 1]

model = ChainCRF()
ssvm = StructuredPerceptron(model = model, max_iter = 10)

ssvm.fit(X_train, y_train)


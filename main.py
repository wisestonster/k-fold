# https://www.codeit.kr/learn/courses/machine-learning/3336
# k겹 교차검증

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd

GENDER_FILE_PATH = './datasets/gender.csv'

# 데이터 셋을 가지고 온다
gender_df = pd.read_csv(GENDER_FILE_PATH)

X = pd.get_dummies(gender_df.drop(['Gender'], axis=1)) # 입력 변수를 one-hot encode한다
y = gender_df[['Gender']].values.ravel()

log_model = LogisticRegression(solver='saga', max_iter=2000)
k_fold_score = np.average(cross_val_score(log_model, X, y, cv=5))

# 체점용 코드
print(k_fold_score)
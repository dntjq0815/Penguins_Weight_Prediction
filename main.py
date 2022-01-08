import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

# 결측치 확인 함수
def check_missing_col(dataframe):
    missing_col = []
    counted_missing_col = 0
    for i, col in enumerate(dataframe.columns):
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            counted_missing_col += 1
            print(f'결측치가 있는 칼럼은: {col}입니다')
            print(f'해당 칼럼에 총 {missing_values}개의 결측치가 존재합니다')
            missing_col.append([col, dataframe[col].dtype])
    if counted_missing_col == 0:
        print('결측치가 존재하지 않습니다.')
    return missing_col

# 데이터 확인
train_missing_col = check_missing_col(train_data)

train_preprocessed = train_data.dropna(subset=['Sex'])
train_preprocessed = train_preprocessed.fillna(0)

train_missing_col = check_missing_col(train_preprocessed)

def make_label_map(dataframe):
    label_maps = {}
    for col in dataframe.columns:
        if dataframe[col].dtype == 'object':
            label_map = {'unknown':0}
            for i, key in enumerate(train_data[col].unique()):
                label_map[key] = i + 1
            label_maps[col] = label_map
    return label_maps

def label_encoder(dataframe, label_map):
    for col in dataframe.columns:
        if dataframe[col].dtype == 'object':
            dataframe[col] = dataframe[col].map(label_map[col])
            dataframe[col] = dataframe[col].fillna(label_map[col]['unknown'])
    return dataframe

label_map = make_label_map(train_preprocessed)
labeled_train = label_encoder(train_preprocessed, label_map)

print(label_map)

def RMSE(true, pred):
    score = np.sqrt(np.mean(np.square(true-pred)))
    return score

target = labeled_train["Body Mass (g)"]
feature = labeled_train.drop(['id', 'Body Mass (g)'], axis=1)

lr = LinearRegression()

kfold = KFold(n_splits=5)

cv_rmse = []  # 각 cv회차의 rmse 점수를 계산하여 넣어줄 리스트를 생성합니다. 이후 RMSE값의 평균을 구하기 위해 사용됩니다.
n_iter = 0  # 반복 횟수 값을 초기 설정해줍니다. 이후 프린트문에서 각 교차검증의 회차를 구분하기 위해 사용됩니다.

# K값이 5이므로 이 반복문은 5번 반복하게 됩니다.
for train_index, test_index in kfold.split(feature):  # feautres 데이터를 위에서 지정한 kfold 숫자로 분할합니다. 인덱스 값을 분할해줍니다.
    x_train, x_test = feature.iloc[train_index], feature.iloc[test_index]  # feature로 사용할 값을 나눠진 인덱스값에 따라 설정합니다.
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]  # label로 사용할 값을 나눠진 인덱스값에 따라 설정합니다.

    lr = lr.fit(x_train, y_train)  # 모델 학습
    pred = lr.predict(x_test)  # 테스트셋 예측
    n_iter += 1  # 반복 횟수 1회 증가

    error = RMSE(y_test, pred)  # RMSE 점수를 구합니다.
    train_size = x_train.shape[0]  # 학습 데이터 크기
    test_size = x_test.shape[0]  # 검증 데이터 크기

    print('\n{0}번째 교차 검증 RMSE : {1},  학습 데이터 크기 : {2},  검증 데이터 크기 : {3}'
          .format(n_iter, error, train_size, test_size))
    print('{0}번째 검증 세트 인덱스 : {1}'.format(n_iter, test_index))
    cv_rmse.append(error)

print('\n==> 이 방정식의 평균 에러(RMSE)는 {} 입니다.'.format(np.mean(cv_rmse)))  # 모델의 평균정확도를 확인합니다.

test_missing_col = check_missing_col(test_data)

test_data['Sex'] = test_data['Sex'].fillna("MALE")
test_preprocessed = test_data.fillna(0)

test_missing_col = check_missing_col(test_preprocessed)

labeled_test = label_encoder(test_preprocessed, label_map)

labeled_test = labeled_test.drop(['id'], axis=1)

predict_test = lr.predict(labeled_test)
print(predict_test)

submission = pd.read_csv('dataset/sample_submission.csv')
submission['Body Mass (g)'] = predict_test

submission.to_csv("submission.csv", index=False)
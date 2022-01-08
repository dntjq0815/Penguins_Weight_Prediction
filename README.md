# Penguins_Weight_Prediction
Prediction of the following penguins' body mass <br/><br/>

**Import Libraries**
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
```
<br/>

**Read csv files for training & test**
```python
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')
```
<br/>

**Function of checking missing columns**
```python
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
```
<br/>

**Check missing columns from the train data**
```python
train_missing_col = check_missing_col(train_data)
```
```
결측치가 있는 칼럼은: Sex입니다
해당 칼럼에 총 3개의 결측치가 존재합니다
결측치가 있는 칼럼은: Delta 15 N (o/oo)입니다
해당 칼럼에 총 3개의 결측치가 존재합니다
결측치가 있는 칼럼은: Delta 13 C (o/oo)입니다
해당 칼럼에 총 3개의 결측치가 존재합니다
```
<br/>

**Preprocessing**
```python
train_preprocessed = train_data.dropna(subset=['Sex'])
train_preprocessed = train_preprocessed.fillna(0)
```
If missing column is 'Sex', just deleted it. Column 'Delta 13 C' or 'Delta 15 N' is filled with integer 0.
<br/><br/>

**Check the preprocessed train data**
```python
train_missing_col = check_missing_col(train_preprocessed)
```
```
결측치가 존재하지 않습니다.
```
No more missing columns in train data.
<br/>
Same thing for the test data as well.
```python
test_missing_col = check_missing_col(test_data)

test_data['Sex'] = test_data['Sex'].fillna("MALE")
test_preprocessed = test_data.fillna(0)

test_missing_col = check_missing_col(test_preprocessed)
```
<br/>

**Label Encoding**
```python
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
```

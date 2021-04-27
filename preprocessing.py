import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
args, _ = parser.parse_known_args()
print('Received arguments {}'.format(args))
split_ratio = args.train_test_split_ratio

input_data_path = os.path.join('/opt/ml/processing/input', 'bank-additional-full.csv')
df = pd.read_csv(input_data_path)

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

one_class = df[df['y'] == 'yes']
one_class_count = one_class.shape[0]
zero_class = df[df['y'] == 'no']
zero_class_count = zero_class.shape[0]
zero_to_one_ratio = zero_class_count / one_class_count
print("Ratio: %.2f" % zero_to_one_ratio)

df['no_previous_contact'] = np.where(df['pdays'] == 999,1, 0)

df['not_working'] = np.where(np.in1d(df['job'],
['student', 'retired', 'unemployed']), 1, 0)

df['not_working'] = np.where(np.in1d(df['job'],
['student', 'retired', 'unemployed']), 1, 0)

X_train, X_test, y_train, y_test = train_test_split(
  df.drop('y', axis=1), df['y'], test_size=split_ratio, random_state=0
)

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocess = make_column_transformer(
(StandardScaler(), ['age', 'duration', 'campaign', 'pdays', 'previous'], StandardScaler()),
(OneHotEncoder(sparse=False), ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome'])
)

train_features = preprocess.fit_transform(X_train)
test_features = preprocess.transform(X_test)


train_features_output_path = os.path.join('/opt/ml/processing/train', 'train_features.csv')
train_labels_output_path = os.path.join('/opt/ml/processing/train', 'train_labels.csv')
test_features_output_path = os.path.join('/opt/ml/processing/test', 'test_features.csv')
test_labels_output_path = os.path.join('/opt/ml/processing/test', 'test_labels.csv')


pd.DataFrame(train_features).to_csv(train_features_output_path, header=False, index=False)
pd.DataFrame(test_features).to_csv(test_features_output_path, header=False, index=False)
y_train.to_csv(train_labels_output_path, header=False, index=False)
y_test.to_csv(test_labels_output_path, header=False, index=False)

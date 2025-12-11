import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('/datasets/train_data_us.csv')

df.loc[df['last_price'] > 113000, 'price_class'] = 1
df.loc[df['last_price'] <= 113000, 'price_class'] = 0

df_train, df_valid = train_test_split(df, test_size=0.25, random_state=54321) # split 25% of data to make validation set

features_train = df_train.drop(['last_price', 'price_class'], axis=1)
target_train = df_train['price_class']
features_valid = df_valid.drop(['last_price', 'price_class'], axis=1)
target_valid = df_valid['price_class']

best_score = 0
best_est = 0
for est in range(1, 11): # choose hyperparameter range
    model = RandomForestClassifier(random_state=54321, n_estimators=est) # set number of trees
    model.fit(features_train, target_train) # train model on training set
    score = model.score(features_valid, target_valid) # calculate accuracy score on validation set
    if score > best_score:
        best_score = score # save best accuracy score on validation set
        best_est = est # save number of estimators corresponding to best accuracy score

print("Accuracy of the best model on the validation set (n_estimators = {}): {}".format(best_est, best_score))

final_model = RandomForestClassifier(random_state=54321, n_estimators=best_est) # change n_estimators to get best model
final_model.fit(features_train, target_train)

from Data_Analysis import *
import shap
from sklearn.feature_selection import VarianceThreshold
import json


models = {
        'rfc': RandomForestClassifier(n_estimators=200, random_state=None),
        'lgb': LGBMClassifier(n_estimators=200)
    }

filePath = "./data/death data - s&f.xlsx"
data_x, data_y = read_excel(filePath)

data_x = standardize(data_x)
# print(data_x.columns)
data_x.columns = data_x.columns.astype(str)
features = np.array(data_x.columns)

# ANOVA
qconstant_filter = VarianceThreshold(threshold=data_x.var().mean())
qconstant_filter.fit(data_x)
qconstant_columns = [column for column in data_x.columns if column not in data_x.columns[qconstant_filter.get_support()]] #查看准恒定特征
print(len(qconstant_columns))
# data_x = qconstant_filter.transform(data_x)
data_x = data_x[data_x.columns[qconstant_filter.get_support()]]
print(data_x.shape)

# correlation analysis
correlated_features = set()
correlation_matrix = data_x.corr(method='pearson')

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
print(len(correlated_features))
data_x.drop(labels=correlated_features, axis=1, inplace=True)
print(data_x.shape)

# X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, stratify=data_y)

def shap_sort(data_x, mean_shap, path, model_key):
    scores = defaultdict(list)
    j = 0
    for i in data_x.columns:
        scores[i] = mean_shap[j]
        j += 1
    # print("Features sorted by their score:")
    # print(sorted([(score, feat) for feat, score in scores.items()], reverse=True))

    tf = open(path + model_key + '_score' + '.json', "w")
    json.dump(scores, tf)
    tf.close()

    data_x.columns = data_x.columns.astype(str)
    metabolite_ID_data = np.array(data_x.columns)
    sorted_idx = list(mean_shap.argsort())
    sorted_idx.reverse()
    # print(sorted_idx)

    np.savetxt(path + model_key + '_fea' + '.txt', metabolite_ID_data[sorted_idx], fmt = '%s')
    np.savetxt(path + model_key + '_sco' + '.txt', mean_shap[sorted_idx])

    print(metabolite_ID_data[sorted_idx[0:20]])

model_list = ['rfc', 'lgb']

path = './data/shap/01/'

for model_key in model_list:
    print('\nthe classifier is:', model_key)
    model = models[model_key]
    model.fit(data_x, data_y)

    explainer = shap.TreeExplainer(model, data_x)

    shap_values = explainer.shap_values(data_x)
    print(np.shape(shap_values))
    print(shap_values)

    if model_key == 'rfc':

        mean_shap = np.mean(np.abs(shap_values[1]), axis = 0)
        print(np.shape(mean_shap))
        shap_sort(data_x, mean_shap, path, model_key)

        shap.summary_plot(shap_values[1], data_x, plot_type="bar")
        shap.summary_plot(shap_values[1], data_x)
    else:
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        print(np.shape(mean_shap))
        shap_sort(data_x, mean_shap, path, model_key)

        shap.summary_plot(shap_values, data_x, plot_type="bar")
        shap.summary_plot(shap_values, data_x)

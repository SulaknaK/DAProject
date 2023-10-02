import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import label_binarize


data = pd.read_csv('flights.csv')
print(data.head().to_string())

# Get the number of rows and columns
num_rows, num_cols = data.shape
print('Number of rows:', num_rows)
print('Number of columns:', num_cols)

print(data.info())

print("Missing Values:\n", data.isnull().sum())
data = data.dropna()

# Standardize multiple columns
scaler = StandardScaler()
data[['price', 'distance', 'time']] = scaler.fit_transform(data[['price', 'distance', 'time']])
print(data.head().to_string())

# show data in boxplot
data_melt = pd.melt(data, id_vars=['userCode'], value_vars=['price', 'distance', 'time'], var_name='variable', value_name='value')
sns.boxplot(x='variable', y='value', data=data_melt)

categorical_features = data.select_dtypes(include=['object']).columns
print("Categorical Features:", categorical_features)

# Convert to date/time format
data['date'] = pd.to_datetime(data['date'])
data['date'] = data['date'].apply(lambda x: x.toordinal())


# check flight types
flight_type_counts = data['flightType'].value_counts()
print("Flight Types:",flight_type_counts)

# check agency types
agency_counts = data['agency'].value_counts()
print("Agency Types:", agency_counts)

# Perform one-hot encoding
one_hot_encoded = pd.get_dummies(data[['flightType', 'agency']])
data = pd.concat([data, one_hot_encoded], axis=1)
data = data.drop(['flightType', 'agency'], axis=1)

# check types of cities in two and from coloumns
to_country_counts = data['to'].value_counts()
print(to_country_counts)

# using label encode to 'from' and 'to' columns to encode values in the order of frequency it appears
df_cities = pd.DataFrame({'city': data['from'].tolist() + data['to'].tolist()})
city_counts = df_cities['city'].value_counts()
city_counts_dict = dict(city_counts)
le = LabelEncoder()
data['from_encoded'] = data['from'].map(city_counts_dict)
data['from_encoded'] = le.fit_transform(data['from_encoded'])
data['to_encoded'] = data['to'].map(city_counts_dict)
data['to_encoded'] = le.fit_transform(data['to_encoded'])

# Print the label and encoded value for each city
for city in city_counts.index:
    label = city_counts_dict[city]
    encoded = le.transform([label])[0]
    print(f"{city}: count={label}, Encoded={encoded}")


data.drop(['from', 'to'], axis=1, inplace=True)

print(data.head().to_string())
print(data.info())

# Anomaly detection/Outlier Analysis
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = lof.fit_predict(data.drop('price', axis=1))
outlier_scores = -lof.negative_outlier_factor_
outlier_indices = np.where(y_pred == -1)[0]
print(f"Number of outliers: {len(outlier_indices)}")
print(f"Outlier indices: {outlier_indices}")

# Remove outliers from data
data = data.drop(data.index[outlier_indices])


#ranndom forest Analysis
X_random = data.drop('price', axis=1)
y_random = data['price']
X_train, X_test, y_train, y_test = train_test_split(X_random, y_random, test_size=0.3, random_state=42)

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
# for f in range(X_train.shape[1]):
#     print("%d. Feature %d (%f): %s" % (f + 1, indices[f], importances[indices[f]], X.columns[indices[f]]))


# Perform PCA
pca = PCA(n_components=8, random_state=42)
X_pca = pca.fit_transform(X_random)
pca_explain_varience = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(pca_explain_varience)
n_components_pca = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print("Explained Variance Ratios After PCA:", pca_explain_varience)
print("Cumulative Variance Ratios After PCA:", pca_explain_varience)
plt.plot(range(1, len(pca_explain_varience)+1), pca_explain_varience)
plt.title("PCA")
plt.xlabel('Number of Principal components')
plt.ylabel('Explained variance ratio')
plt.show()

# Perform SVD
svd = TruncatedSVD(n_components=8, random_state=42)
X_svd = svd.fit_transform(X_random)
explained_variances =  svd.explained_variance_ratio_
cumulative_variance_ratio_svd = np.cumsum(explained_variances)
n_components_svd = np.argmax(cumulative_variance_ratio_svd >= 0.95) + 1
print("Explained Variance Ratios After SVD:", explained_variances)
print("Cumulative Variance Ratios After SVD:", pca_explain_varience)
plt.plot(range(1, len(explained_variances)+1), explained_variances, '-o')
plt.title("SVD")
plt.xlabel('Number of Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(np.arange(1, len(explained_variances)+1))
plt.show()

# take the first 10 features
data.drop(['date', 'userCode','travelCode'], axis=1, inplace=True)

# Covariance Matrix
cov_mat = data.cov()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cov_mat, annot=True, cmap='coolwarm', annot_kws={"fontsize": 8}, fmt=".3f")
ax.set_title("Covariance Matrix Heatmap")
ax.figure.tight_layout()

# Calculate Pearson correlation coefficients matrix
corr_matrix = data.corr(method='pearson')
plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Sample Pearson Correlation Coefficients Matrix')
plt.tight_layout()
plt.show()


# for col in data.columns:
#     # Perform t-test
#     t_stat, p_val = stats.ttest_ind(data[col], data['price'])
#     print("T-test result for column '{}': t-statistic = {:.3f}, p-value = {:.3f}".format(col, t_stat, p_val))
#
#     # Perform F-test
#     f_stat, p_val = stats.f_oneway(data[col], data['price'])
#     print("F-test result for column '{}': F-statistic = {:.3f}, p-value = {:.3f}".format(col, f_stat, p_val))

# Stepwise regression and adjusted R-square analysis to find the features
def stepwise_selection(X, y,
                        initial_list=[],
                        threshold_in=0.01,
                        threshold_out = 0.05,
                        verbose=True):
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded, dtype='float64')
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


X = data.drop('price', axis=1)
y = data['price']
result = stepwise_selection(X, y)
print("Stepwise Selected features:",result)
final_model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[result]))).fit()
#
# # Calculate adjusted R-square
n = len(y)
p = len(result)
R2 = final_model.rsquared
adj_R2 = 1 - ((1 - R2) * (n - 1) / (n - p - 1))
#
# # Print the results
print('Final model:')
print(final_model.summary())
print('Adjusted R-square:', adj_R2)
#
# # Create the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)
#
# # Evaluate the model
y_pred = mlr_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print('Linear Regression Model Results:')
print(f'R-squared: {r2:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')

# #  Perform confidence interval analysis
X_train_with_intercept = sm.add_constant(X_train)
model_with_intercept = sm.OLS(y_train, X_train_with_intercept).fit()
print( model_with_intercept.summary())
print("confidence interval analysis:", model_with_intercept.conf_int(alpha=0.05))


# # Perform collinearity analysis using VIF method
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["predictor"] = X.columns
print(vif)


#Classification
X_c = data.drop(['flightType_economic', 'flightType_firstClass', 'flightType_premium'], axis=1)
y_c = data[['flightType_economic', 'flightType_firstClass', 'flightType_premium']]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
y_train_c_lr = y_train_c.idxmax(axis=1)
y_test_c_lr = y_test_c.idxmax(axis=1)

# Desicion tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_c, y_train_c)
y_pred_c_dt = clf.predict(X_test_c)
print("Accuracy Score Decision Tree: ", accuracy_score(y_test_c, y_pred_c_dt))
print(classification_report(y_test_c, y_pred_c_dt))

cm = confusion_matrix(y_test_c.values.argmax(axis=1), y_pred_c_dt.argmax(axis=1))
print("Confusion Matrix: ", cm)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix of Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# logistic regression model
model_lr = LogisticRegression(multi_class='multinomial', solver='saga')
model_lr.fit(X_train_c, y_train_c_lr)
y_pred_c_lr = model_lr.predict(X_test_c)
accuracy_lr = accuracy_score(y_test_c_lr, y_pred_c_lr)
print('Logistic Regression Accuracy:', accuracy_lr)
print(classification_report(y_test_c_lr, y_pred_c_lr))

#plot confusion matrix
cm_lg = confusion_matrix(y_test_c_lr, y_pred_c_lr)
sns.heatmap(cm_lg, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix of Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

y_prob_c_lr = model_lr.predict_proba(X_test_c)
n_classes = y_c.shape[1]
lw = 2
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_c_lr, y_prob_c_lr[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Convert the target variable to binary matrix
y_test_c_lr_bin = label_binarize(y_test_c_lr, classes=model_lr.classes_)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_c_lr_bin[:, i], y_prob_c_lr[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_c_lr_bin.ravel(), y_prob_c_lr.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve
plt.figure()
lw = 2
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression')
plt.legend(loc="lower right")
plt.show()

# KNN model
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train_c, y_train_c)
y_pred_c_knn = model_knn.predict(X_test_c)
accuracy_knn = accuracy_score(y_test_c, y_pred_c_knn)
print("Accuracy of KNN:", accuracy_knn)
print(classification_report(y_test_c, y_pred_c_knn))


# # # SVM classifier with a linear kernel
model_svm = svm.SVC(kernel='linear', verbose=1, max_iter=5000)
model_svm.fit(X_train_c, y_train_c_lr)
y_pred_c_svm = model_svm.predict(X_test_c)
accuracy_svm = accuracy_score(y_test_c_lr, y_pred_c_svm)
print("Accuracy of SVM:", accuracy_svm)
#
# #Naive Bayes
model_nb = GaussianNB()
model_nb.fit(X_train_c, y_train_c_lr)
y_pred_c_nb = model_nb.predict(X_test_c)
accuracy_nb = model_nb.score(X_test_c, y_test_c_lr)
print("Accuracy of Naive Bayes:", accuracy_nb)
print(classification_report(y_test_c_lr, y_pred_c_nb))

cm_nb = confusion_matrix(y_test_c_lr, y_pred_c_nb)
sns.heatmap(cm_nb, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix of Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ROC Curve for Naive Bayes
y_prob_c_nb = model_nb.predict_proba(X_test_c)
fpr_nb = dict()
tpr_nb = dict()
roc_auc_nb = dict()
for i in range(n_classes):
    fpr_nb[i], tpr_nb[i], _ = roc_curve(y_test_c_lr_bin[:, i], y_prob_c_nb[:, i])
    roc_auc_nb[i] = auc(fpr_nb[i], tpr_nb[i])

# Compute micro-average ROC curve and ROC area
fpr_nb["micro"], tpr_nb["micro"], _ = roc_curve(y_test_c_lr_bin.ravel(), y_prob_c_nb.ravel())
roc_auc_nb["micro"] = auc(fpr_nb["micro"], tpr_nb["micro"])

# #Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_c, y_train_c_lr)
y_pred_c_rf = model_rf.predict(X_test_c)
accuracy_rf = accuracy_score(y_test_c_lr, y_pred_c_rf)
print("Accuracy of Random Forest:", accuracy_rf)
print(classification_report(y_test_c_lr, y_pred_c_rf))

# ROC Curve for Random Forest
y_prob_c_rf = model_rf.predict_proba(X_test_c)
fpr_rf = dict()
tpr_rf = dict()
roc_auc_rf = dict()
for i in range(n_classes):
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test_c_lr_bin[:, i], y_prob_c_rf[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])

# Compute micro-average ROC curve and ROC area
fpr_rf["micro"], tpr_rf["micro"], _ = roc_curve(y_test_c_lr_bin.ravel(), y_prob_c_rf.ravel())
roc_auc_rf["micro"] = auc(fpr_rf["micro"], tpr_rf["micro"])



# # neural Network
model_nn = Sequential()
model_nn.add(Dense(10, input_dim=X_train_c.shape[1], activation='relu'))
model_nn.add(Dense(5, activation='relu'))
model_nn.add(Dense(3, activation='softmax'))
model_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_nn.fit(X_train_c, y_train_c, epochs=10, batch_size=64)
_, accuracy_nn = model_nn.evaluate(X_test_c, y_test_c)
print("Accuracy of Neural network:", accuracy_nn)
y_pred_c_nn = model_nn.predict(X_test_c)
print(classification_report(y_test_c, y_pred_c_nn))

plt.figure(figsize=(10,8))
plt.plot(fpr["micro"], tpr["micro"], label='Logistic Regression (AUC = %0.2f)' % roc_auc["micro"])
plt.plot(fpr_nb["micro"], tpr_nb["micro"], label='Naive Bayes (AUC = %0.2f)' % roc_auc_nb["micro"])
plt.plot(fpr_rf["micro"], tpr_rf["micro"], label='Random Forest (AUC = %0.2f)' % roc_auc_rf["micro"])


plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# K mean
features = ['price', 'time', 'distance']
scaler = StandardScaler()
X = scaler.fit_transform(data[features])
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0,  n_init=10).fit(X)
    inertias.append(kmeans.inertia_)
plt.plot(range(1, 11), inertias)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()


# # Apply K-Means algorithm to dataset with optimal number of clusters
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)
data['cluster'] = kmeans.labels_
print("Silhouette Score: ", silhouette_score(X, kmeans.labels_))
print("Cluster Homogeneity: ", metrics.homogeneity_score(y_test_c_lr, kmeans.labels_))
print("Cluster Completeness: ", metrics.completeness_score(y_test_c_lr, kmeans.labels_))


# perform Apriori algorithm
X = data[['from_encoded', 'to_encoded', 'distance', 'time']]
freq_itemsets = apriori(X, min_support=0.01, use_colnames=True)
rules = association_rules(freq_itemsets, metric="lift", min_threshold=1)
print(rules)


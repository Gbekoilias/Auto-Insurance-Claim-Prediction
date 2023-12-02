#split the data into train and test set with 80% training and 20% test
train, test = train_test_split(df, test_size=0.2, random_state=42)
#print the shape of the train and test set
print(train.shape)
print(test.shape)
df.columns
#Define the features and target variable
features = ['Gender', 'Age',
       'First Transaction Date', 'No_Pol', 'Car_Category',
       'Subject_Car_Colour', 'Subject_Car_Make', 'LGA_Name', 'State',
       'ProductName']
labels = 'target'
numerical_features = ['Age', 'No_Pol']
categorical_features = ['Gender','Car_Category','Subject_Car_Colour', 'Subject_Car_Make', 'LGA_Name', 'State','ProductName']
# train test split
X = train[features]
y = train[labels]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)
#LogisticRegression
## define preprocessing for numeric features(scale them)
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])
## define preprocessing for categorical features(encode them)
categorical_transformer = Pipeline(steps=[
    ('label', OneHotEncoder(handle_unknown='ignore'))])
## amalgamate preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])
## create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression())])

## fit the pipeline to train a logistic regression model on the training set
PIPEmodelLR = pipeline.fit(X_train,y_train)
print (PIPEmodelLR)
# Evaluation Metrics
PIPEpredictionsLR = PIPEmodelLR.predict(X_test)
print('Accuracy: ', accuracy_score(y_test,PIPEpredictionsLR ))
print(classification_report(y_test, PIPEpredictionsLR))
# Print the confusion matrix
cm = confusion_matrix(y_test, PIPEpredictionsLR)
print (cm)
LRprobab = PIPEmodelLR.predict_proba(X_test)
# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, LRprobab[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
#DecisionTreeClassifier
## define preprocessing for numeric features(scale them)
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])
## define preprocessing for categorical features(encode them)
categorical_transformer = Pipeline(steps=[
    ('label', OneHotEncoder(handle_unknown='ignore'))])
## amalgamate preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])
## create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', DecisionTreeClassifier())])

## fit the pipeline to train a logistic regression model on the training set
PIPEmodelDTC = pipeline.fit(X_train,y_train)
print (PIPEmodelDTC)
# Evaluation Metrics
PIPEpredictionsDTC = PIPEmodelDTC.predict(X_test)
print('Accuracy: ', accuracy_score(y_test,PIPEpredictionsDTC ))
print(classification_report(y_test, PIPEpredictionsDTC))
# Print the confusion matrix
cm = confusion_matrix(y_test, PIPEpredictionsDTC)
print (cm)
DTCprobab = PIPEmodelDTC.predict_proba(X_test)
# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, DTCprobab[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

#KNearestNeighbour
## define preprocessing for numeric features(scale them)
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])
## define preprocessing for categorical features(encode them)
categorical_transformer = Pipeline(steps=[
    ('label', OneHotEncoder(handle_unknown='ignore'))])
## amalgamate preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])
## create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', KNeighborsClassifier())])

## fit the pipeline to train a logistic regression model on the training set
PIPEmodelKNN = pipeline.fit(X_train,y_train)
print (PIPEmodelKNN)
KNNprobab = PIPEmodelKNN.predict_proba(X_test)
# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, KNNprobab[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
#SupportVectorMachine
## define preprocessing for numeric features(scale them)
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])
## define preprocessing for categorical features(encode them)
categorical_transformer = Pipeline(steps=[
    ('label', OneHotEncoder(handle_unknown='ignore'))])
## amalgamate preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])
## create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', SVC(probability=True))])

## fit the pipeline to train a logistic regression model on the training set
PIPEmodelSVC = pipeline.fit(X_train,y_train)
print (PIPEmodelSVC)
# Evaluation Metrics
PIPEpredictionsSVC = PIPEmodelSVC.predict(X_test)
print('Accuracy: ', accuracy_score(y_test,PIPEpredictionsSVC ))
print(classification_report(y_test, PIPEpredictionsSVC))
# Print the confusion matrix
cm = confusion_matrix(y_test, PIPEpredictionsSVC)
print (cm)
SVCprobab = PIPEmodelSVC.predict_proba(X_test)
# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, SVCprobab[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

#GradientBoostingClassifier
## define preprocessing for numeric features(scale them)
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])
## define preprocessing for categorical features(encode them)
categorical_transformer = Pipeline(steps=[
    ('label', OneHotEncoder(handle_unknown='ignore'))])
## amalgamate preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])
## create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', GradientBoostingClassifier())])

## fit the pipeline to train a logistic regression model on the training set
PIPEmodelGBC = pipeline.fit(X_train,y_train)
print (PIPEmodelGBC)# Evaluation Metrics
PIPEpredictionsGBC = PIPEmodelGBC.predict(X_test)
print('Accuracy: ', accuracy_score(y_test,PIPEpredictionsGBC ))
print(classification_report(y_test, PIPEpredictionsGBC))
# Print the confusion matrix
cm = confusion_matrix(y_test, PIPEpredictionsGBC)
print (cm)
GBCprobab = PIPEmodelGBC.predict_proba(X_test)
# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, GBCprobab[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

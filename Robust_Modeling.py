def split_data():
    X = df.drop(["target",'ID','Policy Start Date','Policy End Date', 'First Transaction Date'],axis=1)
    y = df["target"] 
    return train_test_split(X, y,test_size=0.3,random_state =42)
def resample():
    X_train = split_data()[0]
    y_train = split_data()[2]
    X_train_over, y_train_over = RandomOverSampler(random_state=42).fit_resample(X_train,y_train)
    return X_train_over, y_train_over
def build_model(voting="hard"):
    estimator = [] 
    estimator.append(('LR',  
                      LogisticRegression(solver ='lbfgs',  
                                         multi_class ='multinomial',  
                                         max_iter = 200))) 
    estimator.append(('SVC', SVC(gamma ='auto', probability = True))) 
    estimator.append(('DTC', DecisionTreeClassifier(max_depth=10))) 
    estimator.append(('KNN',KNeighborsClassifier()))
    estimator.append(('RFC', RandomForestClassifier(random_state=42))) 
    estimator.append(('GBC', GradientBoostingClassifier()))

    # Voting Classifier with hard voting 
    # Build Model
    model = make_pipeline(
        OrdinalEncoder(), 
        RobustScaler(),
        VotingClassifier(estimators=estimator, voting=voting) 
    )
    return model
BOMBmodel = build_model()
BOMBmodel
Pipeline(steps=[('ordinalencoder', OrdinalEncoder()),
                ('robustscaler', RobustScaler()),
                ('votingclassifier',
                 VotingClassifier(estimators=[('LR',
                                               LogisticRegression(max_iter=200,
                                                                  multi_class='multinomial')),
                                              ('SVC',
                                               SVC(gamma='auto',
                                                   probability=True)),
                                              ('DTC',
                                               DecisionTreeClassifier(max_depth=10)),
                                              ('KNN', KNeighborsClassifier()),
                                              ('RFC',
                                               RandomForestClassifier(random_state=42)),
                                              ('GBC',
                                               GradientBoostingClassifier())]))])
TOADmodel = build_model()
X_train_over, y_train_over = resample()
TOADmodel.fit(X_train_over, y_train_over)
Pipeline(steps=[('ordinalencoder',
                 OrdinalEncoder(cols=['Car_Category', 'Subject_Car_Colour',
                                      'Subject_Car_Make', 'LGA_Name', 'State',
                                      'ProductName', 'Car_Colour_Make',
                                      'Car_Colour_Category',
                                      'Car_Make_Category', 'Age_Category'],
                                mapping=[{'col': 'Car_Category',
                                          'data_type': dtype('O'),
                                          'mapping': Saloon                      1
JEEP                        2
Pick Up                     3
Shape Of Vehicle Chasis     4
Truck                       5
Station 4 Wheel             6
Bus                         7
Min...
                ('robustscaler', RobustScaler()),
                ('votingclassifier',
                 VotingClassifier(estimators=[('LR',
                                               LogisticRegression(max_iter=200,
                                                                  multi_class='multinomial')),
                                              ('SVC',
                                               SVC(gamma='auto',
                                                   probability=True)),
                                              ('DTC',
                                               DecisionTreeClassifier(max_depth=10)),
                                              ('KNN', KNeighborsClassifier()),
                                              ('RFC',
                                               RandomForestClassifier(random_state=42)),
                                              ('GBC',
                                               GradientBoostingClassifier())]))])
X_train, X_test, y_train, y_test = split_data()
y_pred = TOADmodel.predict(X_test)
print("Test Accuracy:", round(accuracy_score(y_test, y_pred), 2))     
print("Train Accuracy:", round(TOADmodel.score(X_train, y_train), 2))
print(classification_report(y_test,TOADmodel.predict(X_test)))  
TOADmodel.feature_names_in_ 
X_train_over.columns
# Checking to see if the features are same as the training data columns
all(TOADmodel.feature_names_in_) == all(X_train_over.columns)
TOADmodel.named_steps["votingclassifier"].estimators_
for model in TOADmodel.named_steps["votingclassifier"].estimators_:
    if hasattr(TOADmodel, 'feature_importances_'):
        feature_importances_model = TOADmodel.feature_importances_
print(TOADmodel.named_steps)
{'ordinalencoder': OrdinalEncoder(cols=['Car_Category', 'Subject_Car_Colour', 'Subject_Car_Make',
                     'LGA_Name', 'State', 'ProductName', 'Car_Colour_Make',
                     'Car_Colour_Category', 'Car_Make_Category',
                     'Age_Category'],
               mapping=[{'col': 'Car_Category', 'data_type': dtype('O'),
                         'mapping': Saloon                      1
JEEP                        2
Pick Up                     3
Shape Of Vehicle Chasis     4
Truck                       5
Station 4 Wheel             6
Bus                         7
Mini Bus                    8
Motorcycle                  9
Mini Van                   10
Sedan                      11
Wa...
                         'mapping': TOYOTA_Saloon        1
Ford_JEEP            2
Mercedes_Saloon      3
TOYOTA_JEEP          4
Honda_JEEP           5
                    ..
Volkswagen_Wagon    79
Pontiac_Saloon      80
Kia_Mini Van        81
GMC_JEEP            82
NaN                 -2
Length: 83, dtype: int64},
                        {'col': 'Age_Category',
                         'data_type': CategoricalDtype(categories=['Child', 'Young Adult', 'Adult', 'Senior'], ordered=True, categories_dtype=object),
                         'mapping': Child          1
Young Adult    2
Adult          3
Senior         4
NaN            5
dtype: int64}]), 'robustscaler': RobustScaler(), 'votingclassifier': VotingClassifier(estimators=[('LR',
                              LogisticRegression(max_iter=200,
                                                 multi_class='multinomial')),
                             ('SVC', SVC(gamma='auto', probability=True)),
                             ('DTC', DecisionTreeClassifier(max_depth=10)),
                             ('KNN', KNeighborsClassifier()),
                             ('RFC', RandomForestClassifier(random_state=42)),
                             ('GBC', GradientBoostingClassifier())])}                                      

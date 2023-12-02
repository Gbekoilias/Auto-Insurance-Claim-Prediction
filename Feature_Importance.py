from sklearn.inspection import permutation_importance

# Fit your pipeline on the training data
TOADmodel.fit(X_train_over, y_train_over)

# Calculate permutation importances
result = permutation_importance(TOADmodel, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)

# Get feature importances
feature_importances = result.importances_mean

# Create a series with feature names and importances
feat_imp = pd.Series(feature_importances, index=features).sort_values(ascending=False)
# Get feature names from training data
features = X_train_over.columns

# Get feature importances from the model
feature_importances = TOADmodel.named_steps['GBC'].feature_importances_

# Create a series with feature names and importances
feat_imp = pd.Series(feature_importances, index=features).sort_values(ascending=False)
# Get feature names from training data
features = TOADmodel.feature_names_in_

# Get feature importances from the model
feature_importances = TOADmodel.feature_importances_

# Create a series with feature names and importances
feat_imp = pd.Series(feature_importances, index=features).sort_values(ascending=False)
# Get feature names from training data
features = TOADmodel.feature_names_in_

# Create a series with feature names and importances
feat_imp = pd.Series(feature_importances_model,index=features).sort_values(ascending=False)
# Create a horizontal bar plot
plt.figure(figsize=(10,8))
sns.barplot(x=feat_imp.values, y=feat_imp.index, orient='h',color=sns.color_palette()[0])
plt.xlabel("Gini Importance")
plt.ylabel("Features")
plt.title("Feature Importance");
#save all model to disk
filename = 'BOMBmodel.sav'
pickle.dump(BOMBmodel, open(filename, 'wb'))
#save TOADmodel to disk
filename = 'TOADmodel.sav'
pickle.dump(TOADmodel, open(filename, 'wb'))

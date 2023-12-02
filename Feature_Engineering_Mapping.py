df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
df['Policy End Date'] = pd.to_datetime(df['Policy End Date'])

# Policy Duration
df['Policy Duration'] = (df['Policy End Date'] - df['Policy Start Date']).dt.days

# Extracting Year, Month, Day of Week
df['Policy Start Year'] = df['Policy Start Date'].dt.year
df['Policy Start Month'] = df['Policy Start Date'].dt.month
df['Policy Start Day of Week'] = df['Policy Start Date'].dt.dayofweek

bins = [0, 18, 30, 60, 100]
labels = ['Child', 'Young Adult', 'Adult', 'Senior']
df['Age_Category'] = pd.cut(df['Age'], bins=bins, labels=labels)

df['First Transaction Date'] = pd.to_datetime(df['First Transaction Date'])
df['Days Since First Transaction'] = (df['Policy Start Date'] - df['First Transaction Date']).dt.days
df_encoded = pd.get_dummies(df, columns=['ProductName'])
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Age_No_Pol'] = df['Age'] * df['No_Pol']
df.columns
#shape of df
df.shape
df.info()

# Convert datetime feature to number of days since a certain date
df['Policy Start Period'] = (df['Policy Start Date'] - pd.Timestamp('1970-01-01')) / pd.Timedelta('1 day')
df['Policy End Period'] = (df['Policy End Date'] - pd.Timestamp('1970-01-01')) / pd.Timedelta('1 day')
df['First Transaction Date Period'] = (df['First Transaction Date'] - pd.Timestamp('1970-01-01')) / pd.Timedelta('1 day')

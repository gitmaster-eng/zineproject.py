import pandas as pd
df=pd.read_csv('UNSW_NB15_training-set.csv')
df.head()
x=df.drop(columns=['attack_cat','label'])
y=df['attack_cat']
print(x.columns)
print(y.name)
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
categorical_columns=x.select_dtypes(include=['object']).columns
for col in categorical_columns:
    x[col]=encoder.fit_transform(x[col])
y=encoder.fit_transform(y)  
print(x.head()) 
print(y[:5])
import pandas as pd
df=pd.read_csv('UNSW_NB15_testing-set.csv')
x_test=df.drop(columns=['attack_cat','label'])
y_test=df['attack_cat']
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
categorical_columns=x_test.select_dtypes(include=['object']).columns
for col in categorical_columns:
    x_test[col]=encoder.fit_transform(x_test[col])
y_test=encoder.fit_transform(y_test)
print(x_test.head())
print(y_test[:5])
print('work done')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
rf=RandomForestClassifier()
rf.fit(x,y)
print('ok')
y_pred=rf.predict(x_test)
test_accuracy=accuracy_score(y_pred,y_test)
print(f"Testing Accuracy:{test_accuracy*100:.2f}%")
print("predict values:",y_pred[:10])
from sklearn.metrics import f1_score
y_pred=rf.predict(x_test)
f1=f1_score(y_pred,y_test,average='weighted')
print(f'f1 score:{f1:.2f}')
from imblearn.over_sampling import SMOTE    
from collections import Counter
smote=SMOTE(sampling_strategy="auto",random_state=42)
x_resampled,y_resampled=smote.fit_resample(x,y)
rf_model=RandomForestClassifier(n_estimators=100,random_state=42)
rf_model.fit(x_resampled,y_resampled)
y_pred=rf_model.predict(x_test)
accuracy=accuracy_score(y_pred,y_test)
print(f"accuracy after smote:{accuracy*100:.2f}%")
print('complete')
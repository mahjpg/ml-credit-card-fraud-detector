import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
#data loading and class imbalance predicition
df=pd.read_csv("creditcard.csv")
print(df.shape)
print(df.isna().sum().mean())
print(df["Class"].value_counts())
fraud=df["Class"].mean()*100
legid=100-fraud
print("legid=",legid)
print("Fraud=",fraud)
#stratify test train split
x=df.drop("Class",axis=1)
y=df["Class"]
print("x=",x)
print("y=",y)
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
#pipleline with smote
smote=SMOTE(random_state=42)
log_reg=ImbPipeline(steps=[
    ('scalar',StandardScaler()),
    ('smote',smote),
('clf',LogisticRegression(random_state=42,max_iter=1000)),
])
rf=ImbPipeline(steps=[('smote',smote),('clf',RandomForestClassifier(random_state=42,n_estimators=100))])
#model tuning with gridsearchcv
log_reg.fit(x_train,y_train)
y_pred=log_reg.predict(x_test)
print("logistic regression")
print("classification report=",classification_report(y_test,y_pred))
print("confusion_matrix=",confusion_matrix(y_test,y_pred))
print("roc_auc=",roc_auc_score(y_test,y_pred))
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
print("random forest")
print("classification report=",classification_report(y_test,y_pred_rf))
print("confusion_matrix=",confusion_matrix(y_test,y_pred_rf))
print("roc_auc=",roc_auc_score(y_test,y_pred_rf))




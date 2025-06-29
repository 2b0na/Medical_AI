from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 전처리
def transform(df):
    df['in_marriage'] = df['in_marriage'].map({'Y': 1, 'N': 0})
    df['gender'] = df['gender'].map({'M': 1, 'F': 0})
    df['birth_season'] = df['birth_season'].map({'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3})
    if 'low_birthweight' in df.columns:
        df['low_birthweight'] = df['low_birthweight'].map({'Y': 1, 'N': 0})
    return df

# 데이터 
train_data = pd.read_csv("C:/Users/이보나/Desktop/의료AI/train.csv")
test_data = pd.read_csv("C:/Users/이보나/Desktop/의료AI/test_for_student.csv")

#  전처리
train_data = transform(train_data).fillna(0)
test_data = transform(test_data).fillna(0)

# 변수
features = ['in_marriage', 'father_age', 'mother_age', 'ges_week',
            'n_sibling', 'n_sibling_survived', 'parent_y',
            'father_edu', 'mother_edu']

X = train_data[features]
y = train_data['low_birthweight']
X_test = test_data[features]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,
                                                  stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 모델 
rf = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', random_state=42)
logreg = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', random_state=42)
lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, use_label_encoder=False,
                    eval_metric='logloss', random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)

voting_clf = VotingClassifier(estimators=[
    ('rf', rf), ('logreg', logreg), ('lgbm', lgbm), ('xgb', xgb), ('knn', knn)
], voting='soft')

voting_clf.fit(X_train_scaled, y_train)

y_val_proba = voting_clf.predict_proba(X_val_scaled)[:, 1]
threshold = 0.45
y_val_thresh = (y_val_proba >= threshold).astype(int)

acc = accuracy_score(y_val, y_val_thresh)
print(f"Validation Accuracy: {acc:.5f}")
print("\nClassification Report:\n", classification_report(y_val, y_val_thresh, zero_division=1))

cm = confusion_matrix(y_val, y_val_thresh)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
plt.title(f"Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

test_proba = voting_clf.predict_proba(X_test_scaled)[:, 1]
test_pred = (test_proba >= threshold).astype(int)

submission_df = test_data[['ID']].copy()
submission_df['low_birthweight'] = pd.Series(test_pred).map({1: 'Y', 0: 'N'})
submission_df.to_csv("C:/Users/이보나/Desktop/의료AI/Lee_20225494.csv", index=False)

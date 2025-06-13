**저체중 출생아 예측 모델 (Low Birthweight Prediction)**


**개요**
- 실제 의료 데이터를 기반으로 신생아의 저체중 여부(`low_birthweight`)를 예측하기 위한 앙상블 머신러닝 모델
- 출생 관련 요인을 바탕으로 신생아의 출생 체중이 낮을 확률을 사전에 예측함으로써, 조기 개입 및 의료 대응을 가능하게 함


**데이터 구성**
- train.csv: 학습용 데이터 
- test_for_student.csv: 예측 대상 데이터 
- Lee_20225494.csv: 제출 파일 


**모델**

- 데이터 전처리:
  - 범주형 변수 → 숫자형 매핑
  - 결측값 처리: `fillna(0)`
  - `StandardScaler`로 정규화
- 학습/검증 분리: `train_test_split` (20% validation set)
- 모델 앙상블: `VotingClassifier` (soft voting)
  - `RandomForestClassifier`
  - `LogisticRegression`
  - `LGBMClassifier`
  - `XGBClassifier`
  - `KNeighborsClassifier`

하이퍼파라미터

- threshold = 0.45  -> 예측 확률 기준값

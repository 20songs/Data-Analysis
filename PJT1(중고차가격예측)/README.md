# Used_Cars_Price_Prediction

## 순서

1. EDA

2. Preprocessing

   2.1. 연속형 변수 처리

   ​	2.1.1. Datetime

   ​	2.1.2. 결측치, 이상치 [boxplot]

   ​	2.1.3. 분포 > Log scale

   2.2. 범주형 변수 처리

   ​	2.2.1. 범주화

   2.3. Label 분리

   2.4. PCA

   ​	2.4.1 PCA 설명값 확인하기

   2.5. Pipeline

3. Variable Selection

   3.1. Importance Test

   ​	3.1.1. Random Forest

   ​	3.1.2. XGBoost

   3.2. RFECV

   ​	3.2.1. 평가지표 선정 [Ratio]

   ​	3.2.2. 변수 선택

4. Model

   4.1. Cross Validation

   4.2. Campare Model

5. Inference

   5.1. SHAP

## EDA

#### package

```python
# 행렬
import pandas as pd
import numpy as np

# 시각화
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# 출력화면
from IPython.display import set_matplotlib_formats
mpl.rc('font',family='Malgun Gothic')
mpl.rc('axes',unicode_minus=False)
set_matplotlib_formats('retina')

from IPython.core.display import display,HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# DF option
pd.set_option('display.max.colwidth',100)
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

# multiline print
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

# warning 표시 off
import warnings
warnings.filterwarnings("ignore")
```

#### 연속형 / 범주형 변수 확인

```python
# 연속형 변수
print(train.describe().columns)
# 범주형 변수
print(train.describe(include='object').columns)
```

#### 결측치 확인

```python
# 결측치 확인
train.isna().sum()
```

#### Target 변수 분포 확인

```python
# price 분포: Skewed
# Log scale 조정 필요함!

train.price.plot.hist(color='blue',figsize=(8,6),bins=100,range=(0,200), alpha=0.5, edgecolor='k', linewidth=2)
plt.xlabel('price')
plt.ylabel('Frequency')
plt.title('Distribution of Price')
```

* 옵션
  * `color =` bar의 색깔
  * `figsize = ()` 가로 X 세로 크기
  * `bins =` 막대의 개수
  * `range = ()` 값의 범위
  * `alpha =` 투명도
  * `edgecolor=` figure 모서리 색상
  * `facecolor=` 배경 색상

#### Target 변수(Price) 스케일 조정

```python
log_price = np.log(train.price)
```

#### Name 변수 전처리(연속형 변수 전처리를 위한 범주형 변수 전처리 선행)

```python
split = train.loc[:,'name'].str.split(expand=True)

# brand_name 칼럼 추가
train['brand_name'] = split[0]

# brand_name 정리
# car_name은 데이터 전처리가 복잡해 유지

# brand name이 두 글자 이상인 경우
train.loc[train['brand_name'] == 'Land','brand_name'] = 'Land Rover'
train.loc[train['brand_name'] == 'Force','brand_name'] = 'Force One'
train.loc[train['brand_name'] == 'ISUZU', 'brand_name'] = 'Isuzu'
train.loc[train['brand_name'] == 'Smart','brand_name'] = 'Smart Fortwo'

# car_name 생성
name_df = train['name'].str.split(' ')
name_split = name_df.apply(lambda x:pd.Series(x))
two_brands = ['Land','Force','ISUZU','Smart']
train['car_name'] = name_split[0]

# 브랜가 두 글자인 경우 3번째 항목으로 car name 설정
for i in range(len(train['car_name'])):
  if train.loc[i,'car_name'] not in two_brands:
    train.loc[i,'car_name'] = name_split.iloc[i,1]
  else:
    train.loc[i,'car_name'] = name_split.iloc[i,2]
```

* str.split(expand=True)의 기능
  * array 형태의 자료 구조를 DataFrame 형태로 바꿈
* 조건에 해당하는 열 찾기
  * DataFrame.loc[DataFrae['column'] == 조건, 'column']
* apply(labmda x: 원하는 조건)
  * 일회성 함수
  * x를 원하는 조건으로 변화시킴

## Preprocessing

### 연속형 변수

#### year 변수: 현재 2020년 기준으로 정수값으로 전처리

```python
import datetime
now = datetime.datetime.now()

# 현재 날짜 기준으로 정수 계산
train['year'] = train['year'].apply(lambda
                                    x: now.year-x)
```

#### Kilometer 변수: 이상치 제거

```python
# box plot 그리기
red_dia = dict(markerfacecolor='r',marker='D')
plt.boxplot(train['km_driven'],flierprops=red_dia)
plt.title('km_driven boxplot')
plt.show()

# 아웃라이어 확인 (e6 = 10^6)
train.loc[train.km_driven >= 6e6]

# 아웃라이어 속성 찾기
# BMW, X5 기종 중 year와 km_driven, seats 를 통해 값 범위 찾기
# year <= 5, seats =5, km_driven <=100000

train.loc[(train.brand_name=='BMW') & (train.name.str.contains('X5')) & (train.year <=4) & (train.km_driven <= 1e6) & (train.seats == 5)]

# 샘플 2311: price 54.45, km_driven: 17,738
# 샘플 4101: price 57.00, km_driven: 45,000
# 샘플 4614: price 70.00, km_driven: 15,000
# 샘플 5740: price 55.00, km_driven: 21,000

# 타겟 2328: price 65.00 // km_driven: 6,500,000 > 65,000

# km_driven 조정
train.loc[train['km_driven'] == 6500000,'km_driven'] = 65000
```

* pyplot boxplot
  * boxprops = dict(linestyle, linewidth, color)
    * box의 속성(Property)
  * flierprops = dict(marker=, markerfacecolor, marekrsize, linestyle)
    * 점들의 속성
    * marker는 모양
    * markerfacecolor는 마커의 색상
* DataFrame 다중 조건 인덱싱
  * DataFrame.loc[(DataFrame.column == 조건) & (DataFrame.column.str.contains(조건))&(DataFrame < 조건)]

#### Engine, Mileage, Power 

> 단위 제거
>
> 데이터 타입 변경
>
> 결측치 / 이상치 처리

```python
def trans_type(df,column):
  df[column] = df[column].str.split(expand=True)[0]
  df[column] = pd.to_numeric(df[column],errors='coerce')

trans_type(train,'mileage')
trans_type(train,'engine')
trans_type(train,'power')
```

* pd.to_numeric(errors = 'coerce')
  * 숫자가 될 수 없는 문자인 경우 error가 발생함
  * 이 때 문자를 강제로 NaN으로 변경하는 옵션이 errors = 'coerce'

##### Mileage처리 [engine, power의 경우 isnull() 활용]

````python
# 1)
# fuel_type이 Electric인 경우 mileage의 값의 비교가 다른 연료에 비하면 어려움
# Electric 데이터 삭제

train.loc[train['fuel_type'] == 'Electric',:]
train.drop(train.loc[train['fuel_type'] == 'Electric',:].index, axis=0, inplace=True)

# 2)
# null 값은 같은 브랜드-차종의 평균 값으로 대체
# 0.0인 값도 null로 대체

null_mileage_car_name = train.loc[train['mileage']==0.0,'car_name'].unique()

dic_mileage = {}
for name in null_mileage_car_name:
  dic_mileage[name] = round(train.loc[train['car_name'] == name, 'mileage'].mean(), 2)

for i in train.loc[train['mileage'] == 0.0,'car_name'].unique():
  train.loc[(train['mileage']==0.0) & (train['car_name'] == i),'mileage'] = dic_mileage[i]

# 3)
# 다른 차종이 없어 마일리지의 0 값이 평균으로 대체되지 않은 경우
# 브랜드 평균 값으로 대체

null_mileage_car_name = train.loc[train['mileage']==0.0,'brand_name'].unique()

dic_mileage = {}
for name in null_mileage_car_name:
  dic_mileage[name] = round(train.loc[train['brand_name'] == name, 'mileage'].mean(),2)

for i in train.loc[train['mileage'] == 0.0,'brand_name'].unique():
  train.loc[(train['mileage']==0.0) & (train['brand_name'] == i),'mileage'] = dic_mileage[i]

# 4) car_name과 brand_name으로도 처리되지 않은 값은 전체 평균으로 대체
# car_name과 brand_name 

train.loc[train['mileage'] == 0.0,'mileage'] = train['mileage'].mean()
````

* 결측치 처리 방식
  * 모두 제거하는 방식
    * 전기차 데이터의 경우 유사한 속성 확인이 어려워 모두 제거
  * 해당 값의 평균 데이터로 치환하는 방식
    * 비슷한 데이터 속성을 가진 관측치(샘플)들의 평균을 계산하여 할당함
* engine, power의 경우도 동일한 방식으로 결측치 처리
  * 단, 찾을 조건으로 isnull()을 사용함

#### seats 변수

```python
null_seats_car_name = train.loc[train['seats'].isnull(),'car_name'].unique()

dic_seats = {}
for name in null_seats_car_name:
  dic_seats[name] = train[train['car_name'] == name]['seats'].max()

# Estilo = 평균, seats = 5.0으로 처리
del dic_seats['Estilo']
dic_seats['Estilo'] = 5.0 

try :
  for i in train.loc[train['seats'].isnull(),'car_name'].unique():
    train.loc[(train['seats'].isnull()) & (train['car_name'] == i),'seats'] = dic_seats[i]
except KeyError as e:
  pass

# 이상치 처리
# seats = 0 인 name을 확인해 본 후, 해당 name의 차가 없어, car_name을 참고하여 seats를 넣음.

train.loc[train['seats']==0.0,['name','seats']]
train.loc[train['name'] == 'Audi A4 3.2 FSI Tiptronic Quattro','seats']

train.loc[train['seats']==0.0,['name','seats','car_name']]
train.loc[train['car_name'] == 'A4','seats'].head()

# 5개로 확인
train.loc[train['seats']==0.0,'seats'] = 5.0
```

* 결측치 처리 방식
  * 최대값으로 데이터를 치환
* try - except
  * error 발생 시 pass 혹은 일정 기능을 부과하여 error 예외 처리가 가능하도록 만드는 기능

* 이상치 처리 방식
  * seats = 0인 경우는 존재하지 않으므로
  * 유사한 속성의 값을 그대로 대입

#### new_price

```python
# 1) 값 결측 없는거 ->1 ,  값 결측 있는거 -> 0
train["yn_new_price"] = train['new_price'].notnull().astype(int)
```

* 대량의 결측치가 있는 데이터
  * 버리는 방식을 채택할 수도 있고
  * 새로운 가격이 있다는 것은 차량이 재생산된다는 뜻이므로
    * 해당 가격이 있는지 없는지에 따라 중고차량 vs 새차량의 선택지가 발생하는 것으로 해석 가능함

#### price

```python
# price 분포 확인
green_dia = dict(markerfacecolor='g',marker='D')
plt.boxplot(train['price'],flierprops=green_dia)
plt.title('price boxplot')

# price가 160인 차종
# LandRover Range Rover 3.0 Diesel LWB Vogue의 price가 잘못 표기된 것으로 보임

train.drop(train.loc[train['price'] == 160.0, :].index, axis=0,inplace=True)
```

### 범주형 변수

> 주로 범주형 변수의 경우 해당 주제에 대한 리서치 후 도메인 지식을 활용함

#### location 범주화 [인도의 지역 발전 특성 반영]

```python
# location 처리

# 인도의 도시 발달 역사를 기준으로
# mubai/deli/col/chenai -> metro 1
# bangalroo/ hyd/ameda/pune -> metro 2
# zaipuru / cochi/ coinbatro -> metro 3 / emergiging

loc_list = train['location'].unique()
loc_map = {'Mumbai':'metro 1','Delhi':'metro 1','Kolkata':'metro 1','Chennai':'metro 1',
           'Bangalore':'metro 2','Hyderabad':'metro 2','Ahmedabad':'metro 2','Pune':'metro 2',
           'Jaipur':'emerging','Kochi':'emerging','Coimbatore':'emerging'}
train['loc_type'] = train['location'].map(loc_map)

# 시각화
sns.set(rc={'figure.figsize':(5,4)})
sns.countplot(y='location',data=train)
plt.title('Count of Unique Values in location')

# 시각화
sns.catplot(x='loc_type',y='price',data=train)
```

* map
  * 값을 변화시키는 함수
    * map(조건,변화를 원하는 값의 리스트)
    * map(dictionary 자료구조)

* seaborn set
  * 시각화 시 디스플레이 옵션을 정함
* seaborn countplot
  * 개수를 새는 plot
  * 자동으로 하나의 축이 count 역할을 담당하기 때문에 하나의 변수만 허용
  * hue 옵션을 통해 groupby 효과를 나타낼 수 있음
* seaborn catplot
  * 두 가지 변수 사이의 관계를 시각화
* 범주화
  * 인도의 지역 단위를 Metro1 / Metro2 / Emerging으로 나누기 때문에
  * 이를 반영하여 범주화

#### fuel_type 범주화 [gas 도메인 활용]

```python
# fuel_type 처리

train.loc[(train['fuel_type']=='CNG') | (train['fuel_type']=='LPG'),'fuel_type'] = 'Gas'
train.fuel_type.value_counts()
```

* 연료
  * Oil, Diesel, Gas와 같은 형태로 구분할 수 있음
  * LPG와 CNG는 가스 기반의 연료

#### Owner_Type 범주화 [분석가의 뇌피셜]

```python
# third + Fourth & Above  묶기
train.loc[train['owner_type'] == 'Fourth & Above','owner_type'] = 'Third'
train['owner_type'].value_counts()
```

* 오너 타입
  * 해당 차량이 몇 번째 거래되었냐를 나타내는 지표
  * 첫 중고 거래, 두 번째 중고거래, 그 이후의 3단계로 설정
  * 등간 척도가 아니며 가속하여 차이가 발생할 것으로 추정

#### Transmission

* 특별한 전처리 거치지 않음

#### Brand_Name

```python
# brand_name 처리 

# 전체 개수 및 비율 비교
df_brand_name = pd.DataFrame(train.brand_name.value_counts())
df_brand_name.rename(columns={'brand_name':'count'},inplace=True)
total = sum(df_brand_name['count'])
df_brand_name['ratio'] = df_brand_name['count'].apply(lambda x : (x /total ) *100)

# 1) 전체 개수 중 10개 이하인건 기타로 묶기 
sparse_brand_lst= df_brand_name.loc[df_brand_name['count']<10,:].index.values.tolist()
train.loc[train["brand_name"].isin(sparse_brand_lst),"brand_name"] = "sparse_brand"

df_brand_name.loc[df_brand_name['count']<10,:]
```

* rename 옵션
  * `DataFrame.rename(column={'기존 칼럼':'새 칼럼'},inplace=True)`
    * 칼럼 이름을 변경하는 옵션
    * inplace 옵션에 유의
* isin 함수
  * `DatFrame.loc[DataFrae['column'].isin(list),'column']`
    * containment 테스트를 실행해주는 함수

### Label 분리하기

```python
x_df = train[['year', 'km_driven', 'fuel_type', 'transmission', 'owner_type','mileage', 'engine', 'power', 'seats', 'price', 'brand_name','yn_new_price', 'loc_type']].copy()
x = x_df.drop("price", axis=1)
y_labels = x_df["price"].copy()
y_log = np.log(y_labels) + 1
```

* x data
  * 전처리 한 변수 중 사용하고자 하는 변수 선택
    * 리스트 사용
  * .copy()
    * 원본 데이터를 훼손하지 않기 위해 copy() 함수 사용
* y data
  * target 변수 price [pd.Series 형태]
  * log scale
    * price 분포 확인 후 

### PCA

```python
sns.set(rc={'figure.figsize':(8,6)})
cmap = sns.diverging_palette(200, 0)
corr_heatmap = sns.heatmap(x_df.corr(),cmap=cmap, annot=True)
corr_heatmap
```

* heatmap
  * 상관 관계 점수를 보기 위해
    * annot 옵션: 해당 상관관계숫자값을 표기
    * cmap옵션: 색상 표시
  * 상관 관계 점수가 높은 항목들
    * Engine - Power: 0.86
    * Engine - Mileage: 0.63

```python
sns.pairplot(x_df, vars=['engine', 'power','mileage','price'], hue='brand_name')
```

* pairplot
  * 상관 관계 그래프를 보기 위해
    * hue 옵션: 지정된 변수들을 색상으로 표시
    * vars = [] 옵션: 각 항목들 간의 plot 그림을 표시

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
train_for_pca = np.log(x_df[['engine', 'power']])
pca.fit(train_for_pca)
print()
print('singular value :', pca.singular_values_)
print('singular vector :\n', pca.components_.T)
print('공분산의 설명량',pca.explained_variance_ratio_)
print("PC1 정보량 : ",pca.explained_variance_ratio_[0])
```

* PCA
  * 다 차원의 변수를 저 차원으로 치환하는 기법
    * 여러 축의 변수를 projection(투영) 시켜 variance가 가장 높은 축을 채택하고
    * 그에 수직인 방향의 variance인 축 중 분산이 최대인 축을 찾고
    * 1, 2 축에 직교하면서 분산을 최대 보존하는 세번째 축을 찾음
  * SVD와 매우 유사함
  * log를 취한 이유
    * 두 가지 단위의 통일을 위해
    * scale 조정을 하기 전과 후에 따라 설명 가능한 분산량의 왜곡이 발생함
    * 따라서 두 가지 변수를 표준화시켜야 함

### Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

class Custom_Log(BaseEstimator,TransformerMixin) :
  def __init__(self,centering=False) :
    self.centering = centering

  def fit(self,X,y=None):
    return self 
    
  def transform(self,X):
    log_tmp = np.log(X)
    if self.centering:
      centering_log_tmp = log_tmp - log_tmp.mean(axis=0)
      return centering_log_tmp
    else : 
      return log_tmp
  
class Custom_pass(BaseEstimator,TransformerMixin) :
  def __init__(self) :
    pass

  def fit(self,X,y=None):
    return self
    
  def transform(self,X):
    return X
```

* Cutom_Log
  * 연속 형 변수의 scale 조정
  * PCA 시행 시 centering 조건
    * mean을 구하여 모두 mean에서 출발하도록 지정하는 것
    * value - mean [ mean 만큼 평행이동]
* Custom_Pass
  * 기타 처리가 필요 없는 변수

```python
# PCA 파이프라인
pca_num_pipeline = Pipeline([
                             ('Log', Custom_Log(centering=True)),
                             ('PCA', PCA(n_components=1)),
])

# 연속형 변수 파이프라인
num_pipeline = Pipeline([
                         ('Log', Custom_Log(centering=False)),
])

# 전처리 하지 않을 변수 파이프라인
raw_num_pipeline = Pipeline([
    ('raw_data_pass', Custom_pass())
])
```

* Pipeline 연결
  * 각각의 파이프 라인을 연결하여 Full Pipe line으로 만드는 과정

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# pca 변수 배열
pca_num_attribs = ['engine', 'power']
# 연속형 변수 배열
else_num_attribs = ['mileage','km_driven']
# 전처리 하지 않을 변수
raw_num_attribs = ["year"]
# 범주형 타입 변수
cat_attribs = ['fuel_type',"transmission","owner_type",'seats','brand_name','yn_new_price','loc_type']

full_pipeline = ColumnTransformer([
                                   ("pca_num", pca_num_pipeline, pca_num_attribs), # 로그 -> centering -> PCA 
                                   ("else_num", num_pipeline, else_num_attribs), # 로그만
                                   ("orgin_num", raw_num_pipeline, raw_num_attribs), # 그대로
                                   ("cat", OneHotEncoder(sparse=False), cat_attribs), # 원핫인코딩
])

x_prepared = full_pipeline.fit_transform(x)
```

* Full_Pipeline
  * 파이프 라인을 연결하여 설명 변수를 지정한 후 
  * 파이프라인을 통해 한꺼번에 전처리 하는 방식
  * sklearn에서 제공하는 package
  * 범주형 변수의 경우 내장된 OneHotEncoder를 사용하며
    * sparse 옵션을 False로 지정하면
    * 밀집된 행렬을 구할 수 있음
* 실질적인 변환
  * `full_pipeline.fit_transform(x)`

## Variable Selection

#### 칼럼 명

```python
# one hot encoding 변수 리스트
cat_one_hot_attribs = full_pipeline.named_transformers_["cat"].get_feature_names(cat_attribs).tolist()

# 연속형 설명 변수 추가하기
attributes = ["PC1"] + else_num_attribs + raw_num_attribs + cat_one_hot_attribs

# x_prepared의 칼럼 명 변환 
x_df = pd.DataFrame(x_prepared, columns = attributes)
```

* 리스트 연산
  * +는 리스트 간의 extend연산을 수행함

### Importance Test

#### RF

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)
rf.fit(x_prepared, y_log)
rf_importances = rf.feature_importances_

# 중요도 내림차순으로 20개
rf_indices = np.argsort(rf_importances)[::-1][:20]

# 중요도 시각화
plt.figure(figsize=(12,6))
plt.title("Feature importances")
plt.bar(range(len(rf_indices)), rf_importances[rf_indices], color="r", align="center")
plt.xticks(range(len(rf_indices)), np.array(attributes)[rf_indices], rotation='vertical')
plt.xlim([-1, len(rf_indices)])
```

* RandomForest Regressor
  * Tree 모델은 Feature importance를 계산하는 기능을 내장하고 있음
    * `TreeModel.feature_importances_`
  * 20개의 상위 Feature를 시각화

#### XGBoost

```python
import xgboost as xgb

xgb_reg = xgb.XGBRegressor(objective ='reg:squarederror',random_state=42)
xgb_reg.fit(x_prepared,y_log)
xgb_importances = xgb_reg.feature_importances_

# 중요도 내림차순
sorted( zip(xgb_importances, attributes) , reverse = True)
xgb_indices = np.argsort(xgb_importances)[::-1][:20]

plt.figure(figsize=(12,6))
plt.title("Feature importances")
plt.bar(range(len(xgb_indices)), xgb_importances[xgb_indices], color="b", align="center")
plt.xticks(range(len(xgb_indices)), np.array(attributes)[xgb_indices], rotation='vertical')
plt.xlim([-1, len(xgb_indices)])
plt.show()
```

* argsort
  * 정렬 기준에 맞춰 ndex를 반환하는 함수

#### 두 가지 모델의 중요도 비교

```python
import_df = pd.DataFrame({'xgb':xgb_importances,'rf':rf_importances},
						index=attributes) 
import_df.sort_values(by=['rf'],ascending=False)[:20].plot(kind="bar")
```

* import_df 
  * 칼럼 명은 {}을 통해 선언
  * index = attributes 할당

### RFECV

> Recursive Feature Elimination with Cross-Validation

* RFE 방식
  * Backward 방식 중 하나
  * 모든 변수를 포함한 후 반복하여 학습을 진행하며 중요도가 낮은 변수를 하나씩 제거하는 방식

* Univariate Selection
  * T-test, ANOVA 등
    * `from sklearn.feature_selection import SelectKBest, f_classif`
* ExtraTreesClassifier
  * 트리 기반 모델
    * `from skelarn.ensemble import ExtraTreesClassifier`
    * 위에서 사용한 feature importance와 동일함

#### RMSE 대신 RMSE 비율을 목적 함수로 오차 측정

> 100만원에 10만원 예측이 빗나간 경우와 1,000만원의 10만원 예측이 벗어난 경우의 차이를 보정하기 위해 새로운 Score 지정

```python
from sklearn.metrics import mean_squared_error,make_scorer

def mean_absolute_percentage_error_exp(y_true, y_pred): 
    y_pred[y_pred < 0] = 0
    y_true, y_pred = np.exp(np.array(y_true)-1), np.exp(np.array(y_pred)-1)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
scorer = make_scorer(mean_absolute_percentage_error_exp, greater_is_better=False)
```

* New score를 만드는 함수
  * sklearn의 make_scorer 패키지 사용
    * `greater_is_better=False`
    * 계산 된 값은 오차에 해당하므로
    * 값이 작을 수록 좋은 방향성을 가진다는 옵션을 지정
  * True - Pred / True를 기본으로
    * 절대값을 만들어주고, *100으로 비율을 구하는 함수
  * -1
    * 앞서 labeling 에서 가격의 log 값에 +1을 더하였으므로 다시 -1을 조정
    * (-) 값을 방지하기 위해 +1 평행 이동
  * np.exp
    * log 값을 풀어주기 위해 다시 지수함수를 사용

#### RFECV

```python
from sklearn.feature_selection import RFECV

rf = RandomForestRegressor(random_state=42)

# cv=3을 기준으로 step 당 1개의 변수를 제거
selector = RFECV(rf, step = 1, cv = 3, scoring= scorer)
selector.fit(x_df, y_log)

# 변수 선택 후 추리기
x_selected = selector.transform(x_prepared)
selected_features = x_df.columns[np.where(selector.ranking_==1)]
x_selected_df = pd.DataFrame(x_selected, columns = selected_features)

# 시각화, 대략 40개 부근에서 수렴
plt.plot(selector.grid_scores_)
plt.xlabel('Number of Features')
plt.ylabel('mean_absolute_percentage_error'); plt.title('Feature Selection Scores')

# rank==1인 피처 항목의 데이터 프레임
att_rankings = pd.DataFrame({'feature': list(x_df.columns), 'rank': list(selector.ranking_)}).sort_values('rank')
```

* 변수 선택
  * 변수 개수에 따라 grid_scores 시각화
    * 수렴선의 경계값을 눈으로 확인
  * rank 기준으로 변수 개수를 조정
    * rank == 1인 경우 대략 수렴하는 정도에 해당함
  * 차원이 많으면...
    * 공간이 넓어져서 데이터 밀집도가 낮아지며
    * 이에 따라 OverFitting 문제가 발생하게 됨
    * 특히 데이터의 개수가 6천개 밖에 되지 않기 때문에 차원을 줄임

## Modeling

### Cross Validation

```python
from sklearn.model_selection import cross_val_score

model_results = pd.DataFrame(columns = ['model', 'cv_mean', 'cv_std'])

def cv_model(train, train_labels, model, name, model_results=None):
    cv_scores = cross_val_score(model, train, train_labels, cv = 5, scoring=scorer)
    print(f'5 Fold CV Score: {round(cv_scores.mean(), 2)}, std: {round(cv_scores.std(), 2)}')

    if model_results is not None:
        model_results = model_results.append(pd.DataFrame({'model': name, 
                                                           'cv_mean': cv_scores.mean(), 
                                                            'cv_std': cv_scores.std()},
                                                           index = [0]),
                                             ignore_index = True)

        return model_results
```

* Cross_validation
  * CV 점수를 산출하는 함수 생성
  * `cross_val_score` 
    * `model / Train / Train_Label / cv= / scoring=`
  * model_results
    * 모델의 결과를 저장하기 위해 
    * 비어있는 DF를 선언한 후
    * append기능을 활용하여 row를 추가하였음

#### Models

```python
from sklearn.linear_model import LinearRegression,Lasso,SGDRegressor

model_results = cv_model(x_df, y_log, LinearRegression(), 'LR_OLS', model_results)
model_results = cv_model(x_selected_df, y_log, Lasso(alpha=0.01), 'LR_Lasso_0.01', model_results)
model_results = cv_model(x_selected_df, y_log, Lasso(alpha=0.1), 'LR_Lasso_0.1', model_results)
model_results = cv_model(x_selected_df, y_log, SGDRegressor(max_iter=10000, tol=1e-5, penalty=None, random_state=42),
                         'LR_SGD', model_results)
```

* Linear 모델
  * Lasso를 통해 규제 효과

```python
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Linear SVR
param_grid = [{"epsilon" : [0.5,1,1.5]}]
model = LinearSVR(random_state=42)
grid_search = GridSearchCV(model,param_grid,cv=3,scoring=scorer)

model_results = cv_model(x_selected_df, y_log, grid_search,'Linear_SVR', model_results)

# Polynomial SVR
param_grid = [{"C" : [10,0.1]}]
model = SVR(kernel="poly", gamma="auto", degree=2)
grid_search = GridSearchCV(model,param_grid,cv=3,scoring=scorer)

model_results = cv_model(x_selected_df, y_log, grid_search,'Kernal_poly_SVR', model_results)

# RF regressor
model_results = cv_model(x_selected_df, y_log, RandomForestRegressor(random_state=42, n_jobs = -1), 'RF', model_results)

# XGB regressor
param_grid = [ 
              {"max_depth" : [3,8,10], "eta":[0.02,0.05,0.1],
              'subsample': [0.8,1], 'colsample_bytree': [0.7]}
]

xgb_reg = xgb.XGBRegressor(random_state=42,objective ='reg:squarederror')
grid_search = GridSearchCV(xgb_reg,param_grid,cv=3,
                           scoring=scorer)

grid_search.fit(x_selected_df, y_log)
```

* Linear SVR
  * epsilon
    * SVM 머신의 마진을 결정합니다
    * epsilon의 값이 클 수록 마진의 값을 넓게 정의합니다
* SVR
  * kernel
    * PCA가 변수를 줄이는 개념이라면
    * Kernel을 사용하여 차원을 확장합니다
    * 저차원에서 분석하기 어려운 데이터를 고차원으로 확대하여 결정 plane을 찾아냅니다
    * rbf, sigmoid, linear, poly가 있습니다
    * poly는 각 변수를 다항으로 만들어 줍니다.
  * degree
    * poly 선언 시 각 변수를 몇 차원으로 확장할 지 결정합니다.
  * gamma
    * 하나 하나 샘플에 대한 민감도를 결정합니다.
    * 값이 커질 수록 울퉁 불퉁한 Overfitting에 가까워집니다.
    * 값이 작을수록 평평한 Underfitting에 가까워집니다.
  * C
    * 규제 정도의 역수를 결정합니다.
    * C가 작은 경우(=규제 값이 큰 경우) 모델의 제약이 매우 커집니다.
* RandomForest
  * n_jobs=-1
    * 모든 코어를 사용한다는 뜻
* XGBoost
  * max_depth
    * 트리 가지치기를 수행하는 최대 횟수(깊이)를 결정합니다.
  * eta
    * Learning rate(학습률)
  * subsample
    * 트리 가지치기를 위해 데이터를 샘플링할 때 이전 트리의 사용 비율을 조정합니다.
  * colsample_bytree
    * subsample을 만들 때 사용할 칼럼의 비율을 결정합니다.
  * objective
    * 학습하고자 하는 값을 선언합니다.
* grid search
  * 탐색할 Parameter를 dictionary 형태로 지정하여 선언
  * 해당 파라미터를 모두 탐색하며 scorer를 검사하고
  * 이에 해당하는 파라미터 중 best 값을 자동으로 계산해줌
* * 

```python
grid_search.best_params_ # 사용자 지정 중 최적 파라미터
grid_search.best_estimator_ # 전체 중 최적 파라미터
```

```python
model_results = cv_model(x_selected_df, y_log,xgb.XGBRegressor(colsample_bytree=  0.7, eta= 0.02, max_depth= 10, 
                                                       subsample= 1, random_state=42, n_jobs = -1), 'XGB', model_results)
```

* 최적 모델 결정
  * 최적 모델에 해당하는 파라미터를 지정하여 모델을 다시 한 번 학습

#### Compare Model

```python
model_results.set_index('model', inplace = True)
model_results=model_results.sort_values("cv_mean")
(model_results['cv_mean']*(-1)).plot.bar(color = 'orange', figsize = (8, 6),
                                  yerr = list(model_results['cv_std']),
                                 edgecolor = 'k', linewidth = 2)

plt.title('Model Compare Results')
plt.ylabel('MAPE (with error bar)')
plt.xticks(rotation='horizontal')
model_results.reset_index(inplace = True)
```

* 모델 비교를 위한 시각화

  * -1 곱
    * 오차 값을 + 로 만들어주기 위한 연산입니다.
  * MAPE
    * Mean absolute percantage error를 뜻하는 약어
    * 앞서 설정한 비율 RMSE 값을 뜻합니다

  * yerr
    * 오차 막대
    * std_dev만큼 발생할 수 있는 오차를 구간으로 표시하는 옵션입니다.

## Inference

#### 최종 모델 - 실체/예측 차이 분석

```python
from sklearn.model_selection import train_test_split

# 모델 학습 (앞서 설정한 모델 / 파라미터 그대로 적용)
Xt, Xv, yt, yv = train_test_split(x_selected_df,y_log, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(colsample_bytree=  0.7, eta= 0.02, max_depth= 10, subsample= 1, random_state=42, n_jobs = -1)
model.fit(Xt,yt)

# 실제값 / 예측값
real_val = round(np.exp(yv-1),2).to_numpy()
predict_val = np.round(np.exp(model.predict(Xv)-1),2)

# 차이가 큰 요소부터 [::-1]을 통해 내림차순 정렬
miss_idx = np.argsort((real_val - predict_val) / real_val)[::-1][:10].tolist()
pd.DataFrame({"real":real_val[miss_idx], "pred":predict_val[miss_idx]}, 				index=miss_idx)
```

##### SHAP 분석

```python
!pip install shap
import shap

# use Kernel SHAP to explain test set predictions
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_selected_df)

# 변수중요도 그림
shap.summary_plot(shap_values, x_selected_df)
```

* SHAP

  > SHapley Additive exPlanations

  * [참고 Kaggle](https://www.kaggle.com/dansbecker/shap-values)
  * [참고 DOC](https://shap.readthedocs.io/en/latest/)
  * [참고 블로그](https://datanetworkanalysis.github.io/2019/12/23/shap1)
  * 게임 이론 기반의 설명으로 각 변수들이 미치는 영향력을 확인하는 방법
  * Permutation 기반의 해석 방법
    * 하나의 표본에 대해 모든 가능한 조합에 대해 한계 기여도를 계산하여 평균값을 Shapley Value (SHAP와 다름)로 지정
    * 지수 함수의 시간이 걸리므로 샘플링을 하여 시간을 줄인다
    * 어떤 변수의 값에 대해 다른 변수들의 값을 다양하게 조합하여 해당 값의 기여도를 각각 계산하고 이를 평균 계산
  * 법적으로 설명이 필요한 경우에 사용되는 대표적인 방법
    * 모델에 의한 평가를 뒷받침하는 방식
  * 단, 계산량이 많으므로 반복 횟수를 줄이거나 샘플링을 줄인다면
    * LIME 과 같은 방법이 더 적절할 수 있음
    * 이에 대한 대안이 SHAP!
  * 주의 사항
    * Shapley Value는 특성의 기여도를 나타내는 값
      * 입력값의 변화에 따른 예측값 변화를 설명하는 값이 아님
    * SHAP는 예측치를 설명할 수 있음
      * LIME과 Shapley value를 연결하는 역할을 한다
    * 변수 간 높은 correlation에 취약하다

* SHAP 계산

  * TreeExplainer 함수
    * Tree 모형에 대한 SHAP 지수 계산
    * `DeepExplainer` / `KernelExplainer` / `Gradient Explainer` / `Linear Explainer` ...

  * shap_value계산
    * shap_values는 2차원 배열로 계산된다
    * 첫번째 배열은 Negative outcome [가격에 -를 미치는 요소]
    * 두번째 배열은 Positive outcome [가격에 +를 미치는 요소]
  * summary_plot 시각화
    * 변수 중요도를 요약
    * barplot 대신 density plot을 사용하여 표시

```python
# 전체 결과에 대한 해석
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], x_selected_df.iloc[0,:])

# 개별 결과에 대한 해석
shap.initjs()
shap.force_plot(explainer.expected_value,
                shap_values[Xv.iloc[[miss_idx[0]],].index.values,:],
                x_selected_df.iloc[Xv.iloc[[miss_idx[0]],].index.values,:])
                
# year 변수에 대한 shap value 점수
shap.dependence_plot("year", shap_values, x_selected_df, 
                     interaction_index="mileage")

# SHAP 해석을 통해 
train.iloc[Xv.iloc[[miss_idx[0]],].index.values,]
```

* SHAP
  * shap.initjs()
    * java script 를 사용하기 위해 호출
  * shap.force_plot
    * SHAP 결과를 시각화하는 도구
  * dependence_plot
    * Partial dependency에 상호 작용 효과까지 설명하는 기능
    * 수직 분산의 경우 상호 작용 효과에 의해 유도됨
    * 색상은 +,- 방향성을 나타냄
    * 사용법
      * 1) 특성을 선택한다.
      * 2) 각 관측치에 대해 특성 값을 x축에, 해당하는 Shapley value를 y축에 표시한다.
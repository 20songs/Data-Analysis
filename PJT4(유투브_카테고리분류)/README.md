# Youtube_Category_Classification

[toc]

## 순서

1. Crawling (데이터 모음)

2. Preprocessing (데이터 병합)

3. Feature Extraction (특성 추출)

   3.1. Text [Fxt, W2v]

   3.2. Image [VGG, CNN]

4. Combine (특성 결합)

5. Explain (Gradient-Cam)


## 1. Crawling

### Selenium, bs4

* Selenium
  * 웹사이트 테스트를 위한 도구로 사용됨
    * webdriver라는 API를 통해 브라우저를 제어함
  * JavaScript를 통해 비동기적으로 서버로부터 콘텐츠를 가져오거나 숨겨져 있는 콘텐츠를 열람하는 등의 작업을 수행할 수 있음
    * request 라이브러리 등 태그 기반으로 웹 페이지의 컨텐츠를 읽는 것과 다른 방식
    * 마치 사람이 웹페이지를 이용하는 듯
  * 유용성
    * 웹사이트가 프로그램을 통한 접근을 허용하지 않는 경우
    * 로그인을 요구하는 경우
    * 동적 웹페이지로 구성된 경우
* BS4
  * request - response 방식
  * parsing이 가능함
  * HTML 구조 내부에서 element, id 등의 요소를 찾을 수 있음

### 설치옵션

```python
!apt-get update
!pip install selenium
!pip install bs4
!apt install chromium-chromedriver
!pip install lxml
!pip install openpyxl

from selenium import webdriver
from bs4 import BeautifulSoup

import pandas as pd
import time
import re
```

### 7개 카테고리

> 요리, 경제, 정치, 게임, 스포츠, 영화, 애완동물
>
> > 구독자 수가 비슷한 채널을 선정
> >
> > 동영상의 개수를 3500개 정도 모아 balance를 맞추려고 함

```python
# colab 가상환경에서 driver를 사용하기 위한 세팅
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome('chromedriver',options=options)

# 채널 별 동영상 카테고리의 url을 모두 조사하여 삽입
url = list(url_dict.values())[0]
driver.get(url)
```

* 가상환경에서 diver를 사용하려면 chromeoptions를 지정해야 함
  * 위 옵션에 대한 자세한 내용은 블로그를 참조하였음
* driver 실행
  * `driver.get(url)`
  * url 페이지에서 사용자가 조작하듯 드라이버가 브라우저를 이용하여 url을 조회하기 위해 요청을 보냄

```python
# 동적 웹페이지 구성
# 스크롤 동작을 통해 모든 동영상 데이터를 로딩시킴
# 전체 페이지를 모두 조회하기 위해 last_page의 scroll height를 미리 계산함
last_page_height = driver.execute_script("return document.documentElement.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(1.5)       # 인터벌 1이상으로 줘야 데이터 취득가능(롤링시 데이터 로딩 시간 때문)
    new_page_height = driver.execute_script("return document.documentElement.scrollHeight")

    if new_page_height == last_page_height:
        break
    last_page_height = new_page_height
  
# html_source 정보를 저장
html_source = driver.page_source
driver.close()
```

* 스크롤 작업
  * 동적 웹페이지의 경우 페이지의 컨텐츠가 변할 수 있으므로
  * 스크롤 바를 가장 마지막까지 내려 모든 컨텐츠의 동적 변화를 통제함
* Page_Source
  * `driver.page_source`를 통해 html source를 변수에 저장하고
  * dirver는 멈춤

```python
# BeautifulSoup을 통해 parsing
soup = BeautifulSoup(html_source,'lxml')

# 각 동영상의 링크를 얻기 위해 개별 동영상을 담은 list 생성
video_list = soup.find_all('ytd-grid-video-renderer',{'class':'style-scope ytd-grid-renderer'})

# 축약된 url을 전체 url로 변환시키기
base_url = 'https://www.youtube.com'
url_list = []

# thumbnail 에 a 태그로 포함된 href 링크를 찾고 완전한 url로 변환시켜줌
for i in range(len(video_list)):
  full_url = base_url+video_list[i].find('a',{'id':'thumbnail'})['href']
  url_list.append(full_url)
```

* bs4의 BeautifulSoup
  * html 페이지 구성 요소를 tag, id, class 단위로 찾아냄

```python
# 스포츠 채널의 DATAFRAME
sports = pd.DataFrame({'channel_name' :[],
                           'subscribers' : [],
                           'thumbnail' :[],
                           'video_name':[],
                           'video_length':[],
                           'upload_date':[],
                           'hits':[],
                           'likes_num':[],
                           'dislikes_num':[],
                           'category':[] })

# url_list 순회하며 정보 저장
for i in range(len(url_list)):

  
  try:
    # 전체 채널 정보 (채널이름, 구독자)
    # 동영상 url 이전에 얻기 편한 정보 (동영상 제목, 동영상 길이)
    channel_name = soup.find('title').text
    subscribers = soup.find('yt-formatted-string',{'id':'subscriber-count'}).text    
    video_name = video_list[i].find('a',{'id':'video-title'}).text
    video_length = video_list[i].find('span', {'class':'style-scope ytd-thumbnail-overlay-time-status-renderer'}).text.split('\n')[1]
    
    # error 유발 ㅠ, aria-hidden 속성 때문인 것으로 보임
    # thumbnail = video_list[i].find('img', {'src' : True})['src'] 

    # url에 해당하는 동영상 페이지 들어가기
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome('chromedriver',options=options)
    
    # url 삽입
    driver.get(url_list[i])
    
    # 스크롤
    last_page_height = driver.execute_script("return document.documentElement.scrollHeight")
    
    scrolls = 2
    while scrolls:
      driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
      time.sleep(1.5)       # 인터벌 1이상으로 줘야 데이터 취득가능(롤링시 데이터 로딩 시간 때문)
      scrolls -= 1

    # video_source를 parsing
    video_source = driver.page_source
    video_soup = BeautifulSoup(video_source,'lxml')
    driver.close()

    # 정보 가져오기
    # 올린 날짜, 조회수, 좋아요 수, 싫어요 수, 카테고리 정보
    
    scripts = video_soup.find('script',{'id':'scriptTag'}).text.split('"')
    for script in scripts:
      if 'img' in script:
        thumbnail = script
        break
    upload_date = video_soup.find('div',{'id':'date'}).find('yt-formatted-string').text
    hits = video_soup.find('span',{'class':'view-count'}).text.split()[0]
    likes_num = video_soup.find_all('yt-formatted-string',{'id':'text','class':'style-scope ytd-toggle-button-renderer style-text'})[0]['aria-label'].split()[0]
    dislikes_num = video_soup.find_all('yt-formatted-string',{'id':'text','class':'style-scope ytd-toggle-button-renderer style-text'})[1]['aria-label'].split()[0]
    category = 'sports' # 바꾸면됨

    # 임시 데이터 프레임을 생성하여 기존 데이터 프레임에 삽입
    tmp = pd.DataFrame({'channel_name' :[channel_name],
                                    'subscribers' : [subscribers],
                                    'thumbnail' :[thumbnail],
                                    'video_name':[video_name],
                                    'video_length':[video_length],
                                    'upload_date':[upload_date],
                                    'hits' :[hits],
                                    'likes_num':[likes_num],
                                    'dislikes_num':[dislikes_num],
                                    'category':[category]})
    
    # append 이용하여 아래로 concat
    sports = sports.append(tmp)

  # 동영상의 형식이 기본 구조와 다른 경우 error 발생
  # 이 때문에 정보 조회가 되지 않는 경우 예외 처리하여 데이터에서 삭제
  except KeyError as k: 
      print(k, i)
  except AttributeError as a: 
      print(a ,i)
  except IndexError as e:
      print(e,i)
  except TypeError as t:
      print(t, i)
```

* 저장할 DataFrame을 선언
  * `DF.append`
  * concat의 수직 결합 역할을 함
* 각 특성 값에 해당하는 정보를 파싱
  * 일일이 개발자도구[F12]로 확인해야 함
  * 기본적으로 `find('태그',{'속성키':'속성값'})`의 형식으로 찾음
* try and except
  * 동영상 및 채널마다 형식이 다를 수 있으므로
  * error를 모두 처리할 수 없으니 pass 시키시 위한 용도로 사용함

### 만약 로컬에서 사용한다면

```python
# 드라이버 직접 설치
# 드라이버 실행 파일 경로 삽입
path = '@needs_for_you/chromedriver.exe'
driver = webdriver.Chrome(path)
driver.maximize_window()

# 채널-동영상 url
url = 'https://www.youtube.com/user/sbssportsnow/videos'
driver.get(url)

# 동적 웹페이지 구성
# 스크롤 동작을 통해 모든 동영상 데이터를 로딩시킴
# 전체 페이지를 모두 조회하기 위해 last_page의 scroll height를 미리 계산함
last_page_height = driver.execute_script("return document.documentElement.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(1.5)       # 인터벌 1이상으로 줘야 데이터 취득가능(롤링시 데이터 로딩 시간 때문)
    new_page_height = driver.execute_script("return document.documentElement.scrollHeight")

    if new_page_height == last_page_height:
        break
    last_page_height = new_page_height
  
# html_source 정보를 저장
html_source = driver.page_source
driver.close()

# BeautifulSoup을 통해 parsing
soup = BeautifulSoup(html_source,'lxml')

# 각 동영상의 링크를 얻기 위해 개별 동영상을 담은 list 생성
video_list = soup.find_all('ytd-grid-video-renderer',{'class':'style-scope ytd-grid-renderer'})

# 축약된 url을 전체 url로 변환시키기
base_url = 'https://www.youtube.com'
url_list = []

# thumbnail 에 a 태그로 포함된 href 링크를 찾고 완전한 url로 변환시켜줌
for i in range(len(video_list)):
  full_url = base_url+video_list[i].find('a',{'id':'thumbnail'})['href']
  url_list.append(full_url)

# 스포츠 채널의 DATAFRAME
sports = pd.DataFrame({'channel_name' :[],
                           'subscribers' : [],
                           'thumbnail' :[],
                           'video_name':[],
                           'video_length':[],
                           'upload_date':[],
                           'hits':[],
                           'likes_num':[],
                           'dislikes_num':[],
                           'category':[] })

# url_list 순회하며 정보 저장
for i in range(len(url_list)):

  try:
    # 전체 채널 정보 (채널이름, 구독자)
    # 동영상 url 이전에 얻기 편한 정보 (동영상 제목, 동영상 길이)
    channel_name = soup.find('title').text
    subscribers = soup.find('yt-formatted-string',{'id':'subscriber-count'}).text    
    video_name = video_list[i].find('a',{'id':'video-title'}).text
    video_length = video_list[i].find('span', {'class':'style-scope ytd-thumbnail-overlay-time-status-renderer'}).text.split('\n')[1]
    
    # error 유발 ㅠ, aria-hidden 속성 때문인 것으로 보임
    # thumbnail = video_list[i].find('img', {'src' : True})['src'] 

    # url에 해당하는 동영상 페이지 들어가기
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome('chromedriver',options=options)
    
    # url 삽입
    driver.get(url_list[i])
    
    # 스크롤
    last_page_height = driver.execute_script("return document.documentElement.scrollHeight")
    
    scrolls = 2
    while scrolls:
      driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
      time.sleep(1.5)       # 인터벌 1이상으로 줘야 데이터 취득가능(롤링시 데이터 로딩 시간 때문)
      scrolls -= 1

    # video_source를 parsing
    video_source = driver.page_source
    video_soup = BeautifulSoup(video_source,'lxml')
    driver.close()

    # 정보 가져오기
    # 올린 날짜, 조회수, 좋아요 수, 싫어요 수, 카테고리 정보
    
    scripts = video_soup.find('script',{'id':'scriptTag'}).text.split('"')
    for script in scripts:
      if 'img' in script:
        thumbnail = script
        break
    upload_date = video_soup.find('div',{'id':'date'}).find('yt-formatted-string').text
    hits = video_soup.find('span',{'class':'view-count'}).text.split()[0]
    likes_num = video_soup.find_all('yt-formatted-string',{'id':'text','class':'style-scope ytd-toggle-button-renderer style-text'})[0]['aria-label'].split()[0]
    dislikes_num = video_soup.find_all('yt-formatted-string',{'id':'text','class':'style-scope ytd-toggle-button-renderer style-text'})[1]['aria-label'].split()[0]
    category = 'sports' # 바꾸면됨

    # 임시 데이터 프레임을 생성하여 기존 데이터 프레임에 삽입
    tmp = pd.DataFrame({'channel_name' :[channel_name],
                                    'subscribers' : [subscribers],
                                    'thumbnail' :[thumbnail],
                                    'video_name':[video_name],
                                    'video_length':[video_length],
                                    'upload_date':[upload_date],
                                    'hits' :[hits],
                                    'likes_num':[likes_num],
                                    'dislikes_num':[dislikes_num],
                                    'category':[category]})
    
    # append 이용하여 아래로 concat
    sports = sports.append(tmp)

  # 동영상마다 형식 통일이 되지 않아 정보 조회가 되지 않는 경우 예외 처리
  except KeyError as k: 
      print(k, i)
  except AttributeError as a: 
      print(a ,i)
  except IndexError as e:
      print(e,i)
  except TypeError as t:
      print(t, i)

sports.to_csv('subusu.csv',encoding='utf-8-sig')
```

## 2. Preprocessing

### 설치 패키지

```python
import glob
import os
```

### 2.1. 데이터 병합

```python
# 디렉토리 내부 파일 목록을 조회하기 위해 경로 설정 및 glob 패키지 사용
input_file = r'/content/drive/My Drive/Colab Notebooks/Github정리/딥러닝프로젝트_유튜브/2.Preprocessing/'
allfile_list = glob.glob(os.path.join(input_file,'*.csv'))

# 데이터 프레임을 모두 리스트에 담은 후
# 리스트에 해당하는 데이터프레임을 concat 시킴
all_data=[]
for file in allfile_list:
    try : 
        df=pd.read_csv(file,encoding='utf-8')
        all_data.append(df)
    except UnicodeDecodeError as e:
        e, file
    
data_combine = pd.concat(all_data, axis=0, ignore_index=True, sort=False)
data_combine.shape
data_combine.head()

# 각 카테고리별 균형을 확인해보기 위해
data_combine['category'].value_counts()

# 칼럼 중 필요한 정보만 추려내기 위해
data_combine.columns

data_combine = data_combine[['channel_name', 'subscribers',
               'thumbnail', 'video_name', 'video_length', 'upload_date', 'hits',
               'likes_num', 'dislikes_num', 'category']]

# 저장
data_combine.to_csv('total_dataset.csv',encoding='utf-8-sig')
```

### 2.2. 데이터 전처리

#### 데이터 불러오기

```python
# 데이터 불러오기

df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Github정리/딥러닝프로젝트_유튜브/데이터/total_dataset.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
df.head()

# Nan값 확인 > 대체로 크롤링 과정에서 발생한 오류
df.isnull().sum()

# 개수가 적으므로 삭제
df.dropna(inplace=True)
```

#### 특성별 전처리 (replace, 람다 함수 이용)

* replace
  * DF의 값을 치환하기 위해 사용됨
  * 일부 문자열 일치 여부에 대해 변환하고 싶은 경우 regext 속성값을 이용
    * 정규 표현식으로 찾는 것과 유사한 기능
  * `replace(찾는 패턴 / 변경 패턴 / regex=T/F)`
* apply(lambda x:)
  * 간소한 함수
  * x는 apply 메소드가 적용되는 인자의 값이 할당되며
  * : 이후 부분에 조작하고 싶은 식을 작성하면 반영된다

##### 채널명(channel_name)

```python
# 채널명 변수의 값을 확인
df.channel_name.unique()

# '-'를 기준으로 채널 명을 구분할 수 있으므로 람다 사용
df['channel_name'] = df['channel_name'].apply(lambda x: x.split('-')[0])
```

##### 구독자수(subscribers)

```python
# 값의 구성을 확인
df['subscribers'].unique()

# 앞의 구독자를 날림
df['subscribers'] = df['subscribers'].apply(lambda x: x.split(' ')[1])
# 뒤의 만명을 날림
df['subscribers'] = df['subscribers'].replace('[\만명]', '', regex=True) 
# 정수로 형변환
df['subscribers'] = pd.to_numeric(df['subscribers'],downcast='integer')
# 만 명 단위 이므로 10,000을 곱함
df['subscribers'] = df['subscribers'].apply(lambda x: x*10000)
```

##### 날짜(upload_data)

```python
# 데이터 확인
df['upload_date']

# 일부 형식이 다른 날짜 값을 처리해줍니다.
df['upload_date'] = df['upload_date'].replace('[\최초 공개:]','',regex=True)
df['upload_date'] = df['upload_date'].replace('[\실시간스트리밍시작일]','',regex=True)

# 최초공개, 실시간 스트리밍 등의 경우 날짜 형식이 다른 경우가 발생함
# 이 경우 크롤링 날짜를 기준으로 기간이 얼마 지나지 않았기 때문에 현재 날짜로 단순 대체함
df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce') # '4일전' -> NaN으로 우선 처리
df.loc[df['upload_date'].isnull(),:] # 데이터 수집 날짜 - 4일전 = 2020-03.25 로 처리 
df.loc[df['upload_date'].isnull(),'upload_date'] = '2020-03-25'

# 모든 데이터를 처리한 후 date time 형식으로 바꾸고 그렇게 하고 남아있는 오류는 nan으로 처리
df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce') # 한번 더

# 현재 날짜를 기준으로 역순 정렬
import datetime
now = datetime.datetime.now()
# datetime 형식의 날짜 변환 후 datetime 형식의 오늘 날짜를 쨴 후 (-1)을 곱함
df['upload_date'] = df['upload_date'].apply(lambda x : x - datetime.datetime.now())
df['days_after_upload'] = df['upload_date'].apply(lambda x: x.days * -1)
df.drop('upload_date',axis=1,inplace=True)
```

##### 조회수(hits)

```python
# 조회수 확인
df.hits

# 정규식 이용하여 , / \회를 삭제
df['hits'] = df['hits'].replace('[\회]','',regex=True)
df['hits'] = df['hits'].replace('[\,]','',regex=True)
# 정수형으로 변환
df['hits'] = pd.to_numeric(df['hits'], downcast='integer')
```

##### 좋아요, 싫어요(likes/dislikes)

```python
# 좋아요
df['likes_num'] = df['likes_num'].replace('[\개]', '', regex=True) 
df['likes_num'] = df['likes_num'].replace('[\.]', '', regex=True)
df['likes_num'] = df['likes_num'].replace('[\천]','00',regex=True)
df['likes_num'] = df['likes_num'].replace('[\만]','000',regex=True)
df['likes_num'] = pd.to_numeric(df['likes_num'], downcast='integer')

# 싫어요
df['dislikes_num'] = df['dislikes_num'].replace('[\개]','',regex=True)
df['dislikes_num'] = df['dislikes_num'].replace('[\.]', '', regex=True)
df['dislikes_num'] = df['dislikes_num'].replace('[\천]','00',regex=True)
df['dislikes_num'] = df['dislikes_num'].replace('[\만]','000',regex=True)
df['dislikes_num'] = pd.to_numeric(df['dislikes_num'], downcast='integer')
```

##### 재생 시간(video_length -> video_duration)

```python
# : 으로 구분된 시간
df['video_duration'] = df['video_length'].apply(lambda x: x.split(':'))
# 분 단위로 변환
df['video_duration'] = df['video_duration'].apply(lambda x : int(x[0]) * 60 + int(x[1]))
```

##### 카테고리(0: game , 1: movie, 2: pets, 3: politic, 4: sports, 5: cooking, 6:economy)

```python
# 팩토라이즈는 두 가지 값을 반환함
# labels는 숫자로된 값
# unique는 문자로된 값
# 두 가지는 사전 처럼 연결되어 있음
labels, uniques = df['category'].factorize()
labels
set(list(labels))
uniques

# 팩토라이즈를 통해 변환
# label을 선택
df['category_id'] = df['category'].factorize()[0]
```

##### 칼럼 정리

```python
# 전체 데이터 프레임의 칼럼을 정리함
df = df[['channel_name', 'subscribers', 'video_name', 'days_after_upload','video_duration', 'hits', 'likes_num', 'dislikes_num','category_id', 'thumbnail']]
df.shape
df.head()
df.to_csv('final_preprocessing_data.csv', encoding='utf-8-sig')
```

#### Train-Test split

```python
# 병함, 전처리 된 데이터를 불러오기
df_for_split = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Github정리/딥러닝프로젝트_유튜브/데이터/final_preprocessing_data.csv',encoding='utf-8-sig')
df_for_split.drop('Unnamed: 0',axis=1,inplace=True)
df_for_split.head()
```

##### Test

```python
# category ( 0: game , 1: movie, 2: pets, 3: politic, 4: sports, 5: cooking, 6:economy)
game_test = df_for_split.loc[df_for_split['category_id']==0,:][-500:]
movie_test = df_for_split.loc[df_for_split['category_id']==1,:][-500:]
pets_test = df_for_split.loc[df_for_split['category_id']==2,:][-500:]
politics_test = df_for_split.loc[df_for_split['category_id']==3,:][-500:]
sports_test = df_for_split.loc[df_for_split['category_id']==4,:][-500:]
cooking_test = df_for_split.loc[df_for_split['category_id']==5,:][-500:]
economy_test = df_for_split.loc[df_for_split['category_id']==6,:][-500:]

# 병합 후 저장
total_test_set = pd.concat([cooking_test,economy_test,game_test,movie_test,pets_test,politics_test,sports_test])
total_test_set.to_csv('small_test_set.csv', encoding='utf-8-sig')
```

##### Train

```python
# category ( 0: game , 1: movie, 2: pets, 3: politic, 4: sports, 5: cooking, 6:economy)
game_train = df_for_split.loc[df_for_split['category_id']==0,:][:-3000]
movie_train = df_for_split.loc[df_for_split['category_id']==1,:][:-3000]
pets_train = df_for_split.loc[df_for_split['category_id']==2,:][:-3000]
politics_train = df_for_split.loc[df_for_split['category_id']==3,:][:-3000]
sports_train = df_for_split.loc[df_for_split['category_id']==4,:][:-3000]
cooking_train = df_for_split.loc[df_for_split['category_id']==5,:][:-3000]
economy_train = df_for_split.loc[df_for_split['category_id']==6,:][:-3000]

# 병합 후 저장
total_train_set = pd.concat([cooking_train,economy_train,game_train,movie_train,pets_train,politics_train,sports_train])
total_train_set.to_csv('small_train_set.csv', encoding='utf-8-sig')
```

##### label의 비율을 유지하여 random으로 train/test를 나누어 추출

```python
# sklean package 사용
from sklearn.model_selection import StratifiedShuffleSplit

# train:test 8:2
# index 값을 이용하여 분리함
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(df_for_split,df_for_split['category_id']):
  strat_train_set = df_for_split.iloc[train_index]
  strat_test_set = df_for_split.iloc[test_index]
    
# 각 영상 카테고리 별로 Train 3000개 맞추기
game_train = strat_train_set.loc[strat_train_set['category_id']==0,:][:-918]
movie_train = strat_train_set.loc[strat_train_set['category_id']==1,:]
pets_train = strat_train_set.loc[strat_train_set['category_id']==2,:][:-897]
politics_train = strat_train_set.loc[strat_train_set['category_id']==3,:][:-2265]
sports_train = strat_train_set.loc[strat_train_set['category_id']==4,:][:-1022]
cooking_train = strat_train_set.loc[strat_train_set['category_id']==5,:][:-1030]
economy_train = strat_train_set.loc[strat_train_set['category_id']==6,:][:-820]

# 병합
random_train_set = pd.concat([cooking_train,economy_train,game_train,movie_train,pets_train,politics_train,sports_train])

# 각 영상 카테고리 별로 Test 500개 맞추기
game_test = strat_test_set.loc[strat_test_set['category_id']==0,:][:500]
movie_test = strat_test_set.loc[strat_test_set['category_id']==1,:][:500]
pets_test = strat_test_set.loc[strat_test_set['category_id']==2,:][:500]
politics_test = strat_test_set.loc[strat_test_set['category_id']==3,:][:500]
sports_test = strat_test_set.loc[strat_test_set['category_id']==4,:][:500]
cooking_test = strat_test_set.loc[strat_test_set['category_id']==5,:][:500]
economy_test = strat_test_set.loc[strat_test_set['category_id']==6,:][:500]

# 병합
random_test_set = pd.concat([cooking_test,economy_test,game_test,movie_test,pets_test,politics_test,sports_test])

# 저장
random_train_set.to_csv('random_train_set.csv', encoding='utf-8-sig')
random_test_set.to_csv('random_test_set.csv', encoding='utf-8-sig')
```

## 3. Feature Extraction

### 3.1. Text Feature

#### 패키지

```python
!pip install KoNLPy

import pandas as pd 
import numpy as np
import re
import os
import itertools
import json
from collections import OrderedDict

import seaborn as sns 
import matplotlib as mpl
import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings("ignore")

# 한국어 전처리
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import json
import sys

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_preprocessing import sequence

from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from sklearn.metrics import accuracy_score

# 임베딩
import gensim

# 모델링
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D

from keras import metrics
from keras import optimizers
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
```

#### 텍스트 데이터 전처리

##### 텍스트 정규화 (노이즈 제거)

```python
# 데이터 불러오기
df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Github정리/딥러닝프로젝트_유튜브/데이터/small_train_set.csv", encoding='utf-8')

# 불용어사전 읽어오기
f = open("/content/drive/My Drive/Colab Notebooks/Github정리/딥러닝프로젝트_유튜브/3.Text_Feature/stopwords.txt", 'r', encoding='utf-8')
stopwords = []
for line in f.readlines():
    stopwords.append(line.rstrip())
f.close()

# 분석에 어긋나는 불용어구 제외 (특수문자, 의성어)
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

han = re.compile(r'[ㄱ-ㅎㅏ-ㅣ!?~,".\n\r#\ufeff\u200d]')


clean_name_lst=[]
for i in range(len(df)):
    a = re.sub(han,'',df['video_name'].iloc[i])
    clean_name_lst.append(a)

clean_name_lst=[]
for i in range(len(df)):
    a = re.sub(emoji_pattern,'',df['video_name'].iloc[i])
    b = re.sub(han,'',a)
    
    clean_name_lst.append(b)
    
df['clean_name']=clean_name_lst

# 품사 분리, 토큰화
from konlpy.tag import Okt
okt = Okt()

clean_word = [] 
for i in df['clean_name']:
    text = re.sub("[12?~3.,()''->\n]", '', i)
    word_text = okt.nouns(text)
    if True: word_text = [token for token in word_text if not token in stopwords]
        
    clean_word.append(' '.join(word_text))
    
df['clean_name'] = clean_word
df['okt_tok'] = df['clean_name'].apply(lambda x : okt.nouns(x))

okt_data = df[['category_id','video_name', 'channel_name', 'okt_tok']]
```

##### 토큰화 및 단어 사전 만들기

```python
# 토크나이저 사용
# keras의 tokenizer는 default값이 띄어쓰기 구분이므로 okt로 토큰화된 단어들만 뽑아온다
okt_vocab = Tokenizer()
okt_vocab.fit_on_texts(okt_data['okt_tok'])

threshold = 3
total_cnt = len(okt_vocab.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in okt_vocab.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value
        
# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거.
okt_vocab_size = total_cnt - rare_cnt 
print('단어 집합의 크기 :', okt_vocab_size)

# 빈도수가 3미만인 단어를 w라고 저장
okt_freq = [w for w,c in okt_vocab.word_counts.items() if c < 3]

for w in okt_freq:
    del okt_vocab.word_index[w]
    del okt_vocab.word_counts[w]

okt_vocab_list = list(okt_vocab.word_index.keys())

# 단어 리스트 만들기
okt_all = okt_data['okt_tok'].tolist()

titles = list(okt_data.video_name)
id_num = list(okt_data.category_id)
```

##### 임베딩 벡터

```python
# sg=0 (cbow), sg=1 (skip-gram)
from gensim.models import Word2Vec
w2v = Word2Vec(sentences = okt_all, size = 100, window = 2, min_count = 1, workers = 4,sample=1e-3, iter=5, sg = 1)

from gensim.models import FastText
fxt = FastText(sentences = okt_all, size = 100, window = 4, min_count = 1, workers = 4,sample=1e-3, iter=5, sg = 1)
```

##### 문장 내 단어 벡터의 평균값 = 문장의 테스트 Feature

```python
def get_features(words, model, num_features):
    # 출력 벡터 초기화
    feature_vector = np.zeros((num_features), dtype=np.float32)
    
    num_words = 0
    # 어휘 사전 준비
    index2word_set = set(model.wv.index2word)
    
    for w in words:
        if w in index2word_set:
            num_words = 1
            # 사전에 해당하는 단어에 대해 단어 벡터를 더함
            feature_vector = np.add(feature_vector, model[w])
            
    # 문장의 단어 수만큼 나누어 단어 벡터의 평균값을 문장 벡터로 함
    feature_vector = np.divide(feature_vector, num_words)
    return feature_vector

def get_dataset(data, model, num_features):
    dataset = list()
    
    for s in data:
        dataset.append(get_features(s, model, num_features))
        
    reviewFeatureVecs = np.stack(dataset)
    return reviewFeatureVecs

okt_w2v = get_dataset(okt_all, w2v, 100)
okt_fxt = get_dataset(okt_all, fxt, 100)

np.save("/content/drive/My Drive/Colab Notebooks/Github정리/딥러닝프로젝트_유튜브/3.Text_Featureokt_w2v", okt_w2v) # save.npy
np.save("/content/drive/My Drive/Colab Notebooks/Github정리/딥러닝프로젝트_유튜브/3.Text_Featureokt_fxt", okt_fxt) # save.npy
```

### 3.2. Image Feature

#### VGG16

##### 패키지

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from glob import glob

from PIL import Image
import requests

import io
from io import BytesIO

from tensorflow import keras
```

##### URL > Image 변환

```python
# 데이터
df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Github정리/딥러닝프로젝트_유튜브/데이터/small_train_set.csv", encoding='utf-8-sig')

# 이미지 분리
X = df['thumbnail']
y = df['category_id']
y = np.array(y)

# URL to Image
from keras.preprocessing import image

X_img = []
for i in X :
    url = i
    response = requests.get(url) 
    img = Image.open(io.BytesIO(response.content)) 
    size = (168,94)

    img = img.resize(size)
    # img = img.convert("RGB")
    img = image.img_to_array(img)
    
    X_img.append(img)
X_img = np.array(X_img)

# 이미지 정규화 (255 픽셀 값 단위, 0~1)
plt.imshow(X_img[0]/255)
```

##### 전이 학습

```python
# 학습 데이터 구분
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_img, y)

# 전이 학습을 위해 top 레이어와 trinable 옵션 False 지정
pre_trained_vgg = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(94, 168, 3))
pre_trained_vgg.trainable = False
pre_trained_vgg.summary()

# 모델 설계
additional_model = keras.models.Sequential()
additional_model.add(pre_trained_vgg)
additional_model.add(keras.layers.Flatten()) # 5120
additional_model.add(keras.layers.Dense(2048, activation='relu'))
additional_model.add(keras.layers.Dense(512, activation='relu'))
additional_model.add(keras.layers.Dense(7, activation='softmax'))
 
additional_model.summary()

# plot
keras.utils.plot_model(additional_model, show_shapes=True)

# 학습
additional_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# epochs 6 이상은 RAM error가 발생함 ㅠ
history = additional_model.fit(X_train, y_train, epochs=5, validation_data=(X_valid, y_valid))

# 그래프로 확인
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# 특성 추출
extract_features_model = keras.models.Sequential()
extract_features_model.add(pre_trained_vgg)
extract_features_model.add(keras.layers.Flatten()) # 5120
extract_features_transfer_train = extract_features_model.predict(X_img)

# 저장
np.save("/content/drive/My Drive/Colab Notebooks/Github정리/딥러닝프로젝트_유튜브/4.Image_Feature/extract_features_transfer_train", extract_features_transfer_train) 
```

#### CNN(self_train)

##### 패키지

```python
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import requests
import os
from glob import glob
import io
import numpy as np
from io import BytesIO
```

##### URL > Image 변환

```python
# 데이터 불러오기
df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Github정리/딥러닝프로젝트_유튜브/데이터/small_train_set.csv', encoding='utf-8-sig')

# 이미지 변환하기
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

category = df['category_id']
thumbnail = df['thumbnail']

image_w = 64
image_h = 64
pixels = image_w * image_h * 3
X = []
for i in thumbnail :
    
    url = i+str('.jpg')
    response = requests.get(url, stream=True) 
    img= Image.open(io.BytesIO(response.content)) 
    
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    img = image.img_to_array(img)

    # 데이터를 전처리합니다(채널별 컬러 정규화를 수행합니다)
    img = preprocess_input(img)
    X.append(img)
    
X = np.array(X)
Y = np.array(category)
Y = pd.get_dummies(Y).to_numpy()
```

##### 모델 학습

```python
# 모델 구성
# 카테고리 지정하기
categories = category.unique()
nb_classes = len(categories)
# 이미지 크기 지정하기
image_w = 64
image_h = 64

# 데이터 정규화하기(0~1사이로)
X = X.astype("float") / 64

from keras.models import Sequential
from keras import metrics
from keras import optimizers
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import os

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3))) #전체 train에 돌릴때는 32로
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
last_layer = model.add(Dropout(0.25))

# 전결합층
model.add(Flatten())    # 벡터형태로 reshape
model.add(Dense(512))   # 출력
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# 모델 구축하기
model.compile(loss='categorical_crossentropy',   # 최적화 함수 지정
    optimizer='rmsprop',
    metrics=['accuracy'])

# 모델 요약
model.summary()

# 특성 추출하기
from keras.models import Model
layer_name = 'flatten_1'
flatten_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
flatten_output = flatten_layer_model.predict(X)

# 특성 저장하기
seq_image_features = pd.DataFrame(flatten_output)
np.save("/content/drive/My Drive/Colab Notebooks/Github정리/딥러닝프로젝트_유튜브/4.Image_Feature/seq_image_features",seq_image_features)
```

## 4. 특성 결합

#### 저장된 모델 합치기

```python
# 학습 데이터
df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Github정리/딥러닝프로젝트_유튜브/데이터/small_train_set.csv', encoding='utf-8-sig')
y = df['category_id']
y = np.array(y)
y.shape

# 이미지 특성
feature_self_cnn = np.load('/content/drive/My Drive/Colab Notebooks/Github정리/딥러닝프로젝트_유튜브/4.Image_Feature/seq_image_features.npy')
features_vgg_train = np.load('/content/drive/My Drive/Colab Notebooks/Github정리/딥러닝프로젝트_유튜브/4.Image_Feature/extract_features_transfer_train.npy')

# 텍스트 특성
feature_fxt = np.load('/content/drive/My Drive/Colab Notebooks/Github정리/딥러닝프로젝트_유튜브/3.Text_Feature/okt_fxt.npy')
feature_w2v = np.load('/content/drive/My Drive/Colab Notebooks/Github정리/딥러닝프로젝트_유튜브/3.Text_Feature/okt_w2v.npy')

# 결합

# numpy 배열을 옆으로 붙입니다
case1_data = np.c_[features_vgg_train,feature_w2v]

# text 데이터 중 nan인 것이 있어 not np.isnan 을 통해 nan mask를 통해 데이터에 T/F 표시를 합니다.
nanmask = ~np.isnan(case1_data).any(axis=1)
# True인 데이터를 받습니다.
case1_nan = case1_data[nanmask]
y_nan = y[nanmask]

# 샘플의 개수 중 0.8개를 임의로 고릅니다.
train_idx = np.random.choice(case1_nan.shape[0], round(case1_nan.shape[0]*0.8), replace=False)
# 전체 데이터에서 train data index를 빼어 test data를 만듭니다.
test_idx = np.setdiff1d(range(case1_nan.shape[0]), train_idx)

train_idx.shape, test_idx.shape

def cbind_(imgdata,textdata) :
  global train_idx
  global test_idx
  data = np.c_[imgdata,textdata]
  nanmask = ~np.isnan(data).any(axis=1)
  x_data = data[nanmask]
  y_data = y[nanmask]
  
  x_train = x_data[train_idx,:]
  y_train = y_data[train_idx]
  x_test = x_data[test_idx,:]
  y_test = y_data[test_idx]
  return x_train, y_train, x_test, y_test

# 정의된 함수에 모델 삽입
x_case1_model, y_case1_model, x_case_1_test, y_case1_test = cbind_(features_vgg_train,feature_w2v)
x_case2_model, y_case2_model, x_case_2_test, y_case2_test = cbind_(features_vgg_train,feature_fxt)
x_case3_model, y_case3_model, x_case_3_test, y_case3_test = cbind_(feature_self_cnn,feature_fxt)
x_case4_model, y_case4_model, x_case_4_test, y_case4_test = cbind_(feature_self_cnn,feature_w2v)
```

#### 모델 학습

```python
model1 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[5320]),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=0.4),
    keras.layers.Dense(1024, activation="relu",
                         kernel_regularizer = keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="relu",
                         kernel_regularizer = keras.regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(7, activation="softmax")
])

model1.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

from sklearn.model_selection import train_test_split
np.random.seed(42)
X_train, X_valid, y_train, y_valid = train_test_split(x_case1_model, y_case1_model)
X_train.shape, X_valid.shape
history = model1.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid))

# 그래프
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```

#### 테스트

```python
score1 = model1.evaluate(x_case_1_test, y_case1_test)
score2 = model2.evaluate(x_case_2_test, y_case2_test)
score3 = model3.evaluate(x_case_3_test, y_case3_test)
score4 = model4.evaluate(x_case_4_test, y_case4_test)

pred_ = np.argmax(model2.predict(x_case_2_test), axis=-1)

from sklearn.metrics import confusion_matrix
confusion_matrix( y_case2_test,pred_,)

# labels=["game", "movie", "pets","politics","sports","cooking","economy"]

y_true = pd.Series(y_case2_test)
y_pred = pd.Series(pred_)

pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'])
```



## 5. 해석




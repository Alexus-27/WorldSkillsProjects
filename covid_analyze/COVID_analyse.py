#!/usr/bin/env python
# coding: utf-8

# # Проект  системы диагностики эпидемиологической ситуации в странах мира

# Данная система диагностики будет включать исследование имеющихся открытых статистических данных портала https://github.com/owid. Система будет предсказывать уровень заболеваемости в виде трех кластеров и прогнозировать дальнейшее развитие эпидемиологической ситуации, также отображаемое визуально.
# 
# Система имеет три кластера: зеленого цвета - страна безопасна в данный момент времени (кластер 0), въезд возможен, но не желателен (кластер 1), а также опасный уровень (кластер 2). Прогноз заболевания в выбранной стране осуществляется на 30 дней, однако данный диапазон можно изменить.

# ## Импорт данных
# Для грамотной работы с проектом нам понадобится множество библиотек, методов и алгоритмов. Давайте начнем их импортировать. Первым делом загрузим в проект библиотеки, которые не встроены в jupyter notebook (воспользуемся методом pip install):

# In[1]:


# !pip install requests bs4 lxml
#!pip install phik
#!pip install lightgbm


# Теперь начнем загрузку тех библиотек, методов и алгоритмов, которые встроены в jupyter notebook, а также которые мы сейчас загрузили:

# In[2]:


import pandas as pd
import requests # парсинг
from bs4 import BeautifulSoup # парсинг
import lxml # парсинг
import numpy as np
from sklearn.cluster import KMeans # кластеризация
from sklearn.mixture import GaussianMixture # кластеризация
from sklearn.preprocessing import StandardScaler # масштабирование
import matplotlib.pyplot as plt # графики
import seaborn as sns # графики
from tqdm import notebook
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, TimeSeriesSplit # разделение данных 
from phik.report import plot_correlation_matrix # матрица зависимости данных 
from sklearn.linear_model import LogisticRegression, LinearRegression # линейные модели
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # модели ансамблей
from lightgbm import LGBMClassifier, LGBMRegressor # градиентный бустинг
from sklearn.metrics import accuracy_score, mean_squared_error # метрики 
from datetime import timedelta, datetime # работа с датой

import warnings
warnings.filterwarnings('ignore')


# Далее настроим маленькие удобства: таблица будет выводить все свои столбцы, а также все графики по умолчанию будут иметь размер 20 на 10:

# In[3]:


pd.set_option('display.max_columns', None)
sns.set(rc={'figure.figsize':(20,10)})


# ## Загрузка данных и их анализ
# После того как наши основные библиотеки были загружены, можно приступить к загрузке данных с репозитория и сохранению их в переменную covid_data:

# In[4]:


path = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
covid_data = pd.read_csv(path, index_col=0)


# Таблица создана, давайте проверим, какие столбцы у нас есть и что они из себя представляют:

# In[5]:


covid_data.head()


# - `iso_code` - Три буквы обозначающие страну
# - `continent` - Континент географического местоположения 
# - `location` - Географическое местоположение 
# - `date` - Дата наблюдения 
# - `population` - Население определенной страны 
# - `population_density` - Площадь населения на квадратный километр
# - `median_age` - Медианый возраст 
# - `aged_65_older` - Доля населения в возрасте 65 лет и старше
# - `aged_70_older` - Доля населения в возрасте 70 лет и старше в 2015
# - `gdp_per_capita` - ВВП по паритету покупательной способности
# - `extreme_poverty` - Доля населения жителей, живущих в крайней нищете
# - `cardiovasc_death_rate` - Смертность от сердечно-сосудистых заболеваний (смерть одного человека к 100000)
# - `diabetes_prevalence` - Распространенность диабета (% процент населения возраста с 20 до 79) в 2017
# - `female_smokers` - Доля курящих женщин
# - `male_smokers` - Доля курящих мужчин
# - `handwashing_facilities` - Доля населения с основными приспособлениями для мытья рук в помещении
# - `hospital_beds_per_thousand` - Больничные койки на 1000 людей
# - `life_expectancy` - Ожидаемая продолжительность жизни при рождении
# - `human_development_index` - Измерение среднего объединенного индекса в трех основных измерениях человеческого развития: качество и продолжительность жизни, уровень образования и достойный уровень жизни
# - `total_cases` - Всего подтвержденных случаев COVID-19
# - `new_cases`	- Новые подтвержденные случаи COVID-19.
# - `new_cases_smoothed` - Новые подтвержденные случаи COVID-19 (сглажено за 7 дней)
# - `total_deaths` - Всего смертей, связанных с COVID-19
# - `new_deaths` - Новые смерти, связанные с COVID-19
# - `new_deaths_smoothed` - Новые смерти, связанные с COVID-19 (сглажено за 7 дней)
# - `total_cases_per_million` - Всего подтвержденных случаев COVID-19 на 1 000 000 человек 
# - `new_cases_per_million` - Новые подтвержденные случаи COVID-19 на 1 000 000 населения                   
# - `new_cases_smoothed_per_million` - Новые подтвержденные случаи COVID-19 (сглажено за 7 дней) на 1 000 000 человек.              
# - `total_deaths_per_million` - Всего смертей от COVID-19 на 1 000 000 человек
# - `new_deaths_per_million` - Новые смерти от COVID-19 на 1 000 000 человек   
# - `new_deaths_smoothed_per_million` - Новые смерти, связанные с COVID-19 (сглажено за 7 дней) на 1 000 000 человек             
# - `reproduction_rate` - Оценка эффективной скорости размножения (R) COVID-19 в режиме реального времени.                         
# - `icu_patients` -  Количество пациентов с COVID-19 в отделениях интенсивной терапии (ОИТ) в данный день                              
# - `icu_patients_per_million` - Количество пациентов с COVID-19 в отделениях интенсивной терапии (ОИТ) в данный день на 1 000 000 человек               
# - `hosp_patients` - Количество пациентов с COVID-19 в больнице в данный день 
# - hosp_patients_per_million - Количество пациентов с COVID-19 в больнице в данный день на 1 000 000 человек                   
# - `weekly_icu_admissions` - Количество пациентов с COVID-19, впервые поступивших в отделения интенсивной терапии (ОИТ) за данную неделю         
# - `weekly_icu_admissions_per_million` - Количество пациентов с COVID-19, впервые поступивших в отделения интенсивной терапии (ОИТ) за данную неделю на 1 000 000 человек           
# - `weekly_hosp_admissions` - Количество пациентов с COVID-19, впервые поступивших в больницы за неделю                      
# - `weekly_hosp_admissions_per_million` - Количество пациентов с COVID-19, впервые поступивших в больницы за данную неделю, на 1 000 000 человек       
# - `new_tests` - Новые тесты на COVID-19 (рассчитаны только на последовательные дни)                                   
# - `total_tests` - Всего тестов на COVID-19                                 
# - `total_tests_per_thousand` - Всего тестов на COVID-19 на 1000 человек
# - `new_tests_per_thousand` - Новые тесты на COVID-19 на 1000 человек         
# - `new_tests_smoothed` - Новые тесты на COVID-19 (7-дневные сглаженные)       
# - `new_tests_smoothed_per_thousand` - Новые тесты на COVID-19 (7-дневные сглаженные) на 1000 человек             
# - `positive_rate` - Доля положительных тестов на COVID-19, указанная как скользящее среднее за 7 дней (обратное значение test_per_case)           
# - `tests_per_case` - Количество тестов, проведенных на каждый новый подтвержденный случай COVID-19, представленное как скользящее среднее значение за 7 дней (обратное положительному_коэффициенту)                   
# - `tests_units` - Единицы, используемые местоположением для сообщения данных тестирования                                 
# - `total_vaccinations` - Общее количество введенных доз вакцины против COVID-19                          
# - `people_vaccinated` - Общее количество людей, получивших хотя бы одну дозу вакцины                           
# - `people_fully_vaccinated` - Общее количество людей, получивших все дозы, предусмотренные первоначальным протоколом вакцинации                     
# - `total_boosters` - Общее количество введенных бустерных доз вакцины против COVID-19 (доз, введенных сверх количества, предусмотренного протоколом вакцинации)                              
# - `new_vaccinations` - Введены новые дозы вакцины против COVID-19 (рассчитаны только за последующие дни)                            
# - `new_vaccinations_smoothed` - Введены новые дозы вакцины против COVID-19 (сглажено за 7 дней).                   
# - `total_vaccinations_per_hundred` - Общее количество доз вакцины против COVID-19, введенных на 100 человек всего населения             
# - `people_vaccinated_per_hundred` - Общее количество людей, получивших хотя бы одну дозу вакцины на 100 человек в общей численности населения           
# - `people_fully_vaccinated_per_hundred` - Общее количество людей, получивших все дозы, предусмотренные протоколом первичной вакцинации, на 100 человек в общей численности населения         
# - `total_boosters_per_hundred` - Общее количество бустерных доз вакцины против COVID-19, введенных на 100 человек в общей численности населения     
# - `new_vaccinations_smoothed_per_million` - Введены новые дозы вакцины против COVID-19 (сглажено за 7 дней) на 1 000 000 человек в общей численности населения       
# - `new_people_vaccinated_smoothed` - Ежедневное количество людей, получивших первую дозу вакцины (сглажено за 7 дней)              
# - `new_people_vaccinated_smoothed_per_hundred` - Ежедневное количество людей, получающих первую дозу вакцины (сглажено за 7 дней) на 100 человек в общей численности населения  
# - `stringency_index` - Индекс жесткости реагирования правительства: составной показатель, основанный на 9 индикаторах реагирования, включая закрытие школ, закрытие рабочих мест и запрет на поездки, перемасштабированный до значения от 0 до 100 (100 = самый строгий ответ)                           
# - `excess_mortality_cumulative_absolute` - Совокупная разница между зарегистрированным числом смертей с 1 января 2020 г. и прогнозируемым числом смертей за тот же период на основе предыдущих лет.	
# - `excess_mortality_cumulative` - Процентная разница между совокупным числом смертей с 1 января 2020 г. и совокупным прогнозируемым числом смертей за тот же период на основе предыдущих лет.	
# - `excess_mortality` - Процентная разница между зарегистрированным количеством еженедельных или ежемесячных смертей в 2020–2021 гг. и прогнозируемым количеством смертей за тот же период на основе предыдущих лет	
# - `excess_mortality_cumulative_per_million` - Совокупная разница между зарегистрированным числом смертей с 1 января 2020 года и прогнозируемым числом смертей за тот же период на основе предыдущих лет на миллион человек.
# 
# Описав все столбцы в таблице нужно перейти к краткой информации о данных:

# In[6]:


covid_data.info()


# Всего у нас 170 тысяч строк, из которых максимальное число пропусков до ходит до 163 тысяч строк. Формат столбцов у всех верный, кроме `date` - необходимо заменить на datetime. Так как данных у нас много, стоит выделить те, что нам нужны для предсказания уровня опасности страны, а также предсказания заражения наперед. Стоит рассмотреть следующие столбцы:
# - location - обязателен, ведь нужно знать в какой стране мы будем предсказывать данные
# - date - все строится по дате, это наш ключевой столбец 
# - population_density - площадь населения может пригодится при предсказании новых случаев заражения
# - reproduction_rate - скорости размножения короновирусной инфекции также может понадобится при предсказании уровня опасности и новых случаев заражения
# - hosp_patients - количество пациентов также может показать сильную связь с новыми заражениями, ведь чем больше пациентов, тем больше зараженных
# - new_tests - новые тесты показывают, как много зараженных становится изо дня в день
# - total_vaccinations - общее количество вакцинированных поможет заметить связь с число зараженных (чем больше прививок, тем меньше зараженных)
# - people_fully_vaccinated_per_hundred - количество вакцинированных на 100 человек покажет нам частоту вакцинации населения, что может помочь в предсказании новых случаев заражения
# - total_deaths - чем больше смертей, тем больше зараженных
# - new_deaths - чем больше смертей, тем больше зараженных
# - total_cases - сумма того, что нам нужно предсказать
# - new_cases - целевая переменная

# In[7]:


important_col = ['location', 'date', 'population_density', 'reproduction_rate', 'hosp_patients',
                'new_tests', 'people_fully_vaccinated_per_hundred', 'new_deaths', 'new_cases']
important_data = covid_data.reset_index()[important_col]
important_data.head()


# Необходимые столбцы были выбраны и теперь наша таблица стала более удобной в чтении и анализе. Давайте проверим, все-ли значения в столбце `location` верны и нет ли там ненужных данных:

# In[8]:


covid_data['location'].unique()


# Как можно заметить, в столбце `location` есть неверные значения и их не мало: Africa, Asia ... Теперь нужно проверить, сколько пропусков присутствует в наших столбцах. Для этого создадим графическую таблицу:

# In[9]:


passed_data = important_data.isna().sum().reset_index()
passed_data.columns = ['column', 'count_passed']
passed_data = passed_data.sort_values('count_passed', ascending=False)
passed_data['percent_of_passed'] = passed_data['count_passed']/important_data.shape[0]


# In[10]:


passed_data.style.bar(subset=['count_passed','percent_of_passed'], color='#d65f5f')


# Огромное количество пропусков в столбцах: `hosp_patients`, `people_fully_vaccinated_per_hundred`, `total_vaccinations`, `new_tests` и `reproduction_rate`. Удалять пропуски в этих данных выглядит плохой идей, да и вообще удаление хоть одной строки, может повредить нашу периодичность в данных, ведь все строится по датам и странам, поэтому нужно в любом случае заменять пропуски на какие-либо значения.

# ## Предобработка данных
# На данном этапе мы будем решать проблему пропусков, неверных значений, а также поиском аномалий в данных. 
# 
# Как уже было выше отмечено, удаление пустых значений может привести к нарушению периодичности и порядку наших данных, что может негативно сказаться на предсказании уровня опасности и новых случаев заражения. Удалению предпочтем замену, ведь если вдуматься, большинство пропусков в данных стоят по причине отсутствия каких либо значений: в начале пандемии количество новых тестов и смертей было 0, количество вакцинированных также было на таком уровне - 0. Остальные пропуски можно заменить на предыдущие значения (возможно пропуск означает отсутствие изменений).
# 
# Неверные значения нужно посмотреть в столбце `location`, где можно было заметить "не страны": Africa, Europe и т д.

# У нас много ошибочных значений в столбце `location`, к примеру Aftica, Asia не являются странами, поэтому их стоит убрать отсюда:

# In[11]:


print('До удаления ненужных локаций:', covid_data.shape)
wrong_location = (['Africa', 'Asia', 'Europe', 'European Union', 'High income', 
                   'International', 'Low income', 'Lower middle income', 
                   'North America','South America', 'Oceania', 'Upper middle income', 
                   'World'])
covid_data = covid_data.query("location not in @wrong_location")
print('После удаления ненужных локаций:', covid_data.shape)


# Мы удалили 10 тысяч строк, что не так много и это никак не сказалось на логике данных. Теперь нужно заполнить пустые значения. Выделим два списка: первый - col_fill_values, где мы будем заполнять 0 те места, где у нас вообще нет никаких значений и значением предыдущей строки, если пустое значение находится между данными, которые имеют значения. Второй список - заполнение каким-нибудь заполнителем, например -1:

# In[12]:


col_fill_values = ['hosp_patients', 'people_fully_vaccinated_per_hundred', 'new_tests', 
                   'reproduction_rate', 'new_deaths', 'new_cases',]
col_fill_undefined = ['population_density']


# In[13]:


for col in notebook.tqdm(col_fill_values):
    for index in notebook.tqdm(range(important_data.shape[0])):
        try:
            if important_data.loc[index].isna()[col] == True:
                new_value = important_data.copy().loc[index-1, col]
                important_data.loc[index, col] = new_value
        except:
            important_data.loc[index, col] = 0


# In[14]:


important_data.head(30)


# In[15]:


important_data['population_density'] = important_data['population_density'].fillna(-1)


# Заполнение пустых значений завершено, теперь наши данные осталось проверить на аномалии и изменить формат столбца `date` на date:

# In[16]:


important_data['date'] = pd.to_datetime(important_data['date'], format='%Y-%m-%d')


# In[17]:


important_data.info()


# Как видно по верхней таблице, формат столбца `date` изменен на `datetime64`, что нам и нужно было. Пропуски отсутствуют, значит нам осталось проверить последний момент - наличие аномальных значений в таблице:

# In[18]:


important_data.describe()


# По таблице выше заметно, что в столбце `population_density` минимальное значение = -1, что верно, ведь мы выше все пропуски заменяли на данное значение. Столбец `reprodiction_rate` содержит минмиальное значение -0.04, что опять же логично, ведь скорость заражение COVID может стать отрицательной, благодаря вакцинации населения.
# 
# В итоге в данных все отлично.

# Предобработка данных завершена и можно приступить к этапу создания нового признака - rt_value (коэффициент распространения инфекции)

# ## Создание нового признака
# На данном этапе будет создан новый признак - коэффициент распространения инфекции (Rt). Для того, чтобы понять что это за коэффициент, необходимо привести формулу подсчета данного коэффициента: 
# 
# $Rt = (n_{8}+n_{7}+n_{6}+n_{5}) / (n_{1}+n_{2}+n_{3}+n_{4})$, где n - день заражения
# 
# Для реализации данной формулы мы применим работу двух циклов, которые будут пробегать по всем странам в таблице и по всей таблице, которая была сгруппирована по стране и датам. Если дней с начала ведения записей прошло меньше 8, то rt равен 0. Если в знаменателе получается 0, то rt равен 0. Результаты rt значений мы записываем в список, который в дальнейшем используем:

# In[19]:


rt_values = []
pivot_table = important_data.groupby(['location', 'date'])['new_cases'].sum()
all_countries = list(important_data['location'].unique())
for country in notebook.tqdm(all_countries):
    for days_observation in range(1, pivot_table[country].count()+1):
        if days_observation >= 8:
            rt_value_denominator = sum(pivot_table[country][days_observation-4:days_observation-1])
            rt_value_numerator = sum(pivot_table[country][days_observation-8:days_observation-5])
            try:
                rt_values.append(rt_value_numerator/rt_value_denominator)
            except:
                rt_values.append(0.0)
        else:
            rt_values.append(0.0)


# Отлично, все значения записаны в список и мы можем их применить - для этого создаем новый столбец `rt_value` и вносим в него значения со списка:

# In[20]:


important_data['rt_value'] = rt_values.copy()
important_data['rt_value'] = round(important_data['rt_value'], 4)
important_data.tail()


# Значения добавлены, однако давайте проверим их на адекватность:

# In[21]:


important_data.rt_value.describe()


# Максимальное значение столбца получается довольно большим, а это значит, что данный выброс нужно отбросить (некоторые модели машинного обучения крайне чувствительны к выбросам). Для этого выберим крайнее значение, после которого мы будем отбрасывать данные:

# In[22]:


important_data[important_data.rt_value > 2.5]['rt_value'].count()/important_data.shape[0]


# В процессе отбора значения выяснилось, что значение 2.5 может послужить границей отделения. Обрезаем данные по данное значение включительно, а то, что будет больше граничного значения, то брать в таблицу не будем:

# In[23]:


cutted_data = important_data[important_data.rt_value <= 2.5]
cutted_data.shape[0]


# Создание нового признака завершено, также было проведено отбрасывания аномальных значений. Далее мы приступаем к кластеризации.

# ## Кластеризация
# Задача кластеризации – определить уровни опасности для туристов и дать им наименования. В результате кластеризации может получиться несколько уровней опасности для одной страны в разные периоды времени. 
# 
# Для кластеризации были выбраны две модели: Kmeans (за счет своей простоты и скорости) и GaussianMixture (за счет скорости и качества лучше, чем у Kmeans).
# 
# Для кластеризации мы решили взяли все признаки из таблицы `cutted_data` и создали на ее основе две таблицы для работы с разными кластерами: `data_for_clusters_km` и `data_for_clusters_gm`:

# In[24]:


data_for_clusters = cutted_data.copy()
data_for_clusters_km = data_for_clusters.copy()
data_for_clusters_gm = data_for_clusters.copy()


# Первый алгоритм кластеризации - Kmeans. Его мы обучим на всех данных, кроме `location` и `date`:

# In[25]:


model = KMeans(n_clusters=3, max_iter=300, n_init=10, random_state=49)
data_for_clusters_km['clusters'] = model.fit_predict(data_for_clusters_km.drop(['location', 'date'], axis=1))
data_for_clusters_km.clusters.value_counts()


# На выходе получаем картину, в которой почти все данные принадлежат кластеру 0. Давайте посмотрим какие максимальные новые случаи были в данных кластерах: 

# In[26]:


print("Максимальный коэффициент распространения инфекции в кластере 0:", data_for_clusters_km[data_for_clusters_km.clusters == 0]['new_cases'].max())
print("Максимальный коэффициент распространения инфекции в кластере 1:", data_for_clusters_km[data_for_clusters_km.clusters == 1]['new_cases'].max())
print("Максимальный коэффициент распространения инфекции в кластере 2:", data_for_clusters_km[data_for_clusters_km.clusters == 2]['new_cases'].max())


# Видно, что данные в кластерах 0 и 1 почти не отличаются - это уже настораживает. В кластере 2 данные почти в 3 раза больше данных нулевого кластера, однако их всего 1347, в то время как 0 кластер содержит 141 тысячу строк. Давайте добавим цвета к нашим данным: у самого кластера с самым большим маскимальным значением цвет будет красным и так пойдем по убыванию: сначала оранжевый, потом зеленый: 

# In[27]:


colors = ['lightgreen', 'orange', 'red']
data_for_clusters_km['color'] = data_for_clusters_km.clusters.map({0:colors[0], 1:colors[2], 2:colors[1]})


# Теперь можем визуализировать данные:

# In[28]:


plt.scatter(data_for_clusters_km.rt_value, data_for_clusters_km.new_cases, c=data_for_clusters_km.color, s=30)
#plt.ylim(-0.1, 30)
#plt.xlim(-1, 60)
plt.xlabel('Новые случаи COVID-19')
plt.ylabel('Значение коэффициента rt')


# По графику видно, что опасный уровень страны хоть и выделяется, но только в самые тяжелые случаи, в центре данные смешиваются и непонятно, где опасно, а где безопасно. Необходимо рассмотреть другой алгоритм кластеризации - GaussianMixture.

# In[29]:


gm = GaussianMixture(n_components=3, random_state=49)

data_for_clusters_gm['clusters'] = gm.fit_predict(data_for_clusters_gm.drop(['location', 'date'], axis=1))
data_for_clusters_gm['clusters'].value_counts()


# In[30]:


print("Максимальный коэффициент распространения инфекции в кластере 0:", data_for_clusters_gm[data_for_clusters_gm.clusters == 0]['new_cases'].max())
print("Максимальный коэффициент распространения инфекции в кластере 1:", data_for_clusters_gm[data_for_clusters_gm.clusters == 1]['new_cases'].max())
print("Максимальный коэффициент распространения инфекции в кластере 2:", data_for_clusters_gm[data_for_clusters_gm.clusters == 2]['new_cases'].max())


# У Gaussian более реальная разбивка данных: у безопасного кластера максимальное количество новых заражений было 225, у самого опасного - 1382 тысяч, при этом количество данных в кластерах более реальное: в опасном кластере 21 тысяча строк, в то время как в безопасном 60 тысяч и, неожиданно, в среднем кластере опасности целых 71 тысяча строк. Давайте зададим цвета и посмотрим на график:

# In[31]:


data_for_clusters_gm['color'] = data_for_clusters_gm.clusters.map({0:colors[0], 1:colors[1], 2:colors[2]})


# In[32]:


plt.scatter(data_for_clusters_gm.rt_value, data_for_clusters_gm.new_cases, c=data_for_clusters_gm.color, s=30)
plt.ylim(-10, 12000)
plt.xlabel('Уровень коэффиицента Rt')
plt.ylabel('Новые случаи COVID-19')


# По графику видно, что высокий уровень опасности хорошо отделяется от среднего уровня и тем более от безопасного, однако может заметить присутствие красных точек в гуще оранжевых - это может говорить о том, что в данных странах все совсем плохо, если такое количество новых случаев в день заставляет их соотносить к опасным странам. 
# 
# В качестве основного алгоритма мы выбрали Gaussian, его кластеры и цвета мы добавим в нашу основную таблицу:

# In[33]:


data_for_clusters_gm.columns


# In[34]:


data_ready = cutted_data.merge(data_for_clusters_gm, on=['location', 'date', 'population_density', 'reproduction_rate',
       'hosp_patients', 'new_tests', 'people_fully_vaccinated_per_hundred',
       'new_deaths', 'new_cases', 'rt_value'])
data_ready.head()


# Таблица готова, однако стоит изменить значения кластеров либо сделать отдельный столбец с текстовым описанием кластеров - так будет понятнее и легче определить, какой промежуток времени можно считать опасным, не самым лучшим или безопасным:

# In[35]:


data_ready['cluster_info'] = data_ready['clusters'].replace([0, 1, 2], ['безопасно', 'не советуем', 'опасно'])


# Этап кластеризации завершен, теперь можно приступить к поиску наиболее важных признаков для предсказания уровня опасности страны в определенный промежуток времени.

# ## Поиск наиболее зависимых признаков
# На данном этапе будет проведен поиск наиболее зависимых признаков к столбцу `clusters`. Зависимость будем проверять с помощью корреляции Пирсона и коэффициента корреляции, основанного на нескольких уточнениях проверки гипотезы Пирсона. В результате мы найдем наиболее зависимые признаки, которые и возьмем на обучение моделей.
# 
# Первым делом стоит привести столбец `date` в разряд индексов, так нам станет легче работать при проверке дат, а также прогноза новых случаев заражения:

# In[36]:


data_ready = data_ready.set_index('date')
data_ready.sort_index(inplace=True)
data_ready.head()


# Первый алгоритм корреляции - корреляция Пирсона. В качестве данных, которые мы будем проверять, будут все признаки не текстового формата, ведь алгоритм не может работать с текстовой информацией:

# In[37]:


corr_data = data_ready[list(data_ready.columns[1:-2])]


# In[38]:


sns.heatmap(corr_data.corr(), annot=True)


# По корреляции Пирсона заметно, что признаки `new_tests`, `new_deaths`, `new_cases`, `rt_value` имеют зависимость в 20-42%, что и можно считать допустимой нормой. 
# 
# Следующий алгоритм - обновленная корреляция Пирсона - `phik matrix`. Для этого алгоритма мы должны указать интервальные столбцы (все наши столбцы можно считать интервальными):

# In[39]:


interval_cols = ['population_density', 'reproduction_rate', 'hosp_patients','new_tests', 
                 'people_fully_vaccinated_per_hundred', 'new_deaths',
                 'new_cases', 'rt_value']
phik_overview = corr_data.phik_matrix(interval_cols=interval_cols)

plot_correlation_matrix(phik_overview.values, 
                        x_labels=phik_overview.columns, 
                        y_labels=phik_overview.index, 
                        vmin=0, vmax=1, color_map="Greens", 
                        title="Корреляция признаков", 
                        fontsize_factor=1.5, 
                        figsize=(20, 10))
plt.tight_layout()


# По следующей матрице видно, что признаки `rt_value`, `new_deaths`, `new_tests`, `reproduction_rate`, `new_cases`, `hosp_patients` имеют хорошую зависимость (20-48%), что можно считать допустимой нормой. 
# 
# В отличие от обычной корреляции Пирсона у нас есть два признака, которые по мнению обновленной корреляции также хороши в преедсказании. Давайте возьмем все признаки, которые прошли норму в `phik matrix`. В качестве целевой переменной будет признак `clusters`:

# In[40]:


features = corr_data[['rt_value', 'new_cases', 'hosp_patients', 'new_deaths', 'new_tests', 'reproduction_rate']]
target = corr_data['clusters']


# Мы отобрали самые важные признаки и целевую переменную. Теперь необходимо приступить к этапу обучения и поиску наиболее продуктивной модели.

# ## Обучение моделей, классификация
# Этап "Обучение моделей, классификация" содержит в себе разделение на обучающую и валидационную выборки, обучение моделей через RandomizedSearchCV, масштабирования данных и повторное обучение моделей.
# 
# В данном проекте были выбраны три модели машинного обучения: `RandomForest`, `LogisticRegression` и `LightGradientBoosting`. Почему были выбраны следующие модели:
# - `RandomForest` - крайне удобная модель, способная принимать решения за счет голосования среди большого числа деревьев, что увеличивает точность и качество прогнозов. Также большое количество гиперпараметров, которые можно долго настраивать.
# - `LogisticRegression` - линейные модели крайне быстрые и хорошо предсказывают в задачах где присутствует явное разделение классов. 
# - `LightGradientBoosting` - улучшенная версия моделей на основе деревьев, позволяющая грамотнее предсказывать и учиться на своих ошибках. Также присутствует частичное ускорение в работе, по сравнению с древесными моделями. 

# In[41]:


data_train, data_valid, target_train, target_valid = train_test_split(features, target, test_size=0.2, 
                                                                      random_state=49, shuffle=True)


# Данные поделили на обучающую и валидационную выборки. Следующий шаг - построение функции, которая будет находить лучшую модель по методу обучения данной модели на разных гиперпараметрах:

# In[42]:


def find_best_models(model, train_features, valid_features, train_target, valid_target, params, task_type='clf'):
     
    if task_type == 'clf':
        cv = KFold(n_splits=3, shuffle=True, random_state=49)
    
        grid = RandomizedSearchCV(model, params, scoring='roc_auc_ovr', n_iter=1, cv=cv, random_state=49)

        grid.fit(train_features, train_target)
        best_model = grid.best_estimator_
    
        tr_pred, val_pred = (
            accuracy_score(train_target, best_model.predict(train_features)),
            accuracy_score(valid_target, best_model.predict(valid_features))
        )

        return tr_pred, val_pred, best_model
    
    else:
        tscv = TimeSeriesSplit(n_splits=5)
        
        grid = RandomizedSearchCV(model, params, n_iter=1, cv=tscv)

        grid.fit(train_features, train_target)
        best_model = grid.best_estimator_
        
        tr_pred = mean_squared_error(train_target, best_model.predict(train_features))**0.5

        return tr_pred, best_model


# Наша функция работает следующим образом:
# - принимает модель, которую нужно обучить; выборки; гиперпараметры к модели и, если есть, тип задачи: либо классификация, либо регрессия
# - если тип задачи классификация, то инициализируется деление на 3 последовательных части, которые будут добавлены в `grid` `RandomizedSearchCV`, вместе с моделью, гиперпараметрами и методом проверки оценки качества (в нашем случае это roc кривая)
# - далее идет обучение `grid` и поиск наилучшей модели.
# - последний этап - предсказание на обучающей и валидационной выборках с применением метрики `accuracy score`.
# - если же тип задачи регрессия, то меняются два момента: вместо `KFold` данные делятся `TimeSeriesSplit`, который не позволит перемешать данные; в качестве метрики предсказания будет использовать не `accuracy score`, а `mean squared error`.
# 
# Для того, чтобы функция работала нужно закинуть в нее данные - этим мы сейчас и займемся:

# In[43]:


params = {'params_linear':{'C': [0.001, 0.01, 0.1, 1, 10],
                 'solver': ['newton-cg', 'sag', 'lbfgs']}, 
          'params_forest':{'n_estimators': [i for i in range(50, 400, 50)],
                 'min_samples_leaf': [i for i in range(1, 15)],
                 'max_depth': [i for i in range(3, 15)], 
                 'min_samples_split': [i for i in range(2, 10)]}, 
          'params_lgbm':{'n_estimators': [i for i in range(50, 400, 50)],
                 'max_depth': [i for i in range(5, 15)]},
          'params_linear_reg':{'n_jobs':[1,2,-1]} 
         } 


# In[44]:


model_forest = RandomForestClassifier()
forest_tr_pred, forest_val_pred, best_model_forest = find_best_models(model_forest, 
                                                   data_train, data_valid, target_train, target_valid, 
                                                   params['params_forest'])
print(forest_tr_pred, forest_val_pred)


# Первая модель - случайный лес. Его гиперпараметры: количество деревьев, минимальное количество выборок, необходимое для разделения, максимальная глубина дерева и минимальное количество деление ветвей. Приведенные данные вместе с выборками загружаются в функцию и в итоге мы получаем предсказание на обучающей и валидационной выборках, а также лучшую модель. 
# 
# Результат случайного леса равен 94.8 процентов на валидационной выборке, что выглядит очень хорошо. Следующая модель - логистическая регрессия:

# In[45]:


model_linear = LogisticRegression()
linear_tr_pred, linear_val_pred, best_model_linear = find_best_models(model_linear, 
                                                   data_train, data_valid, target_train, target_valid, 
                                                   params['params_linear'])
print(linear_tr_pred, linear_val_pred)


# Гиперпараметры логистической регресси следующие: сила регуляризации и алгоритм решения.
# 
# Результат - 95.6 процентов на валидационной. Итог выглядит лучше, чем предыдущий, однако давайте проверим последнуюю модель - градиентный бустинг:

# In[46]:


model_lgbm = LGBMClassifier()
lgbm_tr_pred, lgbm_val_pred, best_model_lgbm = find_best_models(model_lgbm, 
                                               data_train, data_valid, target_train, target_valid, 
                                               params['params_lgbm'])
print(lgbm_tr_pred, lgbm_val_pred)


# Гиперпараметры градиентного бустинга идентичны гиперпараметрам случайного леса, только в данном случае используются только количество деревьев и максимальная глубина. 
# 
# Результат максимальный - 99.4 процента, что является наилучшим результатом.
# 
# Модели обучены, результат получен, однако стоит проверить еще изменение качества после масштабирования данных. Поможет ли данный фикс улучшить наши модели, сейчас узнаем:

# In[47]:


scaled_data_train = data_train.copy()
scaled_data_valid = data_valid.copy()
num_cols = ['new_cases', 'hosp_patients', 'new_deaths', 'new_tests']

scaler = StandardScaler()
scaled_data_train[num_cols] = scaler.fit_transform(scaled_data_train[num_cols])
scaled_data_valid[num_cols] = scaler.fit_transform(scaled_data_valid[num_cols])


# В нашем случае было четыре столбца, которые имели огромные диапазоны значений: `new_cases`, `hosp_patients`, `new_deaths`, `new_tests`. Их мы и масштабировали, а затем закинули в функцию `find_best_model`. Ничего кроме выборок не менялось, поэтому комментировать ничего не будем:

# In[48]:


model_forest_scaled = RandomForestClassifier()
forest_scaled_tr_pred, forest_scaled_val_pred, best_model_forest_scaled = find_best_models(model_forest_scaled, 
                                                                 scaled_data_train, scaled_data_valid, 
                                                                 target_train, target_valid, 
                                                                 params['params_forest'])
print(forest_scaled_tr_pred, forest_scaled_val_pred)


# In[49]:


model_linear_scaled = LogisticRegression()
linear_scaled_tr_pred, linear_scaled_val_pred, best_model_linear_scaled = find_best_models(model_linear_scaled, 
                                                                 scaled_data_train, scaled_data_valid, 
                                                                 target_train, target_valid, 
                                                                 params['params_linear'])
print(linear_scaled_tr_pred, linear_scaled_val_pred)


# In[50]:


model_lgbm_scaled = LGBMClassifier()
lgbm_scaled_tr_pred, lgbm_scaled_val_pred, best_model_lgbm_scaled = find_best_models(model_lgbm_scaled, 
                                                             scaled_data_train, scaled_data_valid, 
                                                             target_train, target_valid, 
                                                             params['params_lgbm'])
print(lgbm_scaled_tr_pred, lgbm_scaled_val_pred)


# Как можно заметить, наши результаты стали хуже, в особенности логистическая регрессия, которая потеряла в качестве 22 процента. Масштабирование данных нам не нужно, оставим модели без него.
# 
# В итоге мы можем выбрать лучшую модель - градиентный бустинг без масштабирования данных.

# ## Предсказание новых случаев заражения COVID
# На данном этапе мы будем предсказывать новые случаи заражения COVID. Будут созданы следующие функции:
# - `make_features` - новые признаки, что помогут нам в предсказании: год, месяц, день недели, скользящее среднее по дням и количество дней, которые нужно будет предсказать.  
# - `do_make_features_to_country` - функция добавления новых признаков в таблицу
# - `split_data` - разделение новой таблицы на выборки (без перемешивания)
# - `train_models` - обучение моделей и поиск наилучшей. Также возвращает валидационные данные, в которых есть промежуток дней, которые необходимо предсказать
# - `prepare_data` - составление итоговой таблицы с предсказанием новых заражений
# - `find_danger_level_new_data` - выявляем уровень опасности для предсказанных значений

# In[51]:


predict_cases_data = data_ready[['location', 'new_cases', 'hosp_patients', 'new_deaths', 'new_tests', 'clusters']]


# Создана переменная `predict_cases_data`, содержащая необходимые для предсказания признаки. Следующим шагом будет создание функций, описанных выше:

# In[52]:


def make_features(data, country, max_lag, rolling_mean_size):
    new_data = data.query("location == @country")
    
    new_data['year'] = new_data.index.year
    new_data['month'] = new_data.index.month
    new_data['day_of_week'] = new_data.index.dayofweek
    
    for lag in range(1, max_lag+1):
        new_data['lag_{}'.format(lag)] = new_data['new_cases'].shift(lag)
    
    new_data['rolling_mean'] = new_data['new_cases'].shift().rolling(rolling_mean_size).mean()
    
    return new_data

def do_make_features_to_country(data, country, rolling_size, range_prediction):
    country_data = make_features(data, country, range_prediction, rolling_size)
    country_data = country_data.dropna()
    
    return country_data

def split_data(data):
    data_train, data_valid, target_train, target_valid = train_test_split(
    data.drop(['location', 'new_cases'], axis=1), 
    data['new_cases'],
    test_size=0.2, shuffle=False)
    
    return data_train, data_valid, target_train, target_valid

def train_models(data, params):
    model_linear = LinearRegression()
    model_forest = RandomForestRegressor()
    model_lgbm = LGBMRegressor()
    
    data_train, data_valid, target_train, target_valid = split_data(data)
    
    linear_reg_tr_pred, best_model_linear_reg = find_best_models(model_linear, 
                                                           data_train, data_valid, 
                                                           target_train, target_valid, 
                                                           params['params_linear_reg'], task_type='reg')
    
    forest_reg_tr_pred, best_model_forest_reg = find_best_models(model_forest, 
                                                           data_train, data_valid, 
                                                           target_train, target_valid, 
                                                           params['params_forest'], task_type='reg')    
    
    lgbm_reg_tr_pred, best_model_lgbm = find_best_models(model_lgbm, 
                                                           data_train, data_valid, 
                                                           target_train, target_valid, 
                                                           params['params_lgbm'], task_type='reg')
       
    models_results = {
        best_model_linear_reg:linear_reg_tr_pred,
        best_model_forest_reg:forest_reg_tr_pred,
        best_model_lgbm:lgbm_reg_tr_pred
    }
    
    best_model = min(models_results, key=lambda x: models_results[x])
    
    print(f"Наилучший результат модели {best_model} равен =", models_results[best_model])
    
    return best_model, data_valid

def prepare_data(data, params, country, max_lag, days_predict):
    country_data = do_make_features_to_country(data, country, max_lag, days_predict)
    best_model, prediction_must_data = train_models(country_data, params)
    
    predictions = best_model.predict(prediction_must_data)
    days = timedelta(days_predict)
    
    new_data = pd.DataFrame(data=predictions, columns=['prediction_new_cases'])
    new_data['location'] = country  
    new_data['date'] = data[data['location'] == country][-(len(new_data)):].index
    new_data['date'] += days

    new_data = new_data.set_index('date')
    new_data.sort_index(inplace=True)
    
    return new_data

def find_danger_level_new_data(data):
    gm_predicted_data_danger_level = GaussianMixture(n_components=3, random_state=49)
    data['danger_level'] = gm_predicted_data_danger_level.fit_predict(data['prediction_new_cases'].values.reshape(-1, 1))
    
    max_value = 0
    max_level = None
    
    min_value = 100000
    min_level = None
    
    for level in range(0,3):
        print(f"Максимальный коэффициент распространения инфекции в кластере {level}:", 
          data[data.danger_level == level]['prediction_new_cases'].max())
        
        if data[data.danger_level == level]['prediction_new_cases'].max() > max_value:
            max_value = data[data.danger_level == level]['prediction_new_cases'].max()
            max_level = level
            
        elif data[data.danger_level == level]['prediction_new_cases'].max() < min_value:
            min_value = data[data.danger_level == level]['prediction_new_cases'].max()
            min_level = level
    
    medium_level = data.query("danger_level not in [@max_level,@min_level]")['danger_level'][0]
    
    data['danger_level'] = data['danger_level'].replace(
    [data[data.danger_level == max_level]['danger_level'][0], 
     data[data.danger_level == medium_level]['danger_level'][0], 
     data[data.danger_level == min_level]['danger_level'][0]
    ], ['опасно', 'не советуем', 'безопасно'])
    
    return data


# Функции описаны и готовы к работе. Давайте создадим нашу таблицу с результатами предсказания по стране Таиланд на 30 дней вперед:

# In[53]:


country_predict = 'Thailand'
days_predict = 30
max_lag = 20

infection_data_country = prepare_data(predict_cases_data, params, country_predict, max_lag, days_predict)


# In[54]:


infection_data_country.info()


# Данные записаны в переменную `infection_data_country` и содержат два столбца: `prediction_new_cases` - новые случаи заражения COVID и `location` - изучаемая страна. На основе полученных данных стоит рассмотреть график заражения:

# In[55]:


sns.lineplot(data=infection_data_country[-days_predict:], x=infection_data_country[-days_predict:].index, y='prediction_new_cases')


# В качестве `X` выступает дата, а в качестве `Y` - новые случаи заражения. По графику видно, что в прогнозе есть редкие спады заражения, но сразу после них быстрый подъем (возможно это ошибка модели), который говорит либо о качестве вакцинации, либо о уменьшении заражаемости за счет необъяснимых факторов. 
# 
# Для того, чтобы наши данные были полными необходимо предсказать уровень опасности предсказанных значений. Для этого воспользуемся функцией `find_danger_level_new_data`, которая вернет нам таблицу с новым признаком - `danger_level`, предсказанный алгоритмом кластеризации - `GaussianMixture`:

# In[56]:


infection_data_country = find_danger_level_new_data(infection_data_country)
infection_data_country.head(10)


# ## Заключение
# В заключении хотелось бы привести основные моменты проекта:
# - Выполена загрузка данных с github
# - Совершен предварительный анализ данных, который выявил: 
#     - общее количество строк - 170 тысяч
#     - наибольшее количество пропусков - 163 тысячи
#     - присутствует множество признаков, которые нам не понадобятся при прогнозе 
#     - удаление пропусков не приведет ни к чему лучшему, поэтому мы их заменим
# - Выполнена предобработка данных:
#     - "не страны" были удалены со столбца `location`
#     - пропуски в `hosp_patients`, `people_fully_vaccinated_per_hundred`, `new_tests`, `reproduction_rate`, `new_deaths`, `new_cases` были заполнены либо 0, либо предыдущими значениями
#     - `population_density` был заполен -1 (является заполнителем)
#     - формат занчений столбца `date` был заменен с `object` на `date`
# - Создан новый признак - `rt_value` - коэффициент заражения COVID, вычисляемый по особой формуле
# - Таблица была усечена до значения `rt_value` <= 2.5 
# - Этап кластеризации был выполнен с применением двух алгоритмов: `KMeans` и `GaussianMixture`, среди которых лучшим оказался `GaussianMixture`. Алгоритмы предсказывали уровень опасности стран в определенный день записи
# - Для предсказания новых случаев заражения необходимо было первым делом определить наиболее коррелируемые признаки с столбцу `new_cases`. В ходе построения таблиц корреляции выявилось, что столбцы: `rt_value`, `new_cases`, `hosp_patients`, `new_deaths`, `new_tests`, `reproduction_rate` хорошо коррелируются
# - Далее была решена задача классификации для выявления уровня опасности старых значений. Для этого была создана функция предсказания, инициализированы модели следующих алгоритмов: `RandomForest`, `LogisticRegression` и `LightGradientBoosting`, среди которых `LightGradientBoosting` показал наилучший результат. 
# - Потом последовала задача регрессии - прогноз будущих заражений для определенной страны в определенном диапазоне. В качестве примера был выбран Таиланд с прогнозом в 30 дней. Наилучшей моделью также оказался `LightGradientBoosting`.
# - Последним этапом стало добавление в прогноз новых случаев столбец уровня опасности страны в определенную дату. Для этого была использована функция `find_danger_level_new_data`, которая вернула новую таблицу, содержащую необходимый столбец.

# In[ ]:





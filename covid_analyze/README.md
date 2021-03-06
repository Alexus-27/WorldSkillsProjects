# Диагностика эпидемиологической ситуации в странах мира
В данном проекте было дано задание на анализ всемирного датасета о короновирусной инфекции, который содержал в себе данные о различных странах (площадь, количество новых зараженных, количество вакцинированных и т.д. ). 

Главная цель - спрогнозировать новые случаи COVID на будущие дни (предсказать распространение инфекции), а также визуализировать данные. Также было необходимо составить уровень опасности страны, которые указывали бы на то, опасно ли прилетать в страну в данную дату или нет.

Итогом всей работы стали данные об уровне опасности и о заражении в будущие дни.

В работе были использованы алгоритмы кластеризации:
- KMeans
- GaussianMixture

Также были задействованы алгоритмы машинного обучения:
- RandomForest
- Linear/Logistic Regression
- LightGradientBoosting

Основные метрики качества:
- Accuracy 
- Roc Auc для многоклассовой классификации
- Mean Squared Error

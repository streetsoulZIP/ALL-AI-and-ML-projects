## Краткая таблица проектов

| Проект | Ключевая задача | Основные технологии | Результат (метрики на тесте) | Ссылка на модель/данные, что не поместились сюда  |
|-------|------------------|----------------------|------------------------------|-------------------------|
| **Семантический поиск (retriever)** | Семантический поиск фильмов по запросам пользователей на 10 языках | Pruning/quantization, transfer-learning, fine-tuning | Triplet loss (margin=0.3) = **0.23**. Уменьшение модели на **63%** без потери качества, real-time inference на CPU (int8) | [Kaggle](https://www.kaggle.com/datasets/kehhill/queries) |
| **Генерация данных с LLM** | Генерация описаний фильмов и пользовательских запросов | Prompt-engineering, autocast, transformers, CUDA, Qwen-4B, кластер GPU | **70k** натуральных запросов, **6k** структурированных описаний | – |
| **Классификация и анализ табличных данных** | Предсказать, получит ли студент работу + определить оптимальную стратегию получения места | Scikit-learn, cuML, data preprocessing, data analysis, кластеризация, Optuna | Выявлены топ-параметры и **5 стратегий**, найдена самая успешная | – |
| **Классификация текста** | Классификация русскоязычных статей | Optuna, XGBoost, Scikit-learn, cuML | F1 = **0.86** на логистической регрессии (TF-IDF) | – |
| **Временные ряды** | Прогноз спроса на товары | CatBoost, data analysisis, preprocessing, Optuna | RMSE = **0.0233**, MSE = **0.0117**, R² = **0.9995** (CatBoost) | – |
| **Классификация изображений** | Определение вида спорта по изображению | Torchvision, DenseNet | F1-score = **0.8623** | [Google Drive](https://drive.google.com/file/d/1z9X221ryPWBVLtThGKzyDy0c2mZ_C_nb/view?usp=drive_link) |
| **Семантическая сегментация изображений** | Сегментация подводных изображений | Torchvision, DeepLab v3, data augmentation | mIoU = **0.6813**, Dice = **0.8425**, PA = **0.8425**, mPA = **0.7443** (DeepLab v3 fine-tune) | [Google Drive](https://drive.google.com/file/d/170qd7ajPYatU8iOCPCDvxtt3hFb7cNFL/view?usp=drive_link) |
| **Регрессия табличных данных** | Предсказание энергоэффективности строения | Scikit-learn, data preprocessing, analysis, Optuna | MAE = **0.25**, R² = **0.9984** (LightGBM) | – |

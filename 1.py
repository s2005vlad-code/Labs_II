import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА №1: ПРЕДОБРАБОТКА ДАННЫХ")
print("ДАТАСЕТ: Epic Games Free Giveaway History (2018–2025)")
print("=" * 60)

# 1. Загрузка данных
print("\n1. ЗАГРУЗКА ДАННЫХ")
df = pd.read_csv('epic games data.csv')  # Предполагаем, что файл так называется
print(f"   ✓ Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(f"   ✓ Столбцы: {list(df.columns)}")

# 2. Вывод данных
print("\n2. ВЫВОД ПЕРВЫХ 10 СТРОК:")
print(df.head(10))

# 3. Информация о датасете
print("\n3. ИНФОРМАЦИЯ О ДАТАСЕТЕ:")
print(df.info())

# 4. Пропущенные значения
print("\n4. ПРОВЕРКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ:")
missing_values = df.isnull().sum()
print(missing_values)

# Создаем новые числовые признаки на основе существующих данных
print("\n5. СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ ДЛЯ АНАЛИЗА...")

# Преобразуем даты в формат datetime и создаем числовые признаки
df['Start'] = pd.to_datetime(df['Start'], format='%d-%m-%Y', errors='coerce')
df['End'] = pd.to_datetime(df['End'], format='%d-%m-%Y', errors='coerce')

# Создаем числовые признаки:
df['Game_Duration'] = (df['End'] - df['Start']).dt.days  # Длительность раздачи в днях
df['Year'] = df['Start'].dt.year  # Год начала раздачи
df['Month'] = df['Start'].dt.month  # Месяц начала раздачи
df['Weekday'] = df['Start'].dt.weekday  # День недели (0=Понедельник)

# Длина названия игры как простой числовой признак
df['Name_Length'] = df['Game'].str.len()

# Создаем бинарные признаки на основе жанра
genre_keywords = ['RPG', 'Horror', 'Action', 'Strategy', 'Puzzle', 'Adventure', 'Simulator']
for keyword in genre_keywords:
    df[f'Genre_{keyword}'] = df['Genre'].str.contains(keyword, case=False, na=False).astype(int)

# Удаляем оригинальные столбцы дат, оставляем только числовые
df_clean = df.drop(['Start', 'End'], axis=1)

# Создаем целевую переменную для классификации (пример)
df_clean['Is_Popular_Genre'] = ((df['Genre'].str.contains('Action|RPG|Adventure', case=False, na=False)) &
                                (df['Game_Duration'] > 7)).astype(int)

# 5. Заполнение пропущенных значений
print("\n6. ЗАПОЛНЕНИЕ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ:")

numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df_clean.select_dtypes(exclude=['int64', 'float64']).columns

print(f"   Числовых колонок: {len(numeric_cols)}")
print(f"   Категориальных колонок: {len(categorical_cols)}")

# Заполняем пропуски в числовых колонках медианой
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)
        print(f"   {col}: заполнено медианой ({median_val:.2f})")

# Заполняем пропуски в категориальных колонках модой
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
        df_clean[col].fillna(mode_val, inplace=True)
        print(f"   {col}: заполнено модой ('{mode_val}')")

# Проверка заполнения
missing_after = df_clean.isnull().sum().sum()
print(f"\n   ✓ Пропусков до: {missing_values.sum()}")
print(f"   ✓ Пропусков после: {missing_after}")

# 6. Нормализация числовых данных
print("\n7. НОРМАЛИЗАЦИЯ ЧИСЛОВЫХ ДАННЫХ (MinMaxScaler):")

if len(numeric_cols) > 0:
    scaler = MinMaxScaler()
    df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

    print(f"   Нормализовано колонок: {len(numeric_cols)}")
    print(f"   Диапазон после нормализации: [0, 1]")

    # Показываем пример
    sample_col = numeric_cols[0] if len(numeric_cols) > 0 else None
    if sample_col:
        print(f"   Пример для '{sample_col}':")
        print(f"   Мин: {df_clean[sample_col].min():.3f}, Макс: {df_clean[sample_col].max():.3f}")
else:
    print("   Нет числовых колонок для нормализации")

# 7. Преобразование категориальных данных
print("\n8. ПРЕОБРАЗОВАНИЕ КАТЕГОРИАЛЬНЫХ ДАННЫХ:")

# One-Hot Encoding для столбца 'Genre' (основной категориальный признак)
if 'Genre' in df_clean.columns:
    # Сначала ограничим количество уникальных значений для разумного количества новых колонок
    top_genres = df_clean['Genre'].value_counts().head(10).index.tolist()
    df_clean['Genre_Grouped'] = df_clean['Genre'].apply(
        lambda x: x if x in top_genres else 'Other'
    )

    # Применяем One-Hot Encoding
    genre_dummies = pd.get_dummies(df_clean['Genre_Grouped'], prefix='Genre')
    df_clean = pd.concat([df_clean, genre_dummies], axis=1)
    df_clean = df_clean.drop(['Genre', 'Genre_Grouped'], axis=1)

    print(f"   One-Hot Encoding для жанров: создано {len(genre_dummies.columns)} бинарных колонок")
    print(f"   Уникальные значения: {top_genres}")

# One-Hot Encoding для других категориальных колонок с малым количеством уникальных значений
categorical_cols_remaining = df_clean.select_dtypes(exclude=['int64', 'float64']).columns
for col in categorical_cols_remaining:
    if df_clean[col].nunique() <= 5:  # Только если мало уникальных значений
        dummies = pd.get_dummies(df_clean[col], prefix=col)
        df_clean = pd.concat([df_clean, dummies], axis=1)
        df_clean = df_clean.drop([col], axis=1)
        print(f"   One-Hot Encoding для '{col}': создано {len(dummies.columns)} колонок")

# 8. Разделение на обучающую и тестовую выборки
print("\n9. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ:")

# Создаем целевую переменную для регрессии (продолжительность игры как пример)
X = df_clean.drop(['Is_Popular_Genre', 'Game_Duration', 'Game'], axis=1, errors='ignore')
if 'Game' in df_clean.columns:
    X = df_clean.drop(['Is_Popular_Genre', 'Game_Duration', 'Game'], axis=1, errors='ignore')
else:
    X = df_clean.drop(['Is_Popular_Genre', 'Game_Duration'], axis=1, errors='ignore')

y_classification = df_clean['Is_Popular_Genre']
y_regression = df_clean['Game_Duration']

# Разделение для классификации
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_classification, test_size=0.3, random_state=42, stratify=y_classification
)

# Разделение для регрессии
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_regression, test_size=0.3, random_state=42
)

print(f"   Размеры выборок (классификация):")
print(f"   - X_train: {X_train_clf.shape}")
print(f"   - X_test:  {X_test_clf.shape}")
print(f"   - y_train: {y_train_clf.shape}")
print(f"   - y_test:  {y_test_clf.shape}")

print(f"\n   Размеры выборок (регрессия):")
print(f"   - X_train: {X_train_reg.shape}")
print(f"   - X_test:  {X_test_reg.shape}")
print(f"   - y_train: {y_train_reg.shape}")
print(f"   - y_test:  {y_test_reg.shape}")

# 9. Сохранение обработанных данных
print("\n10. СОХРАНЕНИЕ ОБРАБОТАННЫХ ДАННЫХ:")

# Сохраняем полный обработанный датасет
df_clean.to_csv('processed_epic_games.csv', index=False)
print("   ✓ Полный обработанный датасет: 'processed_epic_games.csv'")

# Сохраняем разделенные выборки
train_data_clf = pd.concat([X_train_clf, y_train_clf], axis=1)
test_data_clf = pd.concat([X_test_clf, y_test_clf], axis=1)
train_data_reg = pd.concat([X_train_reg, y_train_reg], axis=1)
test_data_reg = pd.concat([X_test_reg, y_test_reg], axis=1)

train_data_clf.to_csv('train_classification.csv', index=False)
test_data_clf.to_csv('test_classification.csv', index=False)
train_data_reg.to_csv('train_regression.csv', index=False)
test_data_reg.to_csv('test_regression.csv', index=False)

print("   ✓ Обучающая выборка (классификация): 'train_classification.csv'")
print("   ✓ Тестовая выборка (классификация): 'test_classification.csv'")
print("   ✓ Обучающая выборка (регрессия): 'train_regression.csv'")
print("   ✓ Тестовая выборка (регрессия): 'test_regression.csv'")

# 10. Итоговый отчет
print("\n11. ИТОГОВЫЙ ОТЧЕТ:")
print(f"   Исходный датасет: {df.shape[0]} строк × {df.shape[1]} столбцов")
print(f"   Обработанный датасет: {df_clean.shape[0]} строк × {df_clean.shape[1]} столбцов")
print(f"   Числовых признаков: {len(df_clean.select_dtypes(include=['int64', 'float64']).columns)}")
print(f"   Целевая переменная для классификации: 'Is_Popular_Genre'")
print(f"   Распределение классов: {dict(df_clean['Is_Popular_Genre'].value_counts())}")
print(f"   Целевая переменная для регрессии: 'Game_Duration' (дни)")

print("\n12. ПЕРВЫЕ 5 СТРОК ОБРАБОТАННОГО ДАТАСЕТА:")
print(df_clean.head())

print("\n" + "=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА №1 ВЫПОЛНЕНА УСПЕШНО!")
print("=" * 60)
print("\nСозданные файлы:")
print("   1. processed_epic_games.csv - полный обработанный датасет")
print("   2. train_classification.csv - обучающая выборка для классификации")
print("   3. test_classification.csv - тестовая выборка для классификации")
print("   4. train_regression.csv - обучающая выборка для регрессии")
print("   5. test_regression.csv - тестовая выборка для регрессии")

# Показываем информацию о типах данных
print("\nТипы данных в обработанном датасете:")
print(df_clean.dtypes.value_counts())
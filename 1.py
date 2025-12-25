import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА №1: ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 60)

# 1. Загрузка данных
print("\n1. ЗАГРУЗКА ДАННЫХ С KAGGLE")
df = pd.read_csv('epic games data.csv')
print(f"   ✓ Файл: 'epic games data.csv'")
print(f"   ✓ Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")

# 2. Вывод данных
print("\n2. ВЫВОД ДАННЫХ (первые 5 строк):")
print(df.head())

# 3. Пропущенные значения
print("\n3. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:")
missing_before = df.isnull().sum()
print(missing_before[missing_before > 0])

numeric_cols = df.select_dtypes(include='number').columns
categorical_cols = df.select_dtypes(exclude='number').columns

# 4. Заполнение пропусков
print("\n4. ЗАПОЛНЕНИЕ ПРОПУСКОВ:")
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"   {col}: медиана = {median_val:.2f}")

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"   {col}: мода = '{mode_val}'")

# Проверка заполнения
missing_after = df.isnull().sum().sum()
print(f"\n   Проверка: было {missing_before.sum()} пропусков, стало {missing_after}")

# 5. Нормализация
print("\n5. НОРМАЛИЗАЦИЯ ДАННЫХ (MinMaxScaler):")
if len(numeric_cols) > 0:
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"   Нормализовано колонок: {len(numeric_cols)}")
    print(f"   Колонки: {list(numeric_cols)}")
else:
    print("   Нет числовых колонок для нормализации")

# 6. Преобразование категориальных данных
print("\n6. ПРЕОБРАЗОВАНИЕ КАТЕГОРИАЛЬНЫХ ДАННЫХ:")
categorical_to_encode = []
for col in categorical_cols:
    unique_count = df[col].nunique()
    if 1 < unique_count < 10:
        categorical_to_encode.append(col)
        print(f"   {col}: {unique_count} уникальных значений → One-Hot Encoding")

if categorical_to_encode:
    df = pd.get_dummies(df, columns=categorical_to_encode, drop_first=True)
    print(f"   Создано новых бинарных колонок")
else:
    print("   Нет подходящих категориальных колонок для кодирования")

# Разделение на train/test
print("\n7. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ:")
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
print(f"   Обучающая выборка: {train_df.shape[0]} строк ({train_df.shape[0]/df.shape[0]*100:.1f}%)")
print(f"   Тестовая выборка: {test_df.shape[0]} строк ({test_df.shape[0]/df.shape[0]*100:.1f}%)")

# Сохранение
print("\n8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ:")
train_df.to_csv('train_epic_games.csv', index=False)
test_df.to_csv('test_epic_games.csv', index=False)
df.to_csv('processed_epic_games.csv', index=False)
print("   ✓ train_epic_games.csv")
print("   ✓ test_epic_games.csv")
print("   ✓ processed_epic_games.csv")

# 9. Итоговый вид датасета
print("\n9. ИТОГОВЫЙ ВИД ДАТАСЕТА:")
print(f"   Размер: {df.shape[0]} строк × {df.shape[1]} столбцов")
print(f"   Типы данных:")
print(f"   - Числовые: {len(df.select_dtypes(include='number').columns)}")
print(f"   - Категориальные (бинарные): {len(df.select_dtypes(include='bool').columns)}")
print(f"   - Прочие: {len(df.select_dtypes(exclude=['number', 'bool']).columns)}")

print("\n10. ПЕРВЫЕ 3 СТРОКИ ОБРАБОТАННОГО ДАТАСЕТА:")
print(df.head(3))

print("\n" + "=" * 60)
print("ЗАДАНИЕ ВЫПОЛНЕНО! Все этапы предобработки завершены.")
print("=" * 60)

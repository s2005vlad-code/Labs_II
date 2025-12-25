import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

print("=" * 50)
print("ЛАБОРАТОРНАЯ РАБОТА №1: ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 50)

print("\n1. ЗАГРУЗКА ДАННЫХ")
df = pd.read_csv('epic games data.csv')
print(f"   ✓ Датсет загружен: {df.shape[0]} строк, {df.shape[1]} столбцов")

print("\n2. ПЕРВЫЕ 5 СТРОК:")
print(df.head())

print("\n3. ИНФОРМАЦИЯ О ДАТАСЕТЕ:")
print(df.info())

print("\n4. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:")
missing = df.isnull().sum()
print(missing[missing > 0])

numeric_cols = df.select_dtypes(include='number').columns
categorical_cols = df.select_dtypes(exclude='number').columns

print(f"\n5. ЧИСЛОВЫЕ КОЛОНКИ: {list(numeric_cols)}")
print(f"6. КАТЕГОРИАЛЬНЫЕ КОЛОНКИ: {list(categorical_cols)}")

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\n7. ПРОПУСКИ ПОСЛЕ ЗАПОЛНЕНИЯ:")
print(df.isnull().sum().sum(), "пропусков осталось")

if len(numeric_cols) > 0:
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print("\n8. НОРМАЛИЗАЦИЯ ВЫПОЛНЕНА")

categorical_to_encode = []
for col in categorical_cols:
    if df[col].nunique() < 15 and df[col].nunique() > 1:
        categorical_to_encode.append(col)

if categorical_to_encode:
    df = pd.get_dummies(df, columns=categorical_to_encode, drop_first=True)
    print(f"9. ONE-HOT ENCODING: {categorical_to_encode}")

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
print(f"\n10. РАЗДЕЛЕНИЕ:")
print(f"    Обучающая: {train_df.shape}")
print(f"    Тестовая: {test_df.shape}")

train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
df.to_csv('processed_data.csv', index=False)

print("\n11. ФАЙЛЫ СОХРАНЕНЫ:")
print("    processed_data.csv - полный обработанный датасет")
print("    train_data.csv - обучающая выборка")
print("    test_data.csv - тестовая выборка")

print("\n" + "=" * 50)
print("ПРЕДОБРАБОТКА ЗАВЕРШЕНА!")
print("=" * 50)
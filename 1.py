import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

print("=" * 60)
print("Лабораторная работа №1 - Предобработка данных Epic Games")
print("=" * 60)

games = []
for i in range(1, 541):
    games.append(f"Game_{i}")

np.random.seed(42)
start_dates = []
end_dates = []

for _ in range(540):
    year = np.random.choice([2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
    month = np.random.randint(1, 13)
    day = np.random.randint(1, 29)
    
    start_date = f"{day:02d}-{month:02d}-{year}"
    duration = np.random.randint(7, 31)
    end_date_day = day + duration
    end_date_month = month
    end_date_year = year
    
    if end_date_day > 28:
        end_date_day = end_date_day - 28
        end_date_month += 1
        if end_date_month > 12:
            end_date_month = 1
            end_date_year += 1
    
    end_date = f"{end_date_day:02d}-{end_date_month:02d}-{end_date_year}"
    
    start_dates.append(start_date)
    end_dates.append(end_date)

genres = ['Action', 'Adventure', 'RPG', 'Strategy', 'Simulation', 
          'Sports', 'Puzzle', 'Horror', 'Indie', 'AAA']

df = pd.DataFrame({
    'serial_no': range(1, 541),
    'game': games,
    'start_date': start_dates,
    'end_date': end_dates,
    'genre': np.random.choice(genres, 540),
    'duration_days': np.random.randint(7, 31, 540),
    'popularity_score': np.random.uniform(1.0, 10.0, 540),
    'price_usd': np.random.choice([0, 9.99, 19.99, 29.99, 59.99], 540, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
    'user_rating': np.random.uniform(3.0, 5.0, 540),
    'downloads_k': np.random.randint(10, 1000, 540),
    'is_holiday_release': np.random.choice([0, 1], 540, p=[0.7, 0.3]),
    'has_multiplayer': np.random.choice([0, 1], 540, p=[0.6, 0.4])
})

print(f"Данные: {df.shape[0]} строк, {df.shape[1]} столбцов")
print("\nПервые 5 строк:")
print(df.head())

np.random.seed(42)
missing_mask = np.random.random(df.shape) < 0.05

missing_mask[:, 0] = False
missing_mask[:, 1] = False

for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        if missing_mask[i, j] and df.iloc[i, j] is not None:
            df.iloc[i, j] = np.nan

print("\nПропуски в данных:")
print(df.isnull().sum())

print("\nИнформация о данных:")
print(df.info())

print(f"\nУникальных жанров: {df['genre'].nunique()}")

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
print("\nСтатистика числовых данных:")
print(df[numeric_cols].describe())

print("\nЗаполнение пропусков...")
missing_before = df.isnull().sum()
print("Было пропусков:")
print(missing_before[missing_before > 0])

numeric_cols_missing = [col for col in numeric_cols if df[col].isnull().sum() > 0]
for col in numeric_cols_missing:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

categorical_cols = df.select_dtypes(include=['object']).columns
categorical_cols_missing = [col for col in categorical_cols if df[col].isnull().sum() > 0]

for col in categorical_cols_missing:
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)

missing_after = df.isnull().sum().sum()
print(f"\nОсталось пропусков: {missing_after}")

try:
    df['start_date_dt'] = pd.to_datetime(df['start_date'], format='%d-%m-%Y', errors='coerce')
    df['end_date_dt'] = pd.to_datetime(df['end_date'], format='%d-%m-%Y', errors='coerce')
    
    df['start_year'] = df['start_date_dt'].dt.year
    df['start_month'] = df['start_date_dt'].dt.month
    df['start_day'] = df['start_date_dt'].dt.day
    df['start_weekday'] = df['start_date_dt'].dt.weekday
    
    print("\nДаты преобразованы")
except Exception as e:
    print(f"\nОшибка с датами: {e}")

cols_to_normalize = ['popularity_score', 'price_usd', 'user_rating', 'downloads_k', 'duration_days']

if len(cols_to_normalize) > 0:
    scaler = MinMaxScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    print(f"\nНормализовано столбцов: {len(cols_to_normalize)}")
    print(f"Столбцы: {cols_to_normalize}")

if df['genre'].nunique() <= 15:
    genre_dummies = pd.get_dummies(df['genre'], prefix='genre', drop_first=True)
    df = pd.concat([df, genre_dummies], axis=1)
    print(f"\nOne-Hot Encoding для жанра: {genre_dummies.shape[1]} новых столбцов")
else:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['genre_encoded'] = le.fit_transform(df['genre'])
    print(f"\nLabel Encoding для жанра: {len(le.classes_)} значений")

features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
features = [f for f in features if f not in ['serial_no', 'popularity_score', 'user_rating', 'genre_encoded']]

if len(features) > 0:
    X = df[features]
    y = df['popularity_score']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=True
    )
    
    print(f"\nПризнаков: {len(features)}")
    print(f"Обучающая выборка: {X_train.shape[0]} строк")
    print(f"Тестовая выборка: {X_test.shape[0]} строк")

df.to_csv('processed_epic_games_full.csv', index=False, encoding='utf-8')

if len(features) > 0:
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv('train_epic_games.csv', index=False)
    test_df.to_csv('test_epic_games.csv', index=False)
    
    print("\nСохранены файлы:")
    print("processed_epic_games_full.csv")
    print("train_epic_games.csv")
    print("test_epic_games.csv")
else:
    print("\nСохранен файл:")
    print("processed_epic_games_full.csv")

print("\n" + "=" * 60)
print("Предобработка данных завершена!")
print("=" * 60)

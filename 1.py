# Лабораторная работа №3: Деревья решений в задачах классификации и регрессии. ROC-кривая

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score, accuracy_score
)
import warnings

warnings.filterwarnings('ignore')

# Установка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
print("=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА №3: ДЕРЕВЬЯ РЕШЕНИЙ")
print("=" * 70)

# Загрузка данных - ИСПРАВЛЕНО ИМЯ ФАЙЛА!
try:
    # Попробуем загрузить обработанные данные из лабы 1
    data = pd.read_csv('processed_epic_games.csv')
    print("✓ Загружены обработанные данные из processed_epic_games.csv")
except:
    # Если нет, загружаем исходные данные
    data = pd.read_csv('epic games data.csv')
    print("✓ Загружены исходные данные из epic games data.csv")

# Просмотр первых строк
print("\nПервые 5 строк данных:")
print(data.head())
print(f"\nРазмер датасета: {data.shape}")
print(f"\nСтолбцы: {list(data.columns)}")
print(f"\nТипы данных:\n{data.dtypes}")

# 2. ПРЕОБРАБОТКА ДАННЫХ
print("\n" + "=" * 70)
print("ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 70)

# Если в данных есть столбцы Start и End (даты), преобразуем их
date_columns = ['Start', 'End']
date_columns_present = [col for col in date_columns if col in data.columns]

if 'Start' in data.columns and 'End' in data.columns:
    # Преобразование дат
    data['Start'] = pd.to_datetime(data['Start'], format='%d-%m-%Y', errors='coerce')
    data['End'] = pd.to_datetime(data['End'], format='%d-%m-%Y', errors='coerce')

    # Создание целевых переменных
    # Для регрессии: продолжительность в днях
    data['Duration_days'] = (data['End'] - data['Start']).dt.days

    # Для классификации: является ли игра AAA (1) или инди (0)
    # Определим по ключевым словам в жанре
    if 'Genre' in data.columns:
        aaa_keywords = ['AAA', 'AAA bundle', 'big AAA', 'major AAA', 'open-world',
                        'AAA fantasy', 'AAA story', 'AAA open-world']
        data['Is_AAA'] = data['Genre'].apply(
            lambda x: 1 if any(keyword.lower() in str(x).lower() for keyword in aaa_keywords) else 0
        )

    # Создание дополнительных признаков
    data['Year'] = data['Start'].dt.year
    data['Month'] = data['Start'].dt.month
    data['Day'] = data['Start'].dt.day
    data['Weekday'] = data['Start'].dt.weekday

    # Кодирование жанра
    if 'Genre' in data.columns:
        le_genre = LabelEncoder()
        data['Genre_encoded'] = le_genre.fit_transform(data['Genre'].fillna('Unknown'))

    # Длина названия игры как признак
    if 'Game' in data.columns:
        data['Name_length'] = data['Game'].apply(lambda x: len(str(x)))

    print("✓ Созданы временные признаки и целевые переменные")
else:
    print("⚠ В данных нет столбцов Start и End для создания признаков")
    print("  Используем доступные числовые признаки")

    # Создадим искусственные целевые переменные для демонстрации
    np.random.seed(42)
    data['Duration_days'] = np.random.randint(1, 30, size=len(data))
    data['Is_AAA'] = np.random.randint(0, 2, size=len(data))

# Подготовка признаков
# Выбираем числовые столбцы для признаков
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Удаляем целевые переменные из признаков, если они уже есть
if 'Duration_days' in numeric_features:
    numeric_features.remove('Duration_days')
if 'Is_AAA' in numeric_features:
    numeric_features.remove('Is_AAA')

# Также удалим serial no. если он есть
if 'serial no.' in numeric_features:
    numeric_features.remove('serial no.')

# Если числовых признаков мало, добавим созданные
if len(numeric_features) < 3:
    # Используем созданные признаки
    features_to_use = ['Year', 'Month', 'Day', 'Weekday', 'Genre_encoded', 'Name_length']
    features = [col for col in features_to_use if col in data.columns]

    # Если и их нет, создадим искусственные
    if len(features) < 2:
        for i in range(5):
            data[f'Feature_{i}'] = np.random.randn(len(data))
        features = [f'Feature_{i}' for i in range(5)]
else:
    features = numeric_features[:5]  # Берем первые 5 числовых признаков

print(f"\nИспользуемые признаки: {features}")

X = data[features]

# Целевые переменные
y_reg = data['Duration_days']  # Для регрессии
y_clf = data['Is_AAA']  # Для классификации

print(f"\nРаспределение классов для классификации:")
print(y_clf.value_counts())
if len(y_clf.unique()) > 1:
    print(f"Процент AAA-игр: {y_clf.mean() * 100:.2f}%")
else:
    print("⚠ Только один класс в целевой переменной!")

print(f"\nСтатистика по целевой переменной регрессии (Duration_days):")
print(y_reg.describe())

# Удаление пропущенных значений
data_clean = data.dropna(subset=features + ['Duration_days', 'Is_AAA'], how='any')
X = X.loc[data_clean.index]
y_reg = y_reg.loc[data_clean.index]
y_clf = y_clf.loc[data_clean.index]

print(f"\nПосле очистки от пропусков: {len(X)} строк")

# 3. РАЗДЕЛЕНИЕ ДАННЫХ
print("\n" + "=" * 70)
print("РАЗДЕЛЕНИЕ ДАННЫХ НА ВЫБОРКИ")
print("=" * 70)

# Для регрессии
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.3, random_state=42
)

# Для классификации
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.3, random_state=42, stratify=y_clf if len(y_clf.unique()) > 1 else None
)

print(f"Размеры выборок для регрессии:")
print(f"  Обучающая: {X_train_reg.shape}, Тестовая: {X_test_reg.shape}")
print(f"Размеры выборок для классификации:")
print(f"  Обучающая: {X_train_clf.shape}, Тестовая: {X_test_clf.shape}")

# 4. ЗАДАЧА РЕГРЕССИИ
print("\n" + "=" * 70)
print("РЕШЕНИЕ ЗАДАЧИ РЕГРЕССИИ ДЕРЕВОМ РЕШЕНИЙ")
print("=" * 70)

# Создание и обучение модели
regressor = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
regressor.fit(X_train_reg, y_train_reg)

# Предсказание
y_pred_reg = regressor.predict(X_test_reg)

# Оценка модели
mse = mean_squared_error(y_test_reg, y_pred_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print("\nОценка модели регрессии:")
print(f"  Среднеквадратичная ошибка (MSE): {mse:.2f}")
print(f"  Средняя абсолютная ошибка (MAE): {mae:.2f}")
print(f"  Коэффициент детерминации (R²): {r2:.4f}")

# Визуализация дерева регрессии
try:
    plt.figure(figsize=(20, 10))
    plot_tree(
        regressor,
        feature_names=features,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=3
    )
    plt.title("Дерево решений для задачи регрессии (глубина 3)", fontsize=16)
    plt.tight_layout()
    plt.savefig('regression_tree.png', dpi=150, bbox_inches='tight')
    print("\n✓ Визуализация дерева сохранена в 'regression_tree.png'")
except Exception as e:
    print(f"\n⚠ Не удалось сохранить дерево: {e}")

# График фактических vs предсказанных значений
plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()],
         [y_test_reg.min(), y_test_reg.max()],
         'r--', lw=2, label='Идеальная линия')
plt.xlabel('Фактическая продолжительность (дни)', fontsize=12)
plt.ylabel('Предсказанная продолжительность (дни)', fontsize=12)
plt.title('Фактические vs Предсказанные значения (регрессия)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regression_scatter.png', dpi=150)
print("✓ График сравнения сохранен в 'regression_scatter.png'")

# 5. ЗАДАЧА КЛАССИФИКАЦИИ
print("\n" + "=" * 70)
print("РЕШЕНИЕ ЗАДАЧИ КЛАССИФИКАЦИИ ДЕРЕВОМ РЕШЕНИЙ")
print("=" * 70)

# Проверяем, что есть как минимум 2 класса
if len(y_clf.unique()) > 1:
    # Создание и обучение модели
    classifier = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'  # Для учета дисбаланса классов
    )
    classifier.fit(X_train_clf, y_train_clf)

    # Предсказание
    y_pred_clf = classifier.predict(X_test_clf)
    y_pred_proba = classifier.predict_proba(X_test_clf)[:, 1]  # Вероятности для класса 1

    # Оценка модели
    accuracy = accuracy_score(y_test_clf, y_pred_clf)
    conf_matrix = confusion_matrix(y_test_clf, y_pred_clf)
    class_report = classification_report(y_test_clf, y_pred_clf,
                                         target_names=['Инди', 'AAA'] if len(y_clf.unique()) == 2 else None)

    print("\nОценка модели классификации:")
    print(f"  Точность (accuracy): {accuracy:.4f}")
    print(f"\nМатрица ошибок:")
    print(conf_matrix)
    print(f"\nОтчет по классификации:")
    print(class_report)

    # Визуализация дерева классификации
    try:
        plt.figure(figsize=(20, 10))
        plot_tree(
            classifier,
            feature_names=features,
            class_names=['Инди', 'AAA'] if len(y_clf.unique()) == 2 else None,
            filled=True,
            rounded=True,
            fontsize=8,
            proportion=True
        )
        plt.title("Дерево решений для задачи классификации", fontsize=16)
        plt.tight_layout()
        plt.savefig('classification_tree.png', dpi=150, bbox_inches='tight')
        print("\n✓ Визуализация дерева сохранена в 'classification_tree.png'")
    except Exception as e:
        print(f"\n⚠ Не удалось сохранить дерево: {e}")

    # Визуализация матрицы ошибок
    plt.figure(figsize=(8, 6))
    if len(y_clf.unique()) == 2:
        labels = ['Инди', 'AAA']
    else:
        labels = [f'Class {i}' for i in range(len(y_clf.unique()))]

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Матрица ошибок (Confusion Matrix)', fontsize=14)
    plt.ylabel('Фактический класс', fontsize=12)
    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("✓ Матрица ошибок сохранена в 'confusion_matrix.png'")

    # 6. ROC-КРИВАЯ (только для бинарной классификации)
    if len(y_clf.unique()) == 2:
        print("\n" + "=" * 70)
        print("ПОСТРОЕНИЕ И АНАЛИЗ ROC-КРИВОЙ")
        print("=" * 70)

        # Расчет ROC-кривой
        fpr, tpr, thresholds = roc_curve(y_test_clf, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        print(f"Площадь под ROC-кривой (AUC): {roc_auc:.4f}")

        # Визуализация ROC-кривой
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC кривая (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                 label='Случайный классификатор')
        plt.fill_between(fpr, tpr, alpha=0.2, color='darkorange')

        # Находим оптимальный порог (ближайший к левому верхнему углу)
        distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
        optimal_idx = np.argmin(distances)
        optimal_threshold = thresholds[optimal_idx]

        plt.scatter(fpr[optimal_idx], tpr[optimal_idx],
                    color='red', s=100, zorder=5,
                    label=f'Оптимальный порог\n({optimal_threshold:.2f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR) / Recall', fontsize=12)
        plt.title('ROC-кривая для классификатора', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=150)
        print("✓ ROC-кривая сохранена в 'roc_curve.png'")
        print(f"Оптимальный порог: {optimal_threshold:.4f}")
    else:
        print("\n⚠ ROC-кривая строится только для бинарной классификации")
else:
    print("\n⚠ Недостаточно классов для классификации (требуется минимум 2)")
    print("  Пропускаем задачу классификации")

# 7. ВАЖНОСТЬ ПРИЗНАКОВ
print("\n" + "=" * 70)
print("ВАЖНОСТЬ ПРИЗНАКОВ")
print("=" * 70)

# Важность признаков для регрессии
feature_importance_reg = pd.DataFrame({
    'Признак': features,
    'Важность': regressor.feature_importances_
}).sort_values('Важность', ascending=False)

print("\nВажность признаков (регрессия):")
print(feature_importance_reg.to_string(index=False))

if len(y_clf.unique()) > 1:
    # Важность признаков для классификации
    feature_importance_clf = pd.DataFrame({
        'Признак': features,
        'Важность': classifier.feature_importances_
    }).sort_values('Важность', ascending=False)

    print("\nВажность признаков (классификация):")
    print(feature_importance_clf.to_string(index=False))

# Визуализация важности признаков
fig, axes = plt.subplots(1, 2 if len(y_clf.unique()) > 1 else 1, figsize=(15, 6) if len(y_clf.unique()) > 1 else (8, 6))

if len(y_clf.unique()) > 1:
    # Для классификации
    axes[0].barh(feature_importance_clf['Признак'], feature_importance_clf['Важность'])
    axes[0].set_xlabel('Важность признака', fontsize=12)
    axes[0].set_title('Важность признаков (классификация)', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='x')

    # Для регрессии
    axes[1].barh(feature_importance_reg['Признак'], feature_importance_reg['Важность'])
    axes[1].set_xlabel('Важность признака', fontsize=12)
    axes[1].set_title('Важность признаков (регрессия)', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='x')
else:
    # Только для регрессии
    axes.barh(feature_importance_reg['Признак'], feature_importance_reg['Важность'])
    axes.set_xlabel('Важность признака', fontsize=12)
    axes.set_title('Важность признаков (регрессия)', fontsize=14)
    axes.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("\n✓ График важности признаков сохранен в 'feature_importance.png'")

# 8. ИТОГОВЫЙ ОТЧЕТ
print("\n" + "=" * 70)
print("ИТОГОВЫЙ ОТЧЕТ")
print("=" * 70)

# Создание датафрейма с результатами
if len(y_clf.unique()) > 1:
    results_data = {
        'Задача': ['Регрессия', 'Классификация'],
        'Модель': ['DecisionTreeRegressor', 'DecisionTreeClassifier'],
        'Основная метрика': [f'R² = {r2:.4f}', f'Accuracy = {accuracy:.4f}'],
        'Дополнительные метрики': [
            f'MSE = {mse:.2f}, MAE = {mae:.2f}',
            f'AUC = {roc_auc:.4f}' if len(y_clf.unique()) == 2 else 'N/A'
        ]
    }
else:
    results_data = {
        'Задача': ['Регрессия'],
        'Модель': ['DecisionTreeRegressor'],
        'Основная метрика': [f'R² = {r2:.4f}'],
        'Дополнительные метрика': [f'MSE = {mse:.2f}, MAE = {mae:.2f}']
    }

results_df = pd.DataFrame(results_data)

print("\nСводка результатов:")
print(results_df.to_string(index=False))

print("\n" + "=" * 70)
print("ВЫВОДЫ")
print("=" * 70)

print(f"""
1. РЕГРЕССИЯ (прогнозирование продолжительности):
   - Качество модели: {'Хорошее' if r2 > 0.7 else 'Умеренное' if r2 > 0.5 else 'Низкое'} (R² = {r2:.4f})
   - Наиболее важные признаки: {', '.join(feature_importance_reg.head(3)['Признак'].tolist())}

2. КЛАССИФИКАЦИЯ (определение AAA-игр):""")
if len(y_clf.unique()) > 1:
    print(f"""   - Точность: {accuracy:.2%}
   - {'Высокое' if roc_auc > 0.8 else 'Умеренное' if roc_auc > 0.7 else 'Низкое'} качество разделения (AUC = {roc_auc:.4f})
   - Наиболее важные признаки: {', '.join(feature_importance_clf.head(3)['Признак'].tolist())}""")
else:
    print("   - Не выполнена (недостаточно классов)")

print("""
3. СОХРАНЕННЫЕ ФАЙЛЫ:
   - regression_tree.png - дерево решений для регрессии
   - regression_scatter.png - сравнение предсказанных и фактических значений
   - classification_tree.png - дерево решений для классификации
   - confusion_matrix.png - матрица ошибок
   - roc_curve.png - ROC-кривая
   - feature_importance.png - важность признаков

4. РЕКОМЕНДАЦИИ:
   - Экспериментировать с глубиной дерева
   - Попробовать ансамблевые методы (Random Forest)
   - Использовать кросс-валидацию
   - Создать больше признаков из имеющихся данных
""")

# Показ всех графиков
plt.show()

print("\n" + "=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА №3 УСПЕШНО ЗАВЕРШЕНА!")
print("=" * 70)
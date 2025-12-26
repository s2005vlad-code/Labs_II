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
print("="*70)
print("ЛАБОРАТОРНАЯ РАБОТА №3: ДЕРЕВЬЯ РЕШЕНИЙ")
print("="*70)

# Загрузка данных
data = pd.read_csv('epic_games_data.csv')

# Просмотр первых строк
print("\nПервые 5 строк данных:")
print(data.head())
print(f"\nРазмер датасета: {data.shape}")
print(f"\nСтолбцы: {list(data.columns)}")
print(f"\nТипы данных:\n{data.dtypes}")

# 2. ПРЕДОБРАБОТКА ДАННЫХ
print("\n" + "="*70)
print("ПРЕДОБРАБОТКА ДАННЫХ")
print("="*70)

# Преобразование дат
data['Start'] = pd.to_datetime(data['Start'], format='%d-%m-%Y', errors='coerce')
data['End'] = pd.to_datetime(data['End'], format='%d-%m-%Y', errors='coerce')

# Создание целевых переменных
# Для регрессии: продолжительность в днях
data['Duration_days'] = (data['End'] - data['Start']).dt.days

# Для классификации: является ли игра AAA (1) или инди (0)
# Определим по ключевым словам в жанре
aaa_keywords = ['AAA', 'AAA bundle', 'big AAA', 'major AAA', 'open-world', 'AAA fantasy', 'AAA story']
data['Is_AAA'] = data['Genre'].apply(
    lambda x: 1 if any(keyword.lower() in str(x).lower() for keyword in aaa_keywords) else 0
)

# Создание дополнительных признаков
data['Year'] = data['Start'].dt.year
data['Month'] = data['Start'].dt.month
data['Day'] = data['Start'].dt.day
data['Weekday'] = data['Start'].dt.weekday

# Кодирование жанра
le_genre = LabelEncoder()
data['Genre_encoded'] = le_genre.fit_transform(data['Genre'].fillna('Unknown'))

# Длина названия игры как признак
data['Name_length'] = data['Game'].apply(lambda x: len(str(x)))

# Подготовка признаков
features = ['serial no.', 'Year', 'Month', 'Day', 'Weekday', 'Genre_encoded', 'Name_length']
X = data[features]

# Целевые переменные
y_reg = data['Duration_days']  # Для регрессии
y_clf = data['Is_AAA']         # Для классификации

print(f"\nРаспределение классов для классификации:")
print(y_clf.value_counts())
print(f"\nПроцент AAA-игр: {y_clf.mean()*100:.2f}%")

print(f"\nСтатистика по целевой переменной регрессии (Duration_days):")
print(y_reg.describe())

# Удаление пропущенных значений
data_clean = data.dropna(subset=['Duration_days', 'Is_AAA'])
X = X.loc[data_clean.index]
y_reg = y_reg.loc[data_clean.index]
y_clf = y_clf.loc[data_clean.index]

# 3. РАЗДЕЛЕНИЕ ДАННЫХ
print("\n" + "="*70)
print("РАЗДЕЛЕНИЕ ДАННЫХ НА ВЫБОРКИ")
print("="*70)

# Для регрессии
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.3, random_state=42
)

# Для классификации
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.3, random_state=42, stratify=y_clf
)

print(f"Размеры выборок для регрессии:")
print(f"  Обучающая: {X_train_reg.shape}, Тестовая: {X_test_reg.shape}")
print(f"Размеры выборок для классификации:")
print(f"  Обучающая: {X_train_clf.shape}, Тестовая: {X_test_clf.shape}")

# 4. ЗАДАЧА РЕГРЕССИИ
print("\n" + "="*70)
print("РЕШЕНИЕ ЗАДАЧИ РЕГРЕССИИ ДЕРЕВОМ РЕШЕНИЙ")
print("="*70)

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
plt.figure(figsize=(25, 12))
plot_tree(
    regressor,
    feature_names=features,
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=3
)
plt.title("Дерево решений для задачи регрессии (глубина 3)", fontsize=16)
plt.tight_layout()
plt.savefig('regression_tree.png', dpi=150, bbox_inches='tight')
print("\nВизуализация дерева сохранена в 'regression_tree.png'")

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
print("График сравнения сохранен в 'regression_scatter.png'")

# 5. ЗАДАЧА КЛАССИФИКАЦИИ
print("\n" + "="*70)
print("РЕШЕНИЕ ЗАДАЧИ КЛАССИФИКАЦИИ ДЕРЕВОМ РЕШЕНИЙ")
print("="*70)

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
class_report = classification_report(y_test_clf, y_pred_clf, target_names=['Инди', 'AAA'])

print("\nОценка модели классификации:")
print(f"  Точность (accuracy): {accuracy:.4f}")
print(f"\nМатрица ошибок:")
print(conf_matrix)
print(f"\nОтчет по классификации:")
print(class_report)

# Визуализация дерева классификации
plt.figure(figsize=(25, 12))
plot_tree(
    classifier,
    feature_names=features,
    class_names=['Инди', 'AAA'],
    filled=True,
    rounded=True,
    fontsize=10,
    proportion=True
)
plt.title("Дерево решений для задачи классификации", fontsize=16)
plt.tight_layout()
plt.savefig('classification_tree.png', dpi=150, bbox_inches='tight')
print("\nВизуализация дерева сохранена в 'classification_tree.png'")

# Визуализация матрицы ошибок
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Инди', 'AAA'],
    yticklabels=['Инди', 'AAA']
)
plt.title('Матрица ошибок (Confusion Matrix)', fontsize=14)
plt.ylabel('Фактический класс', fontsize=12)
plt.xlabel('Предсказанный класс', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("Матрица ошибок сохранена в 'confusion_matrix.png'")

# 6. ROC-КРИВАЯ
print("\n" + "="*70)
print("ПОСТРОЕНИЕ И АНАЛИЗ ROC-КРИВОЙ")
print("="*70)

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
distances = np.sqrt((1 - tpr)**2 + fpr**2)
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
print("ROC-кривая сохранена в 'roc_curve.png'")
print(f"Оптимальный порог: {optimal_threshold:.4f}")

# 7. ВАЖНОСТЬ ПРИЗНАКОВ
print("\n" + "="*70)
print("ВАЖНОСТЬ ПРИЗНАКОВ")
print("="*70)

# Важность признаков для классификации
feature_importance_clf = pd.DataFrame({
    'Признак': features,
    'Важность': classifier.feature_importances_
}).sort_values('Важность', ascending=False)

print("\nВажность признаков (классификация):")
print(feature_importance_clf.to_string(index=False))

# Важность признаков для регрессии
feature_importance_reg = pd.DataFrame({
    'Признак': features,
    'Важность': regressor.feature_importances_
}).sort_values('Важность', ascending=False)

print("\nВажность признаков (регрессия):")
print(feature_importance_reg.to_string(index=False))

# Визуализация важности признаков
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

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

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("\nГрафик важности признаков сохранен в 'feature_importance.png'")

# 8. ИТОГОВЫЙ ОТЧЕТ
print("\n" + "="*70)
print("ИТОГОВЫЙ ОТЧЕТ")
print("="*70)

# Создание датафрейма с результатами
results_df = pd.DataFrame({
    'Задача': ['Регрессия', 'Классификация'],
    'Модель': ['DecisionTreeRegressor', 'DecisionTreeClassifier'],
    'Основная метрика': [f'R² = {r2:.4f}', f'Accuracy = {accuracy:.4f}'],
    'Дополнительные метрики': [
        f'MSE = {mse:.2f}, MAE = {mae:.2f}',
        f'AUC = {roc_auc:.4f}'
    ],
    'Параметры модели': [
        f'max_depth=5, min_samples_split=10',
        f'max_depth=4, class_weight=balanced'
    ]
})

print("\nСводка результатов:")
print(results_df.to_string(index=False))

print("\n" + "="*70)
print("ВЫВОДЫ И РЕКОМЕНДАЦИИ")
print("="*70)

print("""
1. **РЕГРЕССИЯ (прогнозирование продолжительности бесплатного периода):**
   - Модель показывает умеренную точность (R² = {:.4f})
   - Наиболее важные признаки: {}
   - Рекомендации: попробовать другие алгоритмы (случайный лес, градиентный бустинг)

2. **КЛАССИФИКАЦИЯ (определение AAA-игр):**
   - Точность классификации: {:.2f}%
   - AUC: {:.4f} (хорошее качество разделения классов)
   - Оптимальный порог: {:.4f}
   - Наиболее важные признаки: {}

3. **СОХРАНЕННЫЕ ФАЙЛЫ:**
   - regression_tree.png - дерево решений для регрессии
   - regression_scatter.png - сравнение предсказанных и фактических значений
   - classification_tree.png - дерево решений для классификации
   - confusion_matrix.png - матрица ошибок
   - roc_curve.png - ROC-кривая с оптимальным порогом
   - feature_importance.png - важность признаков для обеих задач

4. **РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:**
   - Создать больше признаков (например, сезонность, популярность жанра)
   - Использовать кросс-валидацию для настройки гиперпараметров
   - Попробовать ансамблевые методы (Random Forest, Gradient Boosting)
   - Балансировать классы (SMOTE, oversampling)
""".format(r2,
          ', '.join(feature_importance_reg.head(3)['Признак'].tolist()),
          accuracy * 100,
          roc_auc,
          optimal_threshold,
          ', '.join(feature_importance_clf.head(3)['Признак'].tolist())))

# Показ всех графиков
plt.show()

print("\n" + "="*70)
print("ЛАБОРАТОРНАЯ РАБОТА №3 УСПЕШНО ЗАВЕРШЕНА!")
print("="*70)
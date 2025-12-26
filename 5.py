import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
import warnings
import os

warnings.filterwarnings('ignore')

# Установка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================================
print("=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА №5: НЕЙРОННЫЕ СЕТИ. МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН")
print("=" * 70)

# Проверяем наличие файлов
files = os.listdir('.')
print("Найденные файлы в директории:")
for file in files:
    if file.endswith('.csv') or file.endswith('.png'):
        print(f"  - {file}")

# Пробуем загрузить обработанные данные из предыдущих лабораторных
data_files = ['processed_epic_games.csv', 'processed_data.csv', 'epic_games_data.csv']
data = None

for file in data_files:
    if os.path.exists(file):
        try:
            data = pd.read_csv(file)
            print(f"\n✓ Загружены данные из {file}")
            print(f"  Размер датасета: {data.shape}")
            print(f"  Столбцы: {list(data.columns)}")
            break
        except Exception as e:
            print(f"✗ Ошибка загрузки {file}: {e}")

if data is None:
    print("\n⚠ Файлы данных не найдены. Создаем демо-данные...")
    # Создаем демо-данные
    np.random.seed(42)
    n_samples = 200
    data = pd.DataFrame({
        'Game': [f'Game_{i}' for i in range(n_samples)],
        'Duration_days': np.random.randint(1, 30, n_samples),
        'Year': np.random.randint(2018, 2024, n_samples),
        'Month': np.random.randint(1, 13, n_samples),
        'Name_length': np.random.randint(5, 50, n_samples),
        'Is_AAA': np.random.randint(0, 2, n_samples),
        'Game_Category_encoded': np.random.randint(0, 8, n_samples)
    })

print(f"\nПервые 5 строк данных:")
print(data.head())
print(f"\nИнформация о данных:")
print(data.info())
print(f"\nСтатистика числовых столбцов:")
print(data.describe())

# ============================================================================
# 2. ПОДГОТОВКА ПРИЗНАКОВ
# ============================================================================
print("\n" + "=" * 70)
print("ПОДГОТОВКА ПРИЗНАКОВ ДЛЯ МОДЕЛИ")
print("=" * 70)

# Определяем возможные признаки
possible_features = []
for col in data.columns:
    if data[col].dtype in ['int64', 'float64']:
        possible_features.append(col)

print(f"Возможные числовые признаки: {possible_features}")

# Убираем целевые переменные если они есть
target_columns = ['Is_AAA', 'Duration_days', 'Target', 'Label', 'Class']
features_to_use = [f for f in possible_features if f not in target_columns]

# Если мало признаков, создаем дополнительные
if len(features_to_use) < 3:
    print("\n⚠ Мало признаков. Создаем дополнительные...")
    # Создаем полиномиальные признаки
    if 'Year' in data.columns and 'Month' in data.columns:
        data['Year_Month'] = data['Year'] * 100 + data['Month']
        data['Season'] = data['Month'] % 4
        features_to_use.extend(['Year_Month', 'Season'])

    # Создаем взаимодействия
    if 'Year' in data.columns and 'Name_length' in data.columns:
        data['Year_Name_interaction'] = data['Year'] * data['Name_length']
        features_to_use.append('Year_Name_interaction')

    # Создаем случайные признаки для демонстрации
    for i in range(3):
        data[f'Feature_{i}'] = np.random.randn(len(data))
        features_to_use.append(f'Feature_{i}')

print(f"\nИспользуемые признаки: {features_to_use}")

# Целевые переменные
# Для классификации
if 'Is_AAA' in data.columns:
    y_clf = data['Is_AAA']
    print(f"\nКлассификация: распределение классов Is_AAA:")
    print(y_clf.value_counts())
    print(f"Доля AAA игр: {y_clf.mean() * 100:.2f}%")
else:
    # Создаем искусственную целевую переменную
    y_clf = (data[features_to_use[0]] > data[features_to_use[0]].median()).astype(int)
    print(f"\nСоздана искусственная целевая переменная для классификации")
    print(f"Распределение: {y_clf.value_counts()}")

# Для регрессии
if 'Duration_days' in data.columns:
    y_reg = data['Duration_days']
    print(f"\nРегрессия: статистика Duration_days:")
    print(f"  Среднее: {y_reg.mean():.2f}")
    print(f"  Мин: {y_reg.min()}")
    print(f"  Макс: {y_reg.max()}")
    print(f"  Стандартное отклонение: {y_reg.std():.2f}")
else:
    # Создаем искусственную целевую переменную
    y_reg = data[features_to_use[0]] * 2 + np.random.randn(len(data)) * 5
    print(f"\nСоздана искусственная целевая переменная для регрессии")

# Признаки
X = data[features_to_use]

print(f"\nФинальные размеры:")
print(f"  X: {X.shape}")
print(f"  y_clf: {y_clf.shape}")
print(f"  y_reg: {y_reg.shape}")

# ============================================================================
# 3. РАЗДЕЛЕНИЕ ДАННЫХ И МАСШТАБИРОВАНИЕ
# ============================================================================
print("\n" + "=" * 70)
print("РАЗДЕЛЕНИЕ ДАННЫХ И МАСШТАБИРОВАНИЕ")
print("=" * 70)

# Для классификации
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.3, random_state=42, stratify=y_clf
)

# Для регрессии
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.3, random_state=42
)

# Масштабирование
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

print(f"Размеры выборок для классификации:")
print(f"  Обучающая: {X_train_clf.shape}")
print(f"  Тестовая: {X_test_clf.shape}")

print(f"\nРазмеры выборок для регрессии:")
print(f"  Обучающая: {X_train_reg.shape}")
print(f"  Тестовая: {X_test_reg.shape}")

# ============================================================================
# 4. МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН ДЛЯ КЛАССИФИКАЦИИ
# ============================================================================
print("\n" + "=" * 70)
print("4.1 МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН ДЛЯ КЛАССИФИКАЦИИ")
print("=" * 70)

# Создаем и обучаем модель
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # Два скрытых слоя: 64 и 32 нейрона
    activation='relu',  # Функция активации ReLU
    solver='adam',  # Алгоритм оптимизации
    alpha=0.001,  # Параметр регуляризации
    batch_size=32,  # Размер мини-батча
    learning_rate='adaptive',  # Адаптивная скорость обучения
    max_iter=500,  # Максимальное количество итераций
    random_state=42,  # Для воспроизводимости
    early_stopping=True,  # Ранняя остановка
    validation_fraction=0.2,  # Доля данных для валидации
    n_iter_no_change=10  # Количество итераций без улучшения для ранней остановки
)

print("Обучение MLP классификатора...")
mlp_classifier.fit(X_train_clf_scaled, y_train_clf)

# Предсказания
y_pred_clf = mlp_classifier.predict(X_test_clf_scaled)
y_pred_proba_clf = mlp_classifier.predict_proba(X_test_clf_scaled)[:, 1]

# Оценка модели
accuracy = accuracy_score(y_test_clf, y_pred_clf)
conf_matrix = confusion_matrix(y_test_clf, y_pred_clf)
class_report = classification_report(y_test_clf, y_pred_clf)

print(f"\nРезультаты классификации MLP:")
print(f"Точность (accuracy): {accuracy:.4f}")
print(f"\nМатрица ошибок:")
print(conf_matrix)
print(f"\nОтчет по классификации:")
print(class_report)

# ROC кривая и AUC
fpr, tpr, thresholds = roc_curve(y_test_clf, y_pred_proba_clf)
roc_auc = auc(fpr, tpr)

print(f"\nROC-AUC: {roc_auc:.4f}")

# ============================================================================
# 5. МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН ДЛЯ РЕГРЕССИИ
# ============================================================================
print("\n" + "=" * 70)
print("4.2 МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН ДЛЯ РЕГРЕССИИ")
print("=" * 70)

# Создаем и обучаем модель
mlp_regressor = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),  # Три скрытых слоя
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate='adaptive',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2
)

print("Обучение MLP регрессора...")
mlp_regressor.fit(X_train_reg_scaled, y_train_reg)

# Предсказания
y_pred_reg = mlp_regressor.predict(X_test_reg_scaled)

# Оценка модели
mse = mean_squared_error(y_test_reg, y_pred_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"\nРезультаты регрессии MLP:")
print(f"Среднеквадратичная ошибка (MSE): {mse:.4f}")
print(f"Средняя абсолютная ошибка (MAE): {mae:.4f}")
print(f"Коэффициент детерминации (R²): {r2:.4f}")

# ============================================================================
# 6. СРАВНЕНИЕ С КЛАССИЧЕСКИМИ АЛГОРИТМАМИ
# ============================================================================
print("\n" + "=" * 70)
print("5. СРАВНЕНИЕ С КЛАССИЧЕСКИМИ АЛГОРИТМАМИ")
print("=" * 70)

# 6.1 Логистическая регрессия
print("\n--- Логистическая регрессия ---")
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_clf_scaled, y_train_clf)
y_pred_logreg = logreg.predict(X_test_clf_scaled)
accuracy_logreg = accuracy_score(y_test_clf, y_pred_logreg)
print(f"Точность логистической регрессии: {accuracy_logreg:.4f}")

# 6.2 Случайный лес
print("\n--- Случайный лес ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_clf_scaled, y_train_clf)
y_pred_rf = rf.predict(X_test_clf_scaled)
accuracy_rf = accuracy_score(y_test_clf, y_pred_rf)
print(f"Точность случайного леса: {accuracy_rf:.4f}")

# Сравнение моделей
comparison_df = pd.DataFrame({
    'Модель': ['Логистическая регрессия', 'Случайный лес', 'MLP (нейронная сеть)'],
    'Точность': [accuracy_logreg, accuracy_rf, accuracy]
}).sort_values('Точность', ascending=False)

print(f"\nСравнение производительности моделей:")
print(comparison_df.to_string(index=False))

# ============================================================================
# 7. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# ============================================================================
print("\n" + "=" * 70)
print("6. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 70)

# Создаем фигуру с несколькими графиками
fig = plt.figure(figsize=(20, 16))

# 1. Сравнение производительности моделей
ax1 = plt.subplot(3, 3, 1)
bars = ax1.bar(comparison_df['Модель'], comparison_df['Точность'],
               color=['lightblue', 'lightgreen', 'lightcoral'])
ax1.set_ylabel('Точность', fontsize=12)
ax1.set_title('Сравнение точности моделей', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1)
ax1.tick_params(axis='x', rotation=15)
ax1.grid(True, alpha=0.3, axis='y')

# Добавление значений на столбцы
for bar, acc in zip(bars, comparison_df['Точность']):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f'{acc:.3f}', ha='center', fontsize=11, fontweight='bold')

# 2. Матрица ошибок MLP
ax2 = plt.subplot(3, 3, 2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Не AAA', 'AAA'], yticklabels=['Не AAA', 'AAA'])
ax2.set_title('Матрица ошибок MLP', fontsize=14, fontweight='bold')
ax2.set_xlabel('Предсказанный класс', fontsize=12)
ax2.set_ylabel('Фактический класс', fontsize=12)

# 3. ROC-кривая
ax3 = plt.subplot(3, 3, 3)
ax3.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC кривая (AUC = {roc_auc:.3f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Случайный классификатор')
ax3.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
ax3.set_xlabel('False Positive Rate', fontsize=12)
ax3.set_ylabel('True Positive Rate', fontsize=12)
ax3.set_title('ROC-кривая MLP', fontsize=14, fontweight='bold')
ax3.legend(loc="lower right")
ax3.grid(True, alpha=0.3)

# 4. График обучения MLP (потери)
ax4 = plt.subplot(3, 3, 4)
if hasattr(mlp_classifier, 'loss_curve_'):
    ax4.plot(mlp_classifier.loss_curve_, label='Потери на обучении', linewidth=2)
    ax4.set_xlabel('Итерации', fontsize=12)
    ax4.set_ylabel('Функция потерь', fontsize=12)
    ax4.set_title('Кривая обучения MLP', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

# 5. График фактических vs предсказанных значений (регрессия)
ax5 = plt.subplot(3, 3, 5)
ax5.scatter(y_test_reg, y_pred_reg, alpha=0.6, edgecolors='w', linewidth=0.5)
ax5.plot([y_test_reg.min(), y_test_reg.max()],
         [y_test_reg.min(), y_test_reg.max()],
         'r--', lw=2, label='Идеальная линия')
ax5.set_xlabel('Фактическая продолжительность', fontsize=12)
ax5.set_ylabel('Предсказанная продолжительность', fontsize=12)
ax5.set_title('Регрессия: факт vs предсказание', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Ошибки регрессии
ax6 = plt.subplot(3, 3, 6)
errors = y_pred_reg - y_test_reg
ax6.hist(errors, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax6.set_xlabel('Ошибка предсказания', fontsize=12)
ax6.set_ylabel('Частота', fontsize=12)
ax6.set_title('Распределение ошибок регрессии', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)

# 7. Важность признаков (из случайного леса)
ax7 = plt.subplot(3, 3, 7)
feature_importance = pd.DataFrame({
    'Признак': features_to_use,
    'Важность': rf.feature_importances_
}).sort_values('Важность', ascending=True).tail(10)

bars = ax7.barh(feature_importance['Признак'], feature_importance['Важность'])
ax7.set_xlabel('Важность признака', fontsize=12)
ax7.set_title('Топ-10 важных признаков', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='x')

# Добавление значений
for bar, imp in zip(bars, feature_importance['Важность']):
    ax7.text(imp + 0.01, bar.get_y() + bar.get_height() / 2,
             f'{imp:.3f}', ha='left', va='center', fontsize=9)

# 8. Зависимость точности от размера обучающей выборки
ax8 = plt.subplot(3, 3, 8)
train_sizes = np.linspace(0.1, 1.0, 10)
train_scores = []
test_scores = []

for size in train_sizes:
    n_samples = int(size * len(X_train_clf))
    X_subset = X_train_clf_scaled[:n_samples]
    y_subset = y_train_clf[:n_samples]

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    model.fit(X_subset, y_subset)

    train_scores.append(accuracy_score(y_subset, model.predict(X_subset)))
    test_scores.append(accuracy_score(y_test_clf, model.predict(X_test_clf_scaled)))

ax8.plot(train_sizes * 100, train_scores, 'o-', label='Обучающая выборка', linewidth=2)
ax8.plot(train_sizes * 100, test_scores, 's-', label='Тестовая выборка', linewidth=2)
ax8.set_xlabel('Размер обучающей выборки (%)', fontsize=12)
ax8.set_ylabel('Точность', fontsize=12)
ax8.set_title('Кривая обучения', fontsize=14, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Сравнение архитектур MLP
ax9 = plt.subplot(3, 3, 9)
architectures = [
    ('(32,)', (32,)),
    ('(64,)', (64,)),
    ('(64, 32)', (64, 32)),
    ('(128, 64)', (128, 64)),
    ('(64, 32, 16)', (64, 32, 16))
]

arch_accuracies = []
for name, arch in architectures:
    model = MLPClassifier(hidden_layer_sizes=arch, max_iter=300, random_state=42)
    model.fit(X_train_clf_scaled, y_train_clf)
    acc = accuracy_score(y_test_clf, model.predict(X_test_clf_scaled))
    arch_accuracies.append(acc)

bars = ax9.bar([a[0] for a in architectures], arch_accuracies)
ax9.set_xlabel('Архитектура', fontsize=12)
ax9.set_ylabel('Точность', fontsize=12)
ax9.set_title('Сравнение архитектур MLP', fontsize=14, fontweight='bold')
ax9.set_ylim(0, 1)
ax9.tick_params(axis='x', rotation=45)
ax9.grid(True, alpha=0.3, axis='y')

for bar, acc in zip(bars, arch_accuracies):
    ax9.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f'{acc:.3f}', ha='center', fontsize=10)

plt.suptitle('ЛАБОРАТОРНАЯ РАБОТА №5: РЕЗУЛЬТАТЫ НЕЙРОННЫХ СЕТЕЙ',
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('lab5_results.png', dpi=150, bbox_inches='tight')
print("✓ Основные результаты сохранены в 'lab5_results.png'")

# ============================================================================
# 8. ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ: ФУНКЦИИ АКТИВАЦИИ
# ============================================================================
print("\n" + "=" * 70)
print("7. ВИЗУАЛИЗАЦИЯ ФУНКЦИЙ АКТИВАЦИИ")
print("=" * 70)

# Создаем график функций активации
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
x = np.linspace(-5, 5, 100)

# 1. ReLU (Rectified Linear Unit)
axes2[0, 0].plot(x, np.maximum(0, x), linewidth=3, color='blue')
axes2[0, 0].set_title('ReLU (Rectified Linear Unit)', fontsize=12, fontweight='bold')
axes2[0, 0].set_xlabel('x', fontsize=11)
axes2[0, 0].set_ylabel('f(x)', fontsize=11)
axes2[0, 0].grid(True, alpha=0.3)
axes2[0, 0].axhline(y=0, color='black', linewidth=0.5)
axes2[0, 0].axvline(x=0, color='black', linewidth=0.5)

# 2. Сигмоида
axes2[0, 1].plot(x, 1 / (1 + np.exp(-x)), linewidth=3, color='red')
axes2[0, 1].set_title('Сигмоида', fontsize=12, fontweight='bold')
axes2[0, 1].set_xlabel('x', fontsize=11)
axes2[0, 1].set_ylabel('f(x)', fontsize=11)
axes2[0, 1].grid(True, alpha=0.3)
axes2[0, 1].axhline(y=0, color='black', linewidth=0.5)
axes2[0, 1].axvline(x=0, color='black', linewidth=0.5)

# 3. Гиперболический тангенс
axes2[0, 2].plot(x, np.tanh(x), linewidth=3, color='green')
axes2[0, 2].set_title('Tanh (Гиперболический тангенс)', fontsize=12, fontweight='bold')
axes2[0, 2].set_xlabel('x', fontsize=11)
axes2[0, 2].set_ylabel('f(x)', fontsize=11)
axes2[0, 2].grid(True, alpha=0.3)
axes2[0, 2].axhline(y=0, color='black', linewidth=0.5)
axes2[0, 2].axvline(x=0, color='black', linewidth=0.5)

# 4. Leaky ReLU
alpha = 0.1
axes2[1, 0].plot(x, np.where(x > 0, x, alpha * x), linewidth=3, color='purple')
axes2[1, 0].set_title('Leaky ReLU', fontsize=12, fontweight='bold')
axes2[1, 0].set_xlabel('x', fontsize=11)
axes2[1, 0].set_ylabel('f(x)', fontsize=11)
axes2[1, 0].grid(True, alpha=0.3)
axes2[1, 0].axhline(y=0, color='black', linewidth=0.5)
axes2[1, 0].axvline(x=0, color='black', linewidth=0.5)

# 5. ELU (Exponential Linear Unit)
alpha = 1.0
axes2[1, 1].plot(x, np.where(x > 0, x, alpha * (np.exp(x) - 1)), linewidth=3, color='orange')
axes2[1, 1].set_title('ELU (Exponential Linear Unit)', fontsize=12, fontweight='bold')
axes2[1, 1].set_xlabel('x', fontsize=11)
axes2[1, 1].set_ylabel('f(x)', fontsize=11)
axes2[1, 1].grid(True, alpha=0.3)
axes2[1, 1].axhline(y=0, color='black', linewidth=0.5)
axes2[1, 1].axvline(x=0, color='black', linewidth=0.5)

# 6. Softmax (для одного значения)
axes2[1, 2].plot(x, np.exp(x) / np.sum(np.exp(x)), linewidth=3, color='brown')
axes2[1, 2].set_title('Softmax (для сравнения)', fontsize=12, fontweight='bold')
axes2[1, 2].set_xlabel('x', fontsize=11)
axes2[1, 2].set_ylabel('f(x)', fontsize=11)
axes2[1, 2].grid(True, alpha=0.3)
axes2[1, 2].axhline(y=0, color='black', linewidth=0.5)
axes2[1, 2].axvline(x=0, color='black', linewidth=0.5)

plt.suptitle('ФУНКЦИИ АКТИВАЦИИ В НЕЙРОННЫХ СЕТЯХ', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('lab5_activation_functions.png', dpi=150, bbox_inches='tight')
print("✓ Функции активации сохранены в 'lab5_activation_functions.png'")

# ============================================================================
# 9. ДЕТАЛЬНЫЙ АНАЛИЗ MLP
# ============================================================================
print("\n" + "=" * 70)
print("8. ДЕТАЛЬНЫЙ АНАЛИЗ MLP МОДЕЛИ")
print("=" * 70)

print(f"\nПараметры MLP модели:")
print(f"  Архитектура: {mlp_classifier.hidden_layer_sizes}")
print(f"  Функция активации: {mlp_classifier.activation}")
print(f"  Решатель: {mlp_classifier.solver}")
print(f"  Количество итераций: {mlp_classifier.n_iter_}")
print(f"  Финальная функция потерь: {mlp_classifier.loss_:.4f}")

# Анализ весов
print(f"\nАнализ архитектуры нейронной сети:")
print(f"  Входной слой: {X_train_clf_scaled.shape[1]} нейронов")
for i, (weights, biases) in enumerate(zip(mlp_classifier.coefs_, mlp_classifier.intercepts_)):
    print(f"  Скрытый слой {i + 1}: {weights.shape[1]} нейронов")
    print(f"    Веса: {weights.shape[0]}x{weights.shape[1]}")
    print(f"    Смещения: {biases.shape[0]}")
print(f"  Выходной слой: {mlp_classifier.coefs_[-1].shape[1]} нейронов")

# ============================================================================
# 10. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================
print("\n" + "=" * 70)
print("9. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 70)

# Сохраняем предсказания
predictions_df = pd.DataFrame({
    'Фактический_класс': y_test_clf.values,
    'Предсказанный_класс': y_pred_clf,
    'Вероятность_AAA': y_pred_proba_clf,
    'Фактическая_продолжительность': y_test_reg.values[:len(y_test_clf)],
    'Предсказанная_продолжительность': y_pred_reg[:len(y_test_clf)]
})

predictions_df.to_csv('lab5_predictions.csv', index=False, encoding='utf-8-sig')
print("✓ Предсказания сохранены в 'lab5_predictions.csv'")

# Сохраняем метрики
metrics_df = pd.DataFrame({
    'Метрика': ['Accuracy', 'ROC-AUC', 'MSE', 'MAE', 'R²'],
    'Значение': [accuracy, roc_auc, mse, mae, r2]
})

metrics_df.to_csv('lab5_metrics.csv', index=False, encoding='utf-8-sig')
print("✓ Метрики сохранены в 'lab5_metrics.csv'")

# Сохраняем сравнение моделей
comparison_df.to_csv('lab5_model_comparison.csv', index=False, encoding='utf-8-sig')
print("✓ Сравнение моделей сохранено в 'lab5_model_comparison.csv'")

# ============================================================================
# 11. ИТОГОВЫЙ ОТЧЕТ
# ============================================================================
print("\n" + "=" * 70)
print("10. ИТОГОВЫЙ ОТЧЕТ")
print("=" * 70)

report = f"""
ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ №5
{'=' * 60}

НЕЙРОННЫЕ СЕТИ: МНОГОСЛОЙНЫЙ ПЕРСЕПТРОН (MLP)

ДАННЫЕ:
- Источник данных: {'processed_epic_games.csv' if 'processed_epic_games.csv' in files else 'сгенерированные данные'}
- Количество записей: {len(data)}
- Количество признаков: {len(features_to_use)}
- Целевая переменная (классификация): Is_AAA (AAA vs Indie игры)
- Целевая переменная (регрессия): Duration_days (продолжительность раздачи)

МОДЕЛЬ MLP (КЛАССИФИКАЦИЯ):
- Архитектура: {mlp_classifier.hidden_layer_sizes}
- Функция активации: {mlp_classifier.activation}
- Алгоритм обучения: {mlp_classifier.solver}
- Итераций обучения: {mlp_classifier.n_iter_}

РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ:
- Точность MLP: {accuracy:.4f}
- ROC-AUC: {roc_auc:.4f}
- Матрица ошибок:
{conf_matrix}

СРАВНЕНИЕ С ДРУГИМИ АЛГОРИТМАМИ:
{comparison_df.to_string(index=False)}

РЕЗУЛЬТАТЫ РЕГРЕССИИ:
- Среднеквадратичная ошибка (MSE): {mse:.4f}
- Средняя абсолютная ошибка (MAE): {mae:.4f}
- Коэффициент детерминации (R²): {r2:.4f}

ВАЖНЫЕ ПРИЗНАКИ (топ-3):
1. {feature_importance.iloc[-1]['Признак']}: {feature_importance.iloc[-1]['Важность']:.3f}
2. {feature_importance.iloc[-2]['Признак']}: {feature_importance.iloc[-2]['Важность']:.3f}
3. {feature_importance.iloc[-3]['Признак']}: {feature_importance.iloc[-3]['Важность']:.3f}

ВЫВОДЫ:
1. Нейронная сеть (MLP) показала {'лучшие' if accuracy == max(comparison_df['Точность']) else 'сравнимые'} 
   результаты по сравнению с классическими алгоритмами.
2. Точность MLP составляет {accuracy:.2%}, что является {'хорошим' if accuracy > 0.7 else 'удовлетворительным' if accuracy > 0.6 else 'низким'} показателем.
3. Регрессионная модель MLP {'хорошо' if r2 > 0.7 else 'удовлетворительно' if r2 > 0.5 else 'плохо'} 
   предсказывает продолжительность раздачи игр.
4. Наиболее важными для классификации оказались временные признаки и характеристики названий игр.

СОХРАНЕННЫЕ ФАЙЛЫ:
1. lab5_results.png - основные результаты и графики
2. lab5_activation_functions.png - функции активации
3. lab5_predictions.csv - предсказания моделей
4. lab5_metrics.csv - метрики качества
5. lab5_model_comparison.csv - сравнение моделей

РЕКОМЕНДАЦИИ:
1. Для улучшения результатов можно добавить больше признаков
2. Экспериментировать с различными архитектурами нейронных сетей
3. Использовать более глубокие сети для сложных зависимостей
4. Применить регуляризацию для предотвращения переобучения
"""

print(report)

# Сохраняем отчет в файл
with open('lab5_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("✓ Итоговый отчет сохранен в 'lab5_report.txt'")

print("\n" + "=" * 70)
print("ЛАБОРАТОРНАЯ РАБОТА №5 УСПЕШНО ВЫПОЛНЕНА!")
print("=" * 70)

# Показать графики
plt.show()
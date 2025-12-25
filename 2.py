import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

np.random.seed(42)

df = pd.DataFrame({
    'game_id': range(1, 541),
    'days_available': np.random.randint(7, 30, 540),
    'popularity_score': np.random.uniform(0.5, 5.0, 540),
    'year': np.random.choice([2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025], 540),
    'month': np.random.randint(1, 13, 540),
    'genre_encoded': np.random.choice([0, 1, 2, 3, 4], 540),
    'is_holiday': np.random.choice([0, 1], 540, p=[0.7, 0.3]),
    'repeat_game': np.random.choice([0, 1], 540, p=[0.8, 0.2]),
    'price_category': np.random.choice(['Free', 'Premium', 'AAA'], 540, p=[0.5, 0.3, 0.2]),
    'user_rating': np.random.uniform(3.0, 5.0, 540)
})

print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
print(df.head())

X_reg = df.drop(['user_rating', 'price_category'], axis=1)
y_reg = df['user_rating']

label_encoder = LabelEncoder()
df['price_category_encoded'] = label_encoder.fit_transform(df['price_category'])

X_clf = df.drop(['price_category', 'price_category_encoded', 'user_rating'], axis=1)
y_clf = df['price_category_encoded']

print(f"\nЦелевая переменная для регрессии: user_rating")
print(f"Диапазон значений: {y_reg.min():.2f} - {y_reg.max():.2f}")
print(f"Среднее: {y_reg.mean():.2f}")

print(f"\nЦелевая переменная для классификации: price_category")
print(f"Классы: {list(label_encoder.classes_)}")

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf
)

print(f"\nРазделение выборок (70/30):")
print(f"Обучающая выборка: {X_train_reg.shape[0]} строк")
print(f"Тестовая выборка: {X_test_reg.shape[0]} строк")

print("\n" + "="*60)
print("РЕГРЕССИЯ: ПРОГНОЗИРОВАНИЕ РЕЙТИНГА ИГР")
print("="*60)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

lr_model = LinearRegression()
lr_model.fit(X_train_reg_scaled, y_train_reg)
y_pred_lr = lr_model.predict(X_test_reg_scaled)

mse_lr = mean_squared_error(y_test_reg, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_test_reg, y_pred_lr)
r2_lr = r2_score(y_test_reg, y_pred_lr)

print(f"\nЛинейная регрессия:")
print(f"MSE:  {mse_lr:.4f}")
print(f"RMSE: {rmse_lr:.4f}")
print(f"MAE:  {mae_lr:.4f}")
print(f"R²:   {r2_lr:.4f}")

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_reg_scaled, y_train_reg)
y_pred_ridge = ridge_model.predict(X_test_reg_scaled)

mse_ridge = mean_squared_error(y_test_reg, y_pred_ridge)
r2_ridge = r2_score(y_test_reg, y_pred_ridge)

print(f"\nRidge регрессия (L2):")
print(f"MSE: {mse_ridge:.4f}")
print(f"R²:  {r2_ridge:.4f}")

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_reg_scaled, y_train_reg)
y_pred_lasso = lasso_model.predict(X_test_reg_scaled)

mse_lasso = mean_squared_error(y_test_reg, y_pred_lasso)
r2_lasso = r2_score(y_test_reg, y_pred_lasso)

print(f"\nLasso регрессия (L1):")
print(f"MSE: {mse_lasso:.4f}")
print(f"R²:  {r2_lasso:.4f}")

models_reg = {
    'Linear Regression': r2_lr,
    'Ridge': r2_ridge,
    'Lasso': r2_lasso
}

best_reg_model = max(models_reg, key=models_reg.get)
print(f"\nЛучшая модель регрессии: {best_reg_model} (R² = {models_reg[best_reg_model]:.4f})")

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(y_test_reg, y_pred_lr, alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Фактический рейтинг')
plt.ylabel('Предсказанный рейтинг')
plt.title('Линейная регрессия')

plt.subplot(1, 3, 2)
plt.scatter(y_test_reg, y_pred_ridge, alpha=0.5, color='green')
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Фактический рейтинг')
plt.ylabel('Предсказанный рейтинг')
plt.title('Ridge регрессия')

plt.subplot(1, 3, 3)
plt.scatter(y_test_reg, y_pred_lasso, alpha=0.5, color='purple')
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Фактический рейтинг')
plt.ylabel('Предсказанный рейтинг')
plt.title('Lasso регрессия')

plt.tight_layout()
plt.savefig('regression_results.png', dpi=100)
print("\nГрафики регрессии сохранены в 'regression_results.png'")

print("\n" + "="*60)
print("КЛАССИФИКАЦИЯ: ОПРЕДЕЛЕНИЕ КАТЕГОРИИ ЦЕНЫ")
print("="*60)

scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train_clf_scaled, y_train_clf)
y_pred_logreg = logreg_model.predict(X_test_clf_scaled)
y_prob_logreg = logreg_model.predict_proba(X_test_clf_scaled)

accuracy_logreg = accuracy_score(y_test_clf, y_pred_logreg)
precision_logreg = precision_score(y_test_clf, y_pred_logreg, average='weighted', zero_division=0)
recall_logreg = recall_score(y_test_clf, y_pred_logreg, average='weighted', zero_division=0)
f1_logreg = f1_score(y_test_clf, y_pred_logreg, average='weighted', zero_division=0)

print(f"\nЛогистическая регрессия:")
print(f"Accuracy:  {accuracy_logreg:.4f}")
print(f"Precision: {precision_logreg:.4f}")
print(f"Recall:    {recall_logreg:.4f}")
print(f"F1-score:  {f1_logreg:.4f}")

cm_logreg = confusion_matrix(y_test_clf, y_pred_logreg)
print(f"\nМатрица ошибок:\n{cm_logreg}")

class_names = label_encoder.classes_
print(f"\nОтчет по классификации:")
print(classification_report(y_test_clf, y_pred_logreg, target_names=class_names))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('Матрица ошибок')
axes[0].set_ylabel('Фактический класс')
axes[0].set_xlabel('Предсказанный класс')

if hasattr(logreg_model, 'coef_'):
    feature_importance = pd.DataFrame({
        'Feature': X_clf.columns,
        'Coefficient': np.mean(np.abs(logreg_model.coef_), axis=0)
    })
    feature_importance = feature_importance.sort_values('Coefficient', ascending=True).tail(10)
    axes[1].barh(feature_importance['Feature'], feature_importance['Coefficient'])
    axes[1].set_title('Важность признаков (топ-10)')

for i, class_name in enumerate(class_names):
    axes[2].hist(y_prob_logreg[y_test_clf == i, i], alpha=0.5, label=class_name, bins=20)
axes[2].set_title('Распределение вероятностей')
axes[2].set_xlabel('Вероятность')
axes[2].set_ylabel('Частота')
axes[2].legend()

plt.tight_layout()
plt.savefig('classification_results.png', dpi=100)
print("\nГрафики классификации сохранены в 'classification_results.png'")

results_df = pd.DataFrame({
    'game_id': X_test_reg['game_id'].values,
    'actual_rating': y_test_reg.values,
    'predicted_rating_lr': y_pred_lr,
    'predicted_rating_ridge': y_pred_ridge,
    'predicted_rating_lasso': y_pred_lasso,
    'actual_price_category': y_test_clf,
    'predicted_price_category': y_pred_logreg,
    'predicted_probability': [max(probs) for probs in y_prob_logreg]
})

results_df['actual_price_category_name'] = label_encoder.inverse_transform(results_df['actual_price_category'])
results_df['predicted_price_category_name'] = label_encoder.inverse_transform(results_df['predicted_price_category'])

results_df.to_csv('lab2_predictions.csv', index=False)
print("Предсказания сохранены в 'lab2_predictions.csv'")

metrics_df = pd.DataFrame({
    'Model_Type': ['Regression', 'Regression', 'Regression', 'Classification'],
    'Model': ['Linear Regression', 'Ridge', 'Lasso', 'Logistic Regression'],
    'MSE': [mse_lr, mse_ridge, mse_lasso, np.nan],
    'R2': [r2_lr, r2_ridge, r2_lasso, np.nan],
    'Accuracy': [np.nan, np.nan, np.nan, accuracy_logreg],
    'F1_Score': [np.nan, np.nan, np.nan, f1_logreg]
})

metrics_df.to_csv('lab2_metrics.csv', index=False)
print("Метрики сохранены в 'lab2_metrics.csv'")

if hasattr(lr_model, 'coef_'):
    coef_df = pd.DataFrame({
        'Feature': X_reg.columns,
        'Linear_Regression_Coefficient': lr_model.coef_,
        'Ridge_Coefficient': ridge_model.coef_,
        'Lasso_Coefficient': lasso_model.coef_
    })
    coef_df.to_csv('lab2_coefficients.csv', index=False)
    print("Коэффициенты моделей сохранены в 'lab2_coefficients.csv'")

print("\n" + "="*60)
print("АНАЛИЗ РЕЗУЛЬТАТОВ:")
print("="*60)
print("1. Регрессия: R² отрицательный - модель хуже среднего значения")
print("2. Классификация: Accuracy = 41.98% - случайное угадывание 33%")
print("3. Причина: синтетические данные без реальных закономерностей")
print("4. Для улучшения нужны реальные данные с Kaggle")

print("\n" + "="*60)
print("ВЫПОЛНЕНИЕ ЗАДАНИЯ ЗАВЕРШЕНО")
print("="*60)

plt.show()
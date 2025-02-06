import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv('rohan\Major\data.csv')
    
    required_columns = ['TV', 'Radio', 'Newspaper', 'Sales']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Missing required columns in the dataset")
    
    print("First few rows of the dataset:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nCheck for missing values:")
    print(df.isnull().sum())
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Advertising Data')
    plt.tight_layout()
    plt.show()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.scatterplot(data=df, x='TV', y='Sales', ax=axes[0])
    sns.scatterplot(data=df, x='Radio', y='Sales', ax=axes[1])
    sns.scatterplot(data=df, x='Newspaper', y='Sales', ax=axes[2])
    
    axes[0].set_title('TV vs Sales')
    axes[1].set_title('Radio vs Sales')
    axes[2].set_title('Newspaper vs Sales')
    plt.tight_layout()
    plt.show()
    
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\nModel Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")
    
    print("\nModel Equation:")
    equation = f"Sales = {model.intercept_:.4f}"
    for feature, coef in zip(X.columns, model.coef_):
        equation += f" + ({coef:.4f} × {feature})"
    print(equation)
    
    y_pred = model.predict(X_test)
    print("\nModel Evaluation Metrics:")
    print(f"Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred):.4f}")
    print(f"Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred):.4f}")
    print(f"Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R-squared Score: {metrics.r2_score(y_test, y_pred):.4f}")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs Predicted Sales')
    plt.tight_layout()
    plt.show()
    test_sizes = [0.1, 0.2, 0.3, 0.4]
    rmse_scores = []
    r2_scores = []
    
    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2 = metrics.r2_score(y_test, y_pred)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(test_sizes, rmse_scores, marker='o')
    ax1.set_xlabel('Test Size')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE vs Test Size')
    ax1.grid(True)
    
    ax2.plot(test_sizes, r2_scores, marker='o', color='green')
    ax2.set_xlabel('Test Size')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score vs Test Size')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\nScores for different test sizes:")
    for test_size, rmse, r2 in zip(test_sizes, rmse_scores, r2_scores):
        print(f"Test size {test_size}: RMSE = {rmse:.4f}, R² = {r2:.4f}")

except FileNotFoundError:
    print("Error: The advertising.csv file was not found. Please ensure it's in the same directory as this script.")
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
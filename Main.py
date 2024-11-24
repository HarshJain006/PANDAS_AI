import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Function for uploading the file
def upload_file():
    try:
        file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        if file:
            if file.name.endswith("csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            st.success("Data loaded successfully!")
            return df
        return None
    except Exception as e:
        st.error(f"Error uploading file: {e}")
        return None

# Step 1: Data Understanding
def data_understanding(df):
    try:
        st.write("### 1. Data Understanding")
        st.write("**First 10 rows of the dataset:**")
        st.write(df.head(10))
    except Exception as e:
        st.error(f"Error in Data Understanding: {e}")

# Show full data button
def show_full_data(df):
    try:
        if st.button('Show Full Data'):
            st.write(df)
    except Exception as e:
        st.error(f"Error displaying full data: {e}")

# Step 2: Data Cleaning
def data_cleaning(df):
    try:
        st.write("### 2. Data Cleaning")
        
        st.write("**Missing values in each column:**")
        st.write(df.isnull().sum())
        
        st.write("**Drop rows with missing values (if selected):**")
        if st.button('Drop Missing Rows'):
            df_clean = df.dropna()
            st.write("Cleaned DataFrame:")
            st.write(df_clean)
        
        st.write("**Fill missing values (if selected):**")
        fill_option = st.selectbox('Choose fill method:', ['None', 'Mean', 'Median', 'Mode'])
        if fill_option != 'None':
            if fill_option == 'Mean':
                df = df.fillna(df.mean())
            elif fill_option == 'Median':
                df = df.fillna(df.median())
            elif fill_option == 'Mode':
                df = df.fillna(df.mode().iloc[0])
            st.write("Filled Missing Data:")
            st.write(df)
    except Exception as e:
        st.error(f"Error in Data Cleaning: {e}")

# Step 3: Univariate Analysis
def univariate_analysis(df):
    try:
        st.write("### 3. Univariate Analysis")
        column = st.selectbox("Select column for Univariate Analysis", df.columns)
        plot_type = st.selectbox("Plot Type", ['line', 'histogram', 'boxplot', 'pie', 'area', 'violin'])
        
        st.write(f"**Distribution of the selected column: {column}:**")
        if plot_type == 'line':
            df[column].plot(kind='line')
            plt.title(f"Line Plot of {column}")
        elif plot_type == 'histogram':
            df[column].plot(kind='hist', bins=30)
            plt.title(f"Histogram of {column}")
        elif plot_type == 'boxplot':
            df[column].plot(kind='box')
            plt.title(f"Boxplot of {column}")
        elif plot_type == 'pie':
            df[column].value_counts().plot(kind='pie', autopct='%1.1f%%')
            plt.title(f"Pie Chart of {column}")
        elif plot_type == 'area':
            df[column].plot(kind='area')
            plt.title(f"Area Plot of {column}")
        elif plot_type == 'violin':
            sns.violinplot(x=df[column])
            plt.title(f"Violin Plot of {column}")
        
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error in Univariate Analysis: {e}")

# Step 4: Multivariate Analysis
def multivariate_analysis(df):
    try:
        st.write("### 4. Multivariate Analysis")
        st.write("**Correlation between numerical columns (Heatmap):**")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if numeric_df.shape[1] > 0:
            correlation_matrix = numeric_df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
            st.pyplot(plt)
        else:
            st.warning("No numerical columns found for correlation analysis.")
    except Exception as e:
        st.error(f"Error in Multivariate Analysis: {e}")

# Step 5: Feature Engineering
def feature_engineering(df):
    try:
        st.write("### 5. Feature Engineering")
        st.write("**Create New Features (Example: Column Mean):**")
        column_to_avg = st.selectbox("Select column to calculate mean", df.columns)
        df[f'{column_to_avg}_mean'] = df[column_to_avg].mean()
        st.write("New Feature Created:", f"{column_to_avg}_mean")
        st.write(df.head())
    except Exception as e:
        st.error(f"Error in Feature Engineering: {e}")

# Step 6: Anomaly Detection
def anomaly_detection(df):
    try:
        st.write("### 6. Anomaly Detection")
        st.write("**Detect outliers using Z-Score:**")
        z_scores = np.abs((df.select_dtypes(include=['float64', 'int64']) - df.mean()) / df.std())
        outliers = (z_scores > 3).sum()
        st.write(f"Number of outliers detected in columns: {outliers}")
    except Exception as e:
        st.error(f"Error in Anomaly Detection: {e}")

# Step 7: Data Visualization
def data_visualization(df):
    try:
        st.write("### 7. Data Visualization")
        st.write("**Visualize correlations between columns (Correlation Heatmap):**")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if numeric_df.shape[1] > 0:
            correlation_matrix = numeric_df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
            st.pyplot(plt)
        else:
            st.warning("No numerical columns found for correlation analysis.")
    except Exception as e:
        st.error(f"Error in Data Visualization: {e}")

# Step 8: Identify Patterns and Trends
def identify_patterns_and_trends(df):
    try:
        st.write("### 8. Identify Patterns and Trends")
        st.write("**Trends based on time (if time-related column exists):**")
        
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if len(date_columns) > 0:
            date_col = st.selectbox("Select a Date/Time Column", date_columns)
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df.set_index(date_col, inplace=True)
            df.resample('M').mean().plot()
            plt.title("Monthly Trend")
            st.pyplot(plt)
        else:
            st.warning("No 'Date' or 'Time' column found to analyze trends.")
    except Exception as e:
        st.error(f"Error in Identifying Patterns and Trends: {e}")

# Step 9: Insights Summary
def insights_summary(df):
    try:
        st.write("### 9. Insights Summary")
        st.write("**Summary of the analysis:**")
        
        st.write("**Data Overview:**")
        st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
        
        st.write("**Statistical Summary:**")
        st.write(df.describe())
        
        st.write("**Potential Issues Detected:**")
        missing_values = df.isnull().sum()
        st.write(f"Missing values in columns: {missing_values[missing_values > 0]}")
        
        st.write("**Suggestions for improvement:**")
        st.write("Check for any further feature engineering or imputation of missing values.")
    except Exception as e:
        st.error(f"Error in Insights Summary: {e}")

# Step 10: X and Y Column Visualization (Scatter, Line, Bar)
def x_y_visualization(df):
    try:
        st.write("### 10. Custom X and Y Column Visualization")
        x_column = st.selectbox("Select X column", df.columns)
        y_column = st.selectbox("Select Y column", df.columns)
        plot_type = st.selectbox("Select Plot Type", ['scatter', 'line', 'bar', 'histogram'])
        
        if plot_type == 'scatter':
            plt.scatter(df[x_column], df[y_column])
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.title(f"Scatter Plot of {x_column} vs {y_column}")
        elif plot_type == 'line':
            df.plot(x=x_column, y=y_column, kind='line')
            plt.title(f"Line Plot of {x_column} vs {y_column}")
        elif plot_type == 'bar':
            df.plot(x=x_column, y=y_column, kind='bar')
            plt.title(f"Bar Plot of {x_column} vs {y_column}")
        elif plot_type == 'histogram':
            df.plot(x=x_column, y=y_column, kind='hist', bins=30)
            plt.title(f"Histogram of {x_column} and {y_column}")
        
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error in X and Y Visualization: {e}")

def main():
    df = upload_file()
    
    if df is not None:
        show_full_data(df)
        data_understanding(df)
        data_cleaning(df)
        univariate_analysis(df)
        multivariate_analysis(df)
        feature_engineering(df)
        anomaly_detection(df)
        data_visualization(df)
        identify_patterns_and_trends(df)
        insights_summary(df)
        x_y_visualization(df)

if __name__ == "__main__":
    main()

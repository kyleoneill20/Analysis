# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os
import warnings

# Suppressing the warnings for a clean output
warnings.filterwarnings("ignore")

# Function to load and merge the data
def load_and_merge_data(file_path):
    if not os.path.exists(file_path):
        print("File path does not exist. Please check the path.")
        exit()

    excel_data = pd.ExcelFile(file_path)
    try:
        value_info_data = excel_data.parse('Value Info')
        demographic_info_data = excel_data.parse('Demographic Info')
    except ValueError as e:
        print(f"Error reading sheets from Excel file: {e}")
        exit()

    merged_data = pd.merge(value_info_data, demographic_info_data, on='Customer ID')
    merged_data['Title'] = merged_data['Title'].fillna('Unknown')  # Handle missing titles
    return merged_data

# Function used to explore data
def explore_data(data):
    print("\n--- Data Overview ---")
    print(data.info())
    print("\n--- Missing Values ---")
    print(data.isnull().sum())
    print("\n--- Statistical Summary ---")
    print(data.describe())

# Function to handle outliers in the promotion values
def handle_outliers(merged_data):
    merged_data = merged_data[merged_data['Total value of all promotions'] >= 0]  # Remove negatives
    lower_bound = merged_data['Total value of all promotions'].quantile(0.01)
    upper_bound = merged_data['Total value of all promotions'].quantile(0.99)
    merged_data = merged_data[
        merged_data['Total value of all promotions'].between(lower_bound, upper_bound)
    ]
    return merged_data

# Function to analyze top performing cities
def analyze_top_cities(merged_data):
    city_demographics = merged_data.groupby('Address City').agg({
        'Customer ID': 'count',
        '1st Order Profit': 'mean',
        'Subsequent Order Profit': 'mean',
        'Subsequent Orders Count': 'mean',
        'Total value of all promotions': 'mean'
    }).rename(columns={'Customer ID': 'Customer Count'}).sort_values(by='Customer Count', ascending=False)
    
    print(city_demographics.head(10))
    city_demographics.head(10)['Customer Count'].plot(kind='bar', figsize=(10, 6), title='Top Cities by Customer Count', color='skyblue', edgecolor='black')
    plt.ylabel('Customer Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("top_cities_chart.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n--- Insights for Top Cities ---")
    print("Focus marketing efforts in top cities like Dublin and Cork where customer density is highest. "
          "Consider addressing cities with medium customer density like Galway or Kilkenny to improve engagement.")
    return city_demographics

# Function to analyze age groups
def analyze_age_groups(merged_data):
    merged_data['Age'] = pd.to_datetime('today').year - pd.to_datetime(merged_data['Date Of Birth']).dt.year
    merged_data = merged_data[(merged_data['Age'] > 18) & (merged_data['Age'] < 100)]  # Remove outliers
    merged_data['Age Group'] = pd.cut(
        merged_data['Age'], bins=[30, 35, 45, 55, 65, 100],
        labels=['31-35', '36-45', '46-55', '56-65', '65+']
    )
    age_group_analysis = merged_data.groupby('Age Group', observed=False).agg({
        'Customer ID': 'count',
        '1st Order Profit': 'mean',
        'Subsequent Order Profit': 'mean'
    }).rename(columns={'Customer ID': 'Customer Count'})

    print(age_group_analysis)
    age_group_analysis['Customer Count'].plot(kind='bar', figsize=(10, 6), title='Customer Count by Age Group', color='lightgreen', edgecolor='black')
    plt.ylabel('Customer Count')
    plt.xlabel('Age Group')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("age_group_chart.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n--- Insights for Age Groups ---")
    print("The 36-45 and 46-55 age groups contribute the most customers. "
          "Focus promotions targeting these groups while exploring opportunities in the 56-65 bracket, which has high profits.")
    return age_group_analysis

# Function to analyze and visualize channel performance
def analyze_channel_performance(merged_data):
    channel_performance = merged_data.groupby('Source of Customer').agg({
        'Customer ID': 'count',
        '1st Order Profit': 'sum',
        'Subsequent Order Profit': 'sum',
        'Total value of all promotions': 'sum'
    }).rename(columns={'Customer ID': 'Customer Count'})
    channel_performance['Total Revenue'] = channel_performance['1st Order Profit'] + channel_performance['Subsequent Order Profit']
    
    print("\n--- Channel Performance ---")
    print(channel_performance)
    channel_performance['Total Revenue'].sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), title='Total Revenue by Channel', color='orange', edgecolor='black')
    plt.ylabel('Revenue')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("channel_performance_chart.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n--- Insights for Channel Performance ---")
    print("Direct and Organic Search channels generate the most revenue. "
          "Allocate resources to strengthen these while enhancing performance in Paid Search and Affiliates.")
    return channel_performance

# Function to analyze customer retention
def analyze_customer_retention(merged_data):
    retention_analysis = merged_data.groupby('Subsequent Orders Count').agg({
        'Customer ID': 'count',
        'Subsequent Order Profit': 'mean'
    }).rename(columns={'Customer ID': 'Customer Count'})
    retention_analysis = retention_analysis[retention_analysis['Customer Count'] > 10]
    print(retention_analysis)
    retention_analysis['Customer Count'].plot(kind='line', figsize=(10, 6), title='Customer Retention Trends', color='purple', marker='o')
    plt.ylabel('Customer Count')
    plt.xlabel('Subsequent Orders Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("customer_retention_chart.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n--- Insights for Customer Retention ---")
    print("Retention trends are showing consistent profit increases with subsequent orders. "
          "Encourage repeat purchases through loyalty programs targeting customers with 1-3 orders.")
    return retention_analysis

# Function to prepare data for churn prediction
def prepare_data_for_churn_prediction(merged_data):
    merged_data['Churn'] = (merged_data['Subsequent Orders Count'] == 0).astype(int)
    merged_data['Age'] = pd.to_datetime('today').year - pd.to_datetime(merged_data['Date Of Birth']).dt.year
    features = merged_data[['Age', '1st Order Profit', 'Total value of all promotions']]
    features = pd.get_dummies(features, drop_first=True).fillna(0)
    target = merged_data['Churn']
    return features, target

# Function to visualize the confusion matrix
def visualize_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

# Function to build and evaluate churn prediction model
def build_churn_prediction_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n--- Churn Prediction Model Evaluation ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")

    # Visualize the confusion matrix
    visualize_confusion_matrix(y_test, y_pred)

    return model
    
# Main script
if __name__ == "__main__":
    
    file_path = "/Users/kyleoneill/Downloads/2k_Submission_Folder/Quantitative_Task_-_Data_Analysis_-_Auction.xlsx"

    print("\n--- Loading and Merging Data ---")
    data = load_and_merge_data(file_path)

    print("\n--- Exploring Dataset ---")
    explore_data(data)

    print("\n--- Handling Outliers ---")
    data = handle_outliers(data)

    print("\n--- Top Cities Analysis ---")
    analyze_top_cities(data)

    print("\n--- Age Group Analysis ---")
    analyze_age_groups(data)

    print("\n--- Channel Performance ---")
    analyze_channel_performance(data)

    print("\n--- Customer Retention ---")
    analyze_customer_retention(data)

    print("\n--- Preparing Data for Churn Prediction ---")
    features, target = prepare_data_for_churn_prediction(data)

    print("\n--- Building Churn Prediction Model ---")
    build_churn_prediction_model(features, target)

# Analysis
Analysis on a men's fashion retailers data
Project Submission for Data Analysis and URL Investigation

Project Overview

This project includes two major parts:

Data Analysis: A detailed exploration and analysis of customer data for a men's fashion retailer to provide actionable insights.

URL Investigation: Analysis of Marks & Spencer’s URL structure to identify deep-linked pages for specific product categories.
Project Overview

This document outlines the analysis of customer and marketing data for a men’s fashion retailer. The objective was to uncover actionable insights from key areas such as top-performing cities, customer age groups, channel performance, customer retention trends, and churn prediction.

Analysis and Recommendations
1. Top Performing Cities
Findings:
Dublin has the highest customer count (20,853), making it the largest revenue contributor.
Medium-density cities such as Cork (1,028), Galway (632), and Kilkenny (260) show potential for growth.
Recommendations:
Maximize Dublin's Performance:
Enhance marketing efforts with exclusive campaigns for Dublin’s customer base.
Optimize logistics and services (faster delivery) to maintain high customer satisfaction.
Expand in Medium Density Cities:
Create geo targeted online campaigns for Cork, Galway, and Kilkenny.
Offer promotional discounts and referral programs to incentivize purchases in these regions.
Supporting Visualization:
Refer to the top_cities_chart.png file for a bar chart representation of customer counts by city.

2. Age Group Analysis
Findings:
The 36-45 age group has the highest customer count (13,227), followed by the 46-55 group (7,345).
Customers in the 56-65 age group are fewer (3,312) but generate higher profitability per transaction.
Recommendations:
Focus on Dominant Age Groups:
Develop targeted promotions for customers aged 36-55 (product bundles or loyalty programs tailored to their needs).
Engage the 56-65 Group:
Highlight premium product offerings and customer experiences that appeal to mature customers.
Launch campaigns that emphasise quality and value.
Supporting Visualization:
The bar chart age_group_chart.png shows customer distribution across age groups.

3. Channel Performance
Findings:
Direct and Organic Search channels are the most profitable, generating revenues of $515,120 and $392,513, respectively.
Underperforming channels include Paid Social ($82,719) and Affiliates ($133,201).
Recommendations:
Strengthen High-Performing Channels:
Invest further in SEO strategies to boost Organic Search traffic.
Optimize Direct campaigns with personalized offers and retargeting ads.
Improve Underperforming Channels:
For Paid Social:
Use dynamic ads tailored to user behavior.
Focus on fashion oriented platforms such as Instagram or Pinterest.
For Affiliates:
Partner with influencers and content creators in the fashion industry to boost traffic and conversions.
Supporting Visualization:
The bar chart channel_performance_chart.png displays revenue by channel.

4. Customer Retention
Findings:
Customer count drops significantly after the first purchase but stabilises with repeat orders.
Customers making 3+ subsequent purchases generate consistently higher profits.
Recommendations:
Retain First-Time Customers:
Provide substantial discounts or incentives for second purchases.
Use personalized email campaigns to engage first-time buyers.
Encourage Repeat Purchases:
Implement a loyalty program with tiered rewards for customers with 1-3 orders.
Introduce subscription models (e.g., curated seasonal fashion boxes) to encourage frequent purchases.
Supporting Visualization:
The line chart customer_retention_chart.png depicts retention trends across subsequent orders.

5. Churn Prediction
Findings:
A Random Forest Classifier was built to predict churn with an accuracy score of 70%.
The confusion matrix shows the model's ability to differentiate between churn and non churn customers.
Recommendations:
Proactively Engage At-Risk Customers:
Use predictive insights to identify customers at risk of churn.
Design targeted offers, such as discounts or free shipping, to retain these customers.
Improve Model Accuracy:
Continuously train the churn prediction model with updated data to improve performance.
Integrate the model with CRM systems for real-time churn monitoring.
Supporting Visualization:
Refer to confusion_matrix.png for the confusion matrix visualization of the churn model's performance.

Submission Files
Charts:
top_cities_chart.png
age_group_chart.png
channel_performance_chart.png
customer_retention_chart.png
confusion_matrix.png
Code: TaskAnalysis.py
Dataset: Quantitative_Task_-_Data_Analysis_-_Auction.xlsx

Part 2: Technology – URL Investigation
Objective
Marks & Spencer wanted specific deep-linked URLs for targeted marketing campaigns. The investigation focused on understanding their URL structure and creating precise links for:

Regular Fit Shirts
Slim Fit Shirts
Blue Shirts
White Shirts
Green Shirts
Regular Fit Blue/White/Green Shirts
Slim Fit Blue/White/Green Shirts
URL Structure Explanation
Base URL: https://www.marksandspencer.com/ie/
Category Path: /l/men/mens-shirts specifies the product category.
Filters: Keywords like fs5/slim-fit or fs5/blue apply filters for specific criteria.
Filter Combination: Multiple filters can be combined to refine product lists.
More detail on doc.

Submission Files
Code: TaskAnalysis.py – Contains all analysis, visualizations, and churn model scripts.
Charts:
age_group_chart.png
channel_performance_chart.png
confusion_matrix.png
customer_retention_chart.png
top_cities_chart.png
URL Analysis: Part 2 - URLs.docx.
Excel Data: Quantitative_Task_-_Data_Analysis_-_Auction.xlsx.

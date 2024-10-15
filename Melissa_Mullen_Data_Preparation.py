# Melissa Mullen
# MET CS 777
# 10/15/2024

# FINAL PROJECT


##### DATA PREPARATION: 
# Create random sample of 10,000 values for processing on local machine
# Data was pulled from here: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/data?select=US_Accidents_March23.csv
# Note - Data was the sampled, 500k row version also available at that link

import pandas as pd

# Load original CSV file of 500,000 rows
data = pd.read_csv("US_Accidents_March23_sampled_500k.csv")

# Removing rows with missing values before sampling
clean_data = data.dropna()

# Explore target column 
severity_counts = clean_data.groupby("Severity").size().reset_index(name='count')

# Show the results
print(severity_counts)

# Separate majority and minority classes
majority_class = clean_data[clean_data['Severity'].isin([1, 2])]
minority_class = clean_data[clean_data['Severity'].isin([3, 4])]

# Oversample minority class (Severity 1)
minority_class_sampled = minority_class.sample(n=len(majority_class), replace=True, random_state=599)

# Combine the majority class with the oversampled minority class
balanced_data = pd.concat([majority_class, minority_class_sampled])
# Take a random sample of 10,000 rows
# Set random state for reproducibility (last three of BU ID)
sample_data = balanced_data.sample(n=10000, random_state = 599) 

severity_counts = sample_data.groupby("Severity").size().reset_index(name='count')

# Show the results
print(severity_counts)

# print sample_data information to confirm correct sampling
print(f"Shape of Sampled Dataset: {sample_data.shape}")

print(sample_data.head())

# export to CSV
sample_data.to_csv("US_Accidents_Sampled.csv", index=False)


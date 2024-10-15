# Final Project - US Accident Severity Prediciton
Melissa Mullen  
MET CS 777  
10/15/2024


## Project Overview:
This project uses PySpark to analyze and predict the severity of motor vehicle accidents based on weather-related condtions. Logistic Regression, Support Vector Machine (SVM), and Random Forest models will be trained and evaluated.

## Dataset:
The full dataset - US Accidents (2016-2023) is available on Kaggle to download as a CSV file here: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/data?select=US_Accidents_March23.csv    
Kaggle also offers a sampled version of this dataset with just 500,000 rows, which is available for download here: https://drive.google.com/file/d/1m2w1Ml6OHxm6jfXk7UxWPcZFhZTQwJwE/view?pli=1  
To run this project on my local machine, I created a sample of the sampled dataset above, which only includes 10,000 rows. It is available for download here: https://drive.google.com/file/d/1JaLGE2h3UE4brkAsIAvnBq3rEaOTXepO/view

## Requirements:
To run this project, you must have the following installed:
* Python 3.x
* PySpark
* Java JDK (required for Spark)

## Instructions:
To run this code, you must first download the dataset. If you want to bypass using `Melissa_Mullen_Data_Preparation.py`, you can download the 10,000 row sampled dataset titled `US_Accidents_Sampled.csv`. This file can be loaded to the `Melissa_Mullen_Final_Project.py` script.   
However, if you want to sample the sampled dataset as well, download the 500,000 row dataset titled `US_Accidents_March23_sampled_500k.csv`. This file can be loaded into the `Melissa_Mullen_Data_Preparation.py` script, which can be run with a basic spark-submit command. This script will generate the 10,000 row `US_Accidents_Sampled.csv` file, which will be used in `Melissa_Mullen_Final_Project.py`. This file can also be run with a basic spark-submit command as well. 

In short, download `US_Accidents_Sampled.csv` to only run `Melissa_Mullen_Final_Project.py`. If you want to generate the sampled dataset for this project, download `US_Accidents_March23_sampled_500k.csv` to run `Melissa_Mullen_Data_Preparation.py`. The outcome of this script will be a CSV title named `US_Accidents_Sampled.csv`, which can be loaded into `Melissa_Mullen_Final_Project.py`. Both .py scripts can be run with a basic spark-submit command. 

Feel free to edit the filepaths within the code themselves, or bypass the hard coding by including the CSV files in your spark-submit command. 

## Questions?
Feel free to contact me! MDMullen@bu.edu.
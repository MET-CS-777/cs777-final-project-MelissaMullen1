# Melissa Mullen
# MET CS 777
# 10/15/2024

# FINAL PROJECT
# spark-submit --master local Final_Project.py US_Accidents_Sampled.csv

# Import libraries
import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import to_timestamp, unix_timestamp, col, substring, when
from pyspark.ml.classification import LogisticRegression, LinearSVC, RandomForestClassifier


if __name__ == "__main__":

    # Check for input file in argument, else use provided file
    if len(sys.argv) != 2:
        print("Manual input and output files")
        data = "file:///C:/Users/Bean/Desktop/MET_CS_777/Project/US_Accidents_Sampled.csv"  # change here
    else:
        print("Command line input and output files")
        data = (sys.argv[1])

    # Initialize spark context and session
    sc = SparkContext(appName="Project")
    spark = SparkSession.builder.getOrCreate()

    # Set log level to ERROR - mute warnings
    sc.setLogLevel("ERROR")
    
    # Read input file (CSV) as Spark Dataframe
    df = spark.read.csv(data, header=True, inferSchema=True)

    ### DATA PREPARATION: ###
    print("\nData Preparation:\n")
    
    # select relevant columns, where Severity will be the target variable
    # Split into two, where if Severity = 1 or 2, label as 0, else (3 or 4) label as 1
    df = df.withColumn("Severity_Binary", when(col("Severity").isin([1, 2]), 0).otherwise(1))

    # Count the number of accidents with severity 0 and 1 and show results
    # This is used to check balance of dataset
    severity_counts = df.groupBy("Severity_Binary").count()
    severity_counts.show()

    # Gather distinct values in the 'Weather_Condition' column
    distinct_weather_conditions = df.select("Weather_Condition").distinct().collect()

    # Print out all the distinct weather conditions
    # Do this to gather information for categorizing weather below
    print("Distinct Weather Conditions:")
    for row in distinct_weather_conditions:
        print(row["Weather_Condition"])

    # Group weather conditions into categories
    # Categories = Clear/Fair, Cloudy, Rain, Fog/Haze/Smoke, Thunderstorm, Snow/Ice/Freezing, Unknown
    df = df.withColumn(
        "Weather_Grouped",
        when(col("Weather_Condition").isin(
            ["Clear", "Fair", "Fair / Windy", "N/A Precipitation"]), "Clear/Fair")
        .when(col("Weather_Condition").isin(
            ["Cloudy", "Mostly Cloudy", "Partly Cloudy", "Overcast", "Cloudy / Windy",
            "Mostly Cloudy / Windy", "Partly Cloudy / Windy", "Scattered Clouds"]), "Cloudy")
        .when(col("Weather_Condition").isin(
            ["Light Rain", "Light Rain / Windy", "Rain", "Rain / Windy", "Heavy Rain", 
            "Heavy Rain / Windy", "Drizzle", "Light Drizzle", "Light Drizzle / Windy",
            "Heavy Drizzle", "Drizzle and Fog", "Showers in the Vicinity", "Light Rain Shower"]), "Rain")
        .when(col("Weather_Condition").isin(
            ["Fog", "Patches of Fog", "Fog / Windy", "Haze", "Haze / Windy", 
            "Smoke", "Shallow Fog", "Smoke / Windy"]), "Fog/Haze/Smoke")
        .when(col("Weather_Condition").isin(
            ["Thunder", "Thunder / Windy", "T-Storm", "T-Storm / Windy", "Heavy T-Storm", 
            "Heavy T-Storm / Windy", "Thunder in the Vicinity", "Thunderstorms and Rain", 
            "Thunder and Hail", "Light Rain with Thunder"]), "Thunderstorm")
        .when(col("Weather_Condition").isin(
            ["Light Snow", "Snow", "Heavy Snow", "Heavy Snow / Windy", "Light Freezing Rain", "Light Snow / Windy",
            "Light Freezing Drizzle", "Wintry Mix", "Blowing Snow / Windy", "Snow / Windy", "Light Ice Pellets"]), "Snow/Ice/Freezing")
        .otherwise("Unknown")
    )

    # Handle timestamp columns, convert to this format to avoid PySpark error
    df = df.withColumn("start_time", substring(col("Start_Time"), 1, 19)) \
       .withColumn("end_time", substring(col("End_Time"), 1, 19))

    # Convert start_time and end_time to correct timestamp format
    df = df.withColumn("start_time", to_timestamp(col("Start_Time"), "yyyy-MM-dd HH:mm:ss")) \
        .withColumn("end_time", to_timestamp(col("End_Time"), "yyyy-MM-dd HH:mm:ss"))

    # Calculate duration of accident in seconds, then convert to minutes and drop seconds column
    df = df.withColumn("Duration_Seconds", unix_timestamp(col("end_time")) - unix_timestamp(col("start_time")))
    df = df.withColumn("Duration_Minutes", col("Duration_Seconds") / 60)
    df = df.drop("Duration_Seconds")

    # Select relevant columns
    data = df.select("Severity_Binary", "Distance(mi)", "Humidity(%)", "Pressure(in)", "Temperature(F)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "Weather_Grouped", "Duration_Minutes")
   
    # rename columns for ease of use
    data = data.withColumnRenamed("Temperature(F)", "Temperature") \
               .withColumnRenamed("Visibility(mi)", "Visibility") \
               .withColumnRenamed("Wind_Speed(mph)", "Wind_Speed") \
               .withColumnRenamed("Precipitation(in)", "Precipitation") \
               .withColumnRenamed("Humidity(%)", "Humidity") \
               .withColumnRenamed("Pressure(in)", "Pressure") \
               .withColumnRenamed("Distance(mi)", "Distance")
    
    ### FEATURE ENCODING, VECTOR ASSEMBLY AND SCALING: ###

    # Weather_Condition is a string, meaning it will require encoding
    # Will use StringIndexer() to index values, then apply OneHotEncoder()
    # Reference: https://www.machinelearningplus.com/pyspark/pyspark-stringindexer/
    # Reference: https://www.machinelearningplus.com/pyspark/pyspark-onehot-encoding/
    indexer_weather = StringIndexer(inputCol="Weather_Grouped", outputCol="Weather_Index", handleInvalid="keep")
    encoder_weather = OneHotEncoder(inputCol="Weather_Index", outputCol="Weather")

    # Vectorize data
    assembler = VectorAssembler(
        inputCols=["Humidity", "Distance", "Pressure", "Temperature", "Visibility", "Wind_Speed", "Precipitation", "Weather", "Duration_Minutes"],
        outputCol="features"
    )

    # Scale/standardize features to maximize model performance:
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")


    ### Logistic Regression Model: ###
    # General reference for logistic regression: https://www.geeksforgeeks.org/logistic-regression-using-pyspark-python/

    print("\nLogistic Regression:\n")

    # initialize logistic regression model
    lr = LogisticRegression(featuresCol="scaled_features", labelCol="Severity_Binary")

    # Create pipeline with indexer, encoder, assembler, scaler, and logistic regression
    pipeline = Pipeline(stages=[indexer_weather, encoder_weather, assembler, scaler, lr])

    # Split the data into training and testing sets
    # 20 80 split, with random seed for reproducibility
    train_df, test_df = data.randomSplit([0.8, 0.2], seed=599)

    # Train the model
    model = pipeline.fit(train_df)

    # Evaluate the model on the test set
    predictions = model.transform(test_df)

    # Extract predictions and labels for evaluation
    predictionAndLabels = predictions.select("Severity_Binary", "prediction").rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()

    # Calculate accuracy, precision, recall, F1 score, and confusion matrix
    metrics = MulticlassMetrics(predictionAndLabels)
    accuracy = round(metrics.accuracy, 4)
    precision = round(metrics.precision(1.0), 4)  # Precision for class 1.0 (severe accidents)
    recall = round(metrics.recall(1.0), 4)        # Recall for class 1.0
    f1Score = round(metrics.fMeasure(1.0), 4)     # F1 Score for class 1.0
    confusion_matrix = metrics.confusionMatrix().toArray().astype(int)

    # Print metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1Score}")
    print("Confusion Matrix:")
    print(confusion_matrix)


    ### SUPPORT VECTOR MACHINE: ###
    # Reference: https://www.datatechnotes.com/2022/02/pyspark-linear-svc-classification.html
    print("\nSupport Vector Machine:\n")

    # Define Linear SVM Classifier
    svm = LinearSVC(featuresCol="scaled_features", labelCol="Severity_Binary", maxIter=100)

    # Create pipeline for SVM
    pipeline_svm = Pipeline(stages=[indexer_weather, encoder_weather, assembler, scaler, svm])

    # Train the SVM model
    svm_model = pipeline_svm.fit(train_df)

    # Evaluate the SVM model on the test data
    predictions_svm = svm_model.transform(test_df)
    predictionAndLabels_svm = predictions_svm.select("Severity_Binary", "prediction").rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()
    metrics_svm = MulticlassMetrics(predictionAndLabels_svm)

    # Print SVM results
    svm_accuracy = round(metrics_svm.accuracy, 4)
    svm_precision = round(metrics_svm.precision(1.0), 4)
    svm_recall = round(metrics_svm.recall(1.0), 4)
    svm_f1Score = round(metrics_svm.fMeasure(1.0), 4)
    svm_confusion_matrix = metrics_svm.confusionMatrix().toArray().astype(int)

    print(f"SVM Accuracy: {svm_accuracy}")
    print(f"SVM Precision: {svm_precision}")
    print(f"SVM Recall: {svm_recall}")
    print(f"SVM F1 Score: {svm_f1Score}")
    print("Confusion Matrix (SVM):")
    print(svm_confusion_matrix)

    ### Random Forest Classifier: ###
    # Reference: https://towardsdatascience.com/a-guide-to-exploit-random-forest-classifier-in-pyspark-46d6999cb5db
    print("\nRandom Forest Classifier:\n")

    # Define Random Forest Model
    rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="Severity_Binary", numTrees=100)

    # Create pipeline for Random Forest model
    pipeline_rf = Pipeline(stages=[indexer_weather, encoder_weather, assembler, scaler, rf])

    # Train Random Forest model
    rf_model = pipeline_rf.fit(train_df)
    predictions_rf = rf_model.transform(test_df)

    # Evaluate using MulticlassMetrics
    predictionAndLabels_rf = predictions_rf.select("Severity_Binary", "prediction").rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()
    metrics_rf = MulticlassMetrics(predictionAndLabels_rf)

    # Print SVM results with rounded values
    rf_accuracy = round(metrics_rf.accuracy, 4)
    rf_precision = round(metrics_rf.precision(1.0), 4)
    rf_recall = round(metrics_rf.recall(1.0), 4)
    rf_f1Score = round(metrics_rf.fMeasure(1.0), 4)
    rf_confusion_matrix = metrics_rf.confusionMatrix().toArray().astype(int)

    print(f"Random Forest Accuracy: {rf_accuracy}")
    print(f"Random Forest Precision: {rf_precision}")
    print(f"Random Forest Recall: {rf_recall}")
    print(f"Random Forest F1 Score: {rf_f1Score}")
    print("Confusion Matrix (Random Forest):")
    print(rf_confusion_matrix)


    ### Random Forest Classifier with Cross-Validation: ###
    # Reference: https://www.machinelearningplus.com/pyspark/pyspark-random-forest/
    print("\nRandom Forest Classifier with Cross-Validation:\n")


    # Create parameter grid for hyperparameter tuning
    paramGrid_rf = (ParamGridBuilder()
                    .addGrid(rf.numTrees, [50, 100, 200])
                    .addGrid(rf.maxDepth, [5, 10, 20])
                    .build())

    # Cross-validation with three folds 
    evaluator = BinaryClassificationEvaluator(labelCol="Severity_Binary", metricName="areaUnderROC")
    crossval_rf = CrossValidator(estimator=pipeline_rf,
                                estimatorParamMaps=paramGrid_rf,
                                evaluator=evaluator,
                                numFolds=3)

    # Train the cross-validation model
    cvModel_rf = crossval_rf.fit(train_df)

    # Evaluate on test data
    predictions_rf = cvModel_rf.transform(test_df)
    predictionAndLabels_rf = predictions_rf.select("Severity_Binary", "prediction").rdd.map(lambda x: (float(x[0]), float(x[1]))).cache()
    metrics_cvrf = MulticlassMetrics(predictionAndLabels_rf)

    # Print metrics

    # Print SVM results with rounded values
    cvrf_accuracy = round(metrics_cvrf.accuracy, 4)
    cvrf_precision = round(metrics_cvrf.precision(1.0), 4)
    cvrf_recall = round(metrics_cvrf.recall(1.0), 4)
    cvrf_f1Score = round(metrics_cvrf.fMeasure(1.0), 4)
    cvrf_confusion_matrix = metrics_cvrf.confusionMatrix().toArray().astype(int)

    print(f"Random Forest (Cross Validation) Accuracy: {cvrf_accuracy}")
    print(f"Random Forest (Cross Validation): {cvrf_precision}")
    print(f"Random Forest (Cross Validation): {cvrf_recall}")
    print(f"Random Forest (Cross Validation): {cvrf_f1Score}")
    print("Confusion Matrix (Random Forest (Cross Validation)):")
    print(cvrf_confusion_matrix)

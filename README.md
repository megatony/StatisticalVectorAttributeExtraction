# StatisticalVectorAttributeExtraction

This project process a CSV dataset file that contains gyroscope and accelerometer sensor data.

ACC_X, ACC_Y and ACC_Z means accelerometer sensor's values on 3-axis.
GYRO_X, GYRO_Y and GYRO_Z means accelerometer sensor's values on 3-axis. 

Due to the achieve vector calculations, gyroscope and accelerometer parameters merged as a single vector.

Then, Apache Spark and Breeze libraries used to extract statistical attributes which are magnitude, mean, median and variance.

Finally resultant dataframe extracted as CSV file.

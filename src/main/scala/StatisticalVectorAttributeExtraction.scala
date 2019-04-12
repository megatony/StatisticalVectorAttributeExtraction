import breeze.linalg.{DenseVector => BreezeDenseVector}
import org.apache.spark.ml.feature.{VectorAssembler}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SQLContext}

object StatisticalVectorAttributeExtraction {
  val PATH = "src\\main\\resources\\"
  val FILENAME = "demo_preprocessed_glasses.csv"

  var conf = new SparkConf().setAppName("Feature extraction").setMaster("local[*]")
  val sc = new SparkContext(conf)
  val sqlContext = new SQLContext(sc)
  import sqlContext.implicits._
  import org.apache.spark.sql.functions._

  def main(args: Array[String]): Unit = {
    val dataFrame = getReportDataFrame()

    val accAssembler = new VectorAssembler()
      .setInputCols(Array("ACC_X", "ACC_Y", "ACC_Z"))
      .setOutputCol("ACC_TO_VECTOR")
      .transform(dataFrame)
    val gyroAssembler = new VectorAssembler()
      .setInputCols(Array("GYRO_X", "GYRO_Y", "GYRO_Z"))
      .setOutputCol("GYRO_TO_VECTOR")
      .transform(accAssembler)

    val dfWithId = gyroAssembler.withColumn("rowId", monotonically_increasing_id())

    val dfWithAccMagnitude = addAccelerometerMagnitudeToDataFrame(dfWithId)
    val dfWithGyroMagnitude = addGyroscopeMagnitudeToDataFrame(dfWithAccMagnitude)

    val dfWithAccMean = addAccelerometerMeanValueToDataFrame(dfWithGyroMagnitude)
    val dfWithGyroMean = addGyroscopeMeanValueToDataFrame(dfWithAccMean)

    val dfWithAccMedian = addAccelerometerMedianToDataFrame(dfWithGyroMean)
    val dfWithGyroMedian = addGyroscopeMedianToDataFrame(dfWithAccMedian)

    val dfWithAccVariance = addAccelerometerVarianceToDataFrame(dfWithGyroMedian)
    val dfWithGyroVariance = addGyroscopeVarianceToDataFrame(dfWithAccVariance).orderBy("rowId")

    val result = dfWithGyroVariance
      .select("rowId",
        "ACC_X","ACC_Y","ACC_Z",
        "GYRO_X","GYRO_Y","GYRO_Z",
        "activity_type",
        "duration",
        "from",
        "to",
        "ACC_MAG", "GYRO_MAG",
        "ACC_MEAN", "GYRO_MEAN",
        "ACC_MEDIAN", "GYRO_MEDIAN",
        "ACC_VARIANCE", "GYRO_VARIANCE")

    writeDataFrameToFile("result.csv", FILENAME, result)
  }

  def addAccelerometerMagnitudeToDataFrame(dataFrame: DataFrame) : DataFrame = {
    val rddAssembled = dataFrame.select("ACC_TO_VECTOR").rdd.map(x=>x.getAs[DenseVector](0))
    val rddAssembledMagnitude = rddAssembled.map(x=> magnitude(x.toDense.values)).toDF()

    val rddAssembledMagnitudeWithId = rddAssembledMagnitude.withColumn("rowId", monotonically_increasing_id())
      .withColumnRenamed("value", "ACC_MAG")
    dataFrame.join(rddAssembledMagnitudeWithId, "rowId")
  }

  def addGyroscopeMagnitudeToDataFrame(dataFrame: DataFrame) : DataFrame = {
    val rddAssembled = dataFrame.select("GYRO_TO_VECTOR").rdd.map(x=>x.getAs[DenseVector](0))
    val rddAssembledMagnitude = rddAssembled.map(x=> magnitude(x.toDense.values)).toDF()

    val dataFrameWithId = dataFrame.withColumn("rowId", monotonically_increasing_id())
    val rddAssembledMagnitudeWithId = rddAssembledMagnitude.withColumn("rowId", monotonically_increasing_id())
      .withColumnRenamed("value", "GYRO_MAG")
    dataFrameWithId.join(rddAssembledMagnitudeWithId, "rowId")
  }

  def addAccelerometerMeanValueToDataFrame(dataFrame: DataFrame) = {
    val dfWithAccMean = dataFrame.select(col("rowId"), col("ACC_TO_VECTOR")).as[(BigInt, DenseVector)]
      .map { case (group, v) => (group, breeze.stats.mean(BreezeDenseVector(v.toArray)) ) }
      .toDF().withColumnRenamed("_1", "rowId").withColumnRenamed("_2", "ACC_MEAN")
    dataFrame.join(dfWithAccMean, "rowId")
  }

  def addGyroscopeMeanValueToDataFrame(dataFrame: DataFrame) = {
    val dfWithGyroMean = dataFrame.select(col("rowId"), col("GYRO_TO_VECTOR")).as[(BigInt, DenseVector)]
      .map { case (group, v) => (group, breeze.stats.mean(BreezeDenseVector(v.toArray)) ) }
      .toDF().withColumnRenamed("_1", "rowId").withColumnRenamed("_2", "GYRO_MEAN")
    dataFrame.join(dfWithGyroMean, "rowId")
  }

  def addAccelerometerMedianToDataFrame(dataFrame: DataFrame) = {
    val dfWithAccMedian = dataFrame.select(col("rowId"), col("ACC_TO_VECTOR")).as[(BigInt, DenseVector)]
      .map { case (group, v) => (group, breeze.stats.median(BreezeDenseVector(v.toArray)) ) }
      .toDF().withColumnRenamed("_1", "rowId").withColumnRenamed("_2", "ACC_MEDIAN")
    dataFrame.join(dfWithAccMedian, "rowId")
  }

  def addGyroscopeMedianToDataFrame(dataFrame: DataFrame) = {
    val dfWithGyroMedian = dataFrame.select(col("rowId"), col("GYRO_TO_VECTOR")).as[(BigInt, DenseVector)]
      .map { case (group, v) => (group, breeze.stats.median(BreezeDenseVector(v.toArray)) ) }
      .toDF().withColumnRenamed("_1", "rowId").withColumnRenamed("_2", "GYRO_MEDIAN")
    dataFrame.join(dfWithGyroMedian, "rowId")
  }

  def addAccelerometerVarianceToDataFrame(dataFrame: DataFrame) = {
    val dfWithAccVariance = dataFrame.select(col("rowId"), col("ACC_TO_VECTOR")).as[(BigInt, DenseVector)]
      .map { case (group, v) => (group, breeze.stats.variance(BreezeDenseVector(v.toArray)) ) }
      .toDF().withColumnRenamed("_1", "rowId").withColumnRenamed("_2", "ACC_VARIANCE")
    dataFrame.join(dfWithAccVariance, "rowId")
  }

  def addGyroscopeVarianceToDataFrame(dataFrame: DataFrame) = {
    val dfWithGyroVariance = dataFrame.select(col("rowId"), col("GYRO_TO_VECTOR")).as[(BigInt, DenseVector)]
      .map { case (group, v) => (group, breeze.stats.variance(BreezeDenseVector(v.toArray)) ) }
      .toDF().withColumnRenamed("_1", "rowId").withColumnRenamed("_2", "GYRO_VARIANCE")
    dataFrame.join(dfWithGyroVariance, "rowId")
  }

  def magnitude(x: Array[Double]): Double = {
    math.sqrt(x map(i => i*i) sum).toDouble
  }

  def mergeToCsvFile(srcPath: String, dstPath: String): Unit =  {
    val hadoopConfig = new Configuration()
    val hdfs = FileSystem.get(hadoopConfig)
    FileUtil.copyMerge(hdfs, new Path(srcPath), hdfs, new Path(dstPath), true, hadoopConfig, null)
  }

  def writeDataFrameToFile(outputFileName: String, mergedFileName : String, dataFrame: DataFrame) : Unit = {
    dataFrame.write
      .format("com.databricks.spark.csv")
      .option("header", "false")
      .mode("overwrite")
      .save(outputFileName)
    mergeToCsvFile(PATH + mergedFileName, PATH + outputFileName)
    dataFrame.unpersist()
  }

  def getReportDataFrame() : DataFrame = {
    sqlContext.read.format("com.databricks.spark.csv")
      .option("treatEmptyValuesAsNulls", "true")
      .option("header", "true")
      .option("inferSchema", "true")
      .option("dateFormat", "yyyy-MM-dd HH:mm")
      .option("mode","DROPMALFORMED")
      .load(PATH + FILENAME)
  }
}
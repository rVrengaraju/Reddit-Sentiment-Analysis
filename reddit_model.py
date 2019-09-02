from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext, DataFrame
from cleantext import sanitize
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def main(sqlContext):
    """Main function takes a Spark SQL context."""
    spark = SparkSession(SparkContext.getOrCreate())
    # Task 1
    # if parquet exists, read 
    try:
        # These are the two data frames we're working with
        comments = sqlContext.read.parquet("comments.parquet")
        submissions = sqlContext.read.parquet("submissions.parquet")
    except:
    # Otherwise do the following
        comments = sqlContext.read.json("comments-minimal.json.bz2")
        submissions = sqlContext.read.json("submissions.json.bz2")
        labeled_data = sqlContext.read.format("csv").option("header", "true").load("labeled_data.csv")
        comments.write.parquet("comments.parquet")
        submissions.write.parquet("submissions.parquet")

    # Task 2
    # join on labeled_data.Input_id and comments.id
    labeled_data = sqlContext.read.format("csv").option("header", "true").load("labeled_data.csv")
    labeled_data.createOrReplaceTempView("labeled_data")
    comments.createOrReplaceTempView("comments")
    sqlDF = spark.sql("SELECT l.id as id, l.body, r.labeldem as Dem,r.labelgop as GOP,r.labeldjt as Trump FROM labeled_data as r INNER JOIN comments as l ON r.Input_id = l.id ")
    # Task 3

    # Task 4 & Task 5
    sqlDF.createOrReplaceTempView("sqlDF")

    def parse(z):
        res1 = []
        res2 = []
        wordList = sanitize(z)
        for i, val in enumerate(wordList[1:]):
            res1.append(val)
        for i, value in enumerate(wordList[1:]):
            for j, val in enumerate((value.split(" "))):
                res2.append(val)
        return res1 + res2

    sqlContext.registerFunction("parser", lambda z: parse(z), ArrayType(StringType()))

    parsedTable = spark.sql("SELECT id, body, Trump, parser(body) as parsed FROM sqlDF")
    parsedTable.createOrReplaceTempView("parsedTable")
    
    # Task 6a
    # parsedTableID = spark.sql("SELECT id, parsed FROM parsedTable")
    cv = CountVectorizer(inputCol="parsed", outputCol="vectors", minDF=10.0)
    model = cv.fit(parsedTable)
    parsedVectorTable = model.transform(parsedTable)
    parsedVectorTable.createOrReplaceTempView("parsedVectorTable")

    # Task 6b
    resTable = spark.sql("SELECT id, body, Trump, vectors, CASE WHEN Trump=1 THEN 1 ELSE 0 END AS positive, CASE WHEN Trump=-1 THEN 1 ELSE 0 END AS negative FROM parsedVectorTable")

    # TASK 7
    # Initialize two logistic regression models.
    # Replace labelCol with the column containing the label, and featuresCol with the column containing the features.
    poslr = LogisticRegression(labelCol="label", featuresCol="vectors", maxIter=10)
    neglr = LogisticRegression(labelCol="label", featuresCol="vectors", maxIter=10)
    # This is a binary classifier so we need an evaluator that knows how to deal with binary classifiers.
    posEvaluator = BinaryClassificationEvaluator()
    negEvaluator = BinaryClassificationEvaluator()
    # There are a few parameters associated with logistic regression. We do not know what they are a priori.
    # We do a grid search to find the best parameters. We can replace [1.0] with a list of values to try.
    # We will assume the parameter is 1.0. Grid search takes forever.
    posParamGrid = ParamGridBuilder().addGrid(poslr.regParam, [1.0]).build()
    negParamGrid = ParamGridBuilder().addGrid(neglr.regParam, [1.0]).build()
    # We initialize a 5 fold cross-validation pipeline.
    posCrossval = CrossValidator(
        estimator=poslr,
        evaluator=posEvaluator,
        estimatorParamMaps=posParamGrid,
        numFolds=5)
    negCrossval = CrossValidator(
        estimator=neglr,
        evaluator=negEvaluator,
        estimatorParamMaps=negParamGrid,
        numFolds=5)
    # Although crossvalidation creates its own train/test sets for
    # tuning, we still need a labeled test set, because it is not
    # accessible from the crossvalidator (argh!)
    # Split the data 50/50
    pos = resTable.select("positive", "vectors")
    pos = pos.withColumnRenamed("positive", "label")
    neg = resTable.select("negative", "vectors")
    neg = neg.withColumnRenamed("negative", "label")

    posTrain, posTest = pos.randomSplit([0.5, 0.5])
    negTrain, negTest = neg.randomSplit([0.5, 0.5])
    # Train the models
    print("Training positive classifier...")
    posModel = posCrossval.fit(posTrain)
    print("Training negative classifier...")
    negModel = negCrossval.fit(negTrain)

    # Once we train the models, we don't want to do it again. We can save the models and load them again later.
    posModel.save("project2/pos.model")
    negModel.save("project2/neg.model")

    # Task 8
    print('task 8 started')
    def strip_t3(s):
        return s[3:]

    # Sample data if needed
    #comment out later
    # comments = comments.sample(False, 0.0002)

    sqlContext.registerFunction("strip_t3", lambda z: strip_t3(z), StringType())
    comments.createOrReplaceTempView("comments")
    submissions.createOrReplaceTempView("submissions")
    joined_data = spark.sql("SELECT c.created_utc as created_time, s.title as post_title, c.author_flair_text as com_state, c.body as body, c.id as comment_id, s.id as submission_id, s.score as s_score, c.score as c_score FROM comments as c INNER JOIN submissions as s ON strip_t3(c.link_id) = s.id")
    joined_data.createOrReplaceTempView("joined_data")
    # joined_data.show()

    # Task 9
    # dataframe_task9 = spark.sql("SELECT * FROM joined_data WHERE body NOT LIKE '&gt;%' AND body NOT LIKE '%/s%'")
    print('task 9 started')
    dataframe_task9 = spark.sql("SELECT created_time, post_title, com_state, parser(body) as parsed, comment_id, submission_id, c_score, s_score FROM joined_data WHERE body NOT LIKE '&gt;%' AND body NOT LIKE '%/s%'")
    dataframe_task9.createOrReplaceTempView("dataframe_task9")
    # dataframe_task9.show()

    cv_result = model.transform(dataframe_task9)
    pos_model = CrossValidatorModel.load('project2/pos.model')
    neg_model = CrossValidatorModel.load('project2/neg.model')

    pos = pos_model.transform(cv_result)
    pos.createOrReplaceTempView('pos')
    def posProbUDF(z):
        if z[1] > 0.2:
            return 1
        else: 
            return 0

    def negProbUDF(z):
        if z[1] > 0.25:
            return 1
        else:
            return 0

    posProb = udf(posProbUDF, IntegerType())
    negProb = udf(negProbUDF, IntegerType())

    # sqlContext.registerFunction("posProbUDF", lambda z: parse(z), IntegerType())
    # sqlContext.registerFunction("negProbUDF", lambda z: parse(z), IntegerType())
    # pos = spark.sql("SELECT com_state, vectors, submission_id, created_time, s_score, c_score, rawPrediction as pos_rawPrediction, posProbUDF(probability) as pos_probability, prediction as pos_prediction FROM pos")
    # pos.createOrReplaceTempView('pos')
    pos = pos.select(col('com_state'), col('vectors'), col('submission_id'), col('created_time'), col('s_score'), col('c_score'), posProb("probability").alias("pos_probability"), col("prediction").alias('pos_pred'))
    all_results = neg_model.transform(pos)
    total_result = all_results.select(col('com_state'), col('vectors'), col('submission_id'), col('created_time'), col('s_score'), col('pos_probability'), col('c_score'), negProb("probability").alias("neg_probability"), col("prediction"))

    # Task 10
    def getState(input_flair):
        states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
        if str(input_flair) in states:
            return str(input_flair)
        return 'not_state'
    get_state = udf(getState, StringType())
    total_result = total_result.select('*', get_state('com_state').alias('state'))
    total_result.createOrReplaceTempView('final_results')

    # parsedTable = spark.sql("SELECT id, body, Trump, parser(body) as parsed FROM sqlDF")
    
    query_1 = spark.sql("SELECT submission_id, AVG(pos_probability) as pos_prob, AVG(neg_probability) as neg_prob FROM final_results GROUP BY submission_id")
    print('q1')
    query_2 = spark.sql("SELECT date(from_unixtime(created_time)) as date, AVG(pos_probability) as pos_prob, AVG(neg_probability) as neg_prob FROM final_results GROUP BY date")
    print('q2')
    query_3 = spark.sql("SELECT state, AVG(pos_probability) as pos_prob, AVG(neg_probability) as neg_prob FROM final_results WHERE state !='not_state' GROUP BY state")
    # print('q3')
    query_4c = spark.sql("SELECT c_score, AVG(pos_probability) as pos_prob, AVG(neg_probability) as neg_prob  FROM final_results GROUP BY c_score")
    query_4s = spark.sql("SELECT s_score, AVG(pos_probability) as pos_prob, AVG(neg_probability) as neg_prob FROM final_results GROUP BY s_score")
    
    query_1.toPandas().to_csv("query_1.csv")
    query_2.toPandas().to_csv("query_2.csv")
    query_3.toPandas().to_csv("query_3.csv")
    query_4c.toPandas().to_csv("query_4c.csv")
    query_4s.toPandas().to_csv("query_4s.csv")


if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    # sc   = SparkContext(conf=conf)
    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)
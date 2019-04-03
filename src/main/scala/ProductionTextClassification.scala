import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.feature._
import org.apache.spark.ml.mleap.SparkUtil
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import resource._

//Naive bayes classfication + hashingTF
//Test set accuracy = 0.9507696051590923 => 1000000 features
//Test set accuracy = 0.9578104399628888 => 500000 features
//Test set accuracy = 0.9549639760654537 => 100000 features
//Test set accuracy = 0.9477678191392331 => 50000 features
//Test set accuracy = 0.8809176098332725 => 5000 features
//TODO: reduce features !
object ProductionTextClassification extends App with SparkSupport {
  override def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("Spark Word Count")
      .getOrCreate()

    import spark.implicits._

    val df = spark.read.json("./dataset/*")

    val ds = df.as[LeanItem3]

    println(s"Amount of catyegories in cat_0: ${ds.groupByKey(x => x.cat_0).keys.count()}")
    
    val labelIndexer = new StringIndexer()
      .setInputCol("cat_0")
      .setOutputCol("indexed_category")
      .fit(ds)

    val tokenizer = new RegexTokenizer()
      .setInputCol("title")
      .setOutputCol("words")

    StopWordsRemover.loadDefaultStopWords("spanish")

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered_words")

    import org.apache.spark.ml.feature.HashingTF

    val hashingTF = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("features")
      .setVocabSize(1000000)
      .setMinDF(20)

    val processPipeline = new Pipeline().setStages(Array(tokenizer, remover, hashingTF)).fit(ds)

    val trainPipeline = new Pipeline().setStages(Array(labelIndexer, processPipeline)).fit(ds)

    val dsProcessed = trainPipeline.transform(ds)

    val naiveBayesModel = new NaiveBayes().setFeaturesCol(hashingTF.getOutputCol).setLabelCol(labelIndexer.getOutputCol).fit(dsProcessed)

    val converter = new IndexToString()
      .setInputCol(naiveBayesModel.getPredictionCol)
      .setOutputCol("category")
      .setLabels(labelIndexer.labels)

    val predictionPipeline = SparkUtil.createPipelineModel(uid = "" , Array(processPipeline, naiveBayesModel, converter))

//    val predictionPipeline = new Pipeline().setStages(Array(processPipeline, naiveBayesModel, converter)).fit(ds)
//    predictionPipeline.transform(ds).select("title" ,"category", "probability").show(10, false)

    val sbc = SparkBundleContext()
    //Must use full path
    for(bf <- managed(BundleFile("jar:file:/home/martin/dev/projects/SparkML/bayes-mleap-export-1M-Features.zip"))) {
      predictionPipeline.writeBundle.save(bf)(sbc).get
    }
  }
}






//    val naiveBayesModel = new NaiveBayes().setFeaturesCol("features").setLabelCol("label").fit(trainDF)

//    //naiveBayesModel.transform(hashed).show()
//    val predictions = naiveBayesModel.transform(testDF)
//    //predictions.map{ case Row(t: String,l,c: String,w,f,rp,pr,p: Double) => (t, c, categoriesByIndex(p.toInt)(0)._1)}.show(100, false)
//
//
//    // Select (prediction, true label) and compute test error
//    val evaluator = new MulticlassClassificationEvaluator()
//      .setLabelCol("label")
//      .setPredictionCol("prediction")
//      .setMetricName("accuracy")
//    val accuracy = evaluator.evaluate(predictions)
//    println("Test set accuracy = " + accuracy)


//    naiveBayesModel.save("./myNaiveBayesModel-1000000features")
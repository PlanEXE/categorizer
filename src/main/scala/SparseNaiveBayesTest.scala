//import org.apache.spark.sql.SparkSession
//import org.apache.spark.ml.feature.RegexTokenizer
//import org.apache.spark.ml.feature.Word2Vec
//import org.apache.spark.ml.linalg.Vector
//import org.apache.spark.sql.Row
//import org.apache.spark.ml.Pipeline
//import org.apache.spark.ml.classification.NaiveBayes
//
//case class Item(
//                 title: String,
//                 description: String,
//                 price: Double,
//                 category_0: String,
//                 category_1: String,
//                 category_2: String
//               )
//
//object SparseNaiveBayesTest extends App {
//  override def main(args: Array[String]): Unit = {
//    val spark = SparkSession.builder
//      .master("local[*]")
//      .appName("Spark Word Count")
//      .getOrCreate()
//
//    val df = spark.read.json("/home/martin/dev/projects/scripts/item_desc_price_cat0_cat1_cat2.json")
//
//    val tokenizer = new RegexTokenizer()
//      .setInputCol("title")
//      .setOutputCol("words")
//
//    // Learn a mapping from words to Vectors.
//    val word2Vec = new Word2Vec()
//      .setInputCol("words")
//      .setOutputCol("result")
//      .setWindowSize(5)
//      .setVectorSize(500) //CUIDADO CON LA MEMORIA !!
//      .setMinCount(0)
//
//
//    val pipeline = new Pipeline().setStages(Array(tokenizer, word2Vec))
//
//    val word2VecModel = word2Vec.fit(tokenizer.transform(df))
//
//
//    val splits = word2VecModel.transform(df).randomSplit(Array(0.9, 0.1))
//    val training = splits(0)
//    val test = splits(1)
//
//    val numTraining = training.count()
//    val numTest = test.count()
//
//    println(s"numTraining = $numTraining, numTest = $numTest.")
//
//    val naiveBayesModel = new NaiveBayes().setLabelCol("category_0").fit(training)
//
//
//    naiveBayesModel.
//    val prediction = naiveBayesModel.predict(test.map(_.features))
//    val predictionAndLabel = prediction.zip(test.map(_.label))
//    val accuracy = predictionAndLabel.filter(x => x._1 == x._2).count().toDouble / numTest
//
//    println(s"Test accuracy = $accuracy.")
//
//    spark.stop()
//  }
//}
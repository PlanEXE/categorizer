import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.classification.{DecisionTreeClassifier, MultilayerPerceptronClassifier, NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.mleap.SparkUtil
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}
import resource._
import org.apache.spark.sql.functions.{avg, max, mean, min, stddev, stddev_pop}

//Naive bayes classfication + hashingTF
//Test set accuracy = 0.9507696051590923 => 1000000 features
//Test set accuracy = 0.9578104399628888 => 500000 features
//Test set accuracy = 0.9549639760654537 => 100000 features
//Test set accuracy = 0.9477678191392331 => 50000 features
//Test set accuracy = 0.8809176098332725 => 5000 features
//TODO: reduce features !

//Test set accuracy = 0.7897178702471406 => 1M Features - CAT_0
//Test set accuracy = 0.6884807772179786 - CAT_1 (Sin separar)

object NaiveBayesCellhpones extends App with SparkSupport {
  override def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("Spark Word Count")
      .config("spark.driver.maxResultSize", "10g")
      .getOrCreate()

    import spark.implicits._

    val ds = spark.read.json("./cellphones/*").as[LeanItem3]
//    val ds2 = ds.groupBy("cat_1").agg(
//      stddev("price"),
//      mean("price"),
//      max("price"),
//      min("price"),
//      avg("price"),
//     stddev_pop("price")
//    )
//    ds2.show(50, false)


    /**
      * compute percentile from an unsorted Spark RDD
      * @param data: input data set of Long integers
      * @param tile: percentile to compute (eg. 85 percentile)
      * @return value of input data at the specified percentile
      */
    def computePercentile(data: RDD[Double], tile: Double): Double = {
      // NIST method; data to be sorted in ascending order
      val r = data.sortBy(x => x)
      val c = r.count()
      if (c == 1) r.first()
      else {
        val n = (tile / 100d) * (c + 1d)
        val k = math.floor(n).toLong
        val d = n - k
        if (k <= 0) r.first()
        else {
          val index = r.zipWithIndex().map(_.swap)
          val last = c
          if (k >= c) {
            index.lookup(last - 1).head
          } else {
            index.lookup(k - 1).head + d * (index.lookup(k).head - index.lookup(k - 1).head)
          }
        }
      }
    }

//    ds.groupByKey(_.cat_1).flatMapGroups((g, i) => {
//      val p85 = computePercentile(i.flatMap(_.price), 85)
//      val p15 = computePercentile(ds.rdd.filter(_.cat_1 == g).flatMap(_.price), 15)
//
//    })

//    val dsFiltered = ds

    val dsFiltered = ds.groupByKey(_.cat_1).keys.collect().map(g => {
      val filtered = ds.filter(_.cat_1 == g)
      val priced = filtered.rdd.flatMap(_.price)
      val p85 = computePercentile(priced, 90)
      val p15 = computePercentile(priced, 10)
      println(g)
      println(p15)
      println(p85)

      filtered.filter(_.price.exists(x => x <= p85 && x >= p15))
    }).reduce((x,y) => x.union(y))



    def savePipeline(name: String, transoformers: Transformer*): Unit = {
      val sbc = SparkBundleContext()
      for(bf <- managed(BundleFile(s"jar:file:/home/martin/dev/wolla/SparkML/$name.zip"))) {
        SparkUtil.createPipelineModel(transoformers.toArray).writeBundle.save(bf)(sbc).get
      }
    }

//    val ds2 = ds.groupBy("cat_1").agg(stddev("price"))


    //--------------PRE-PROCESS-------------

    val tokenizer = new Tokenizer()
      .setInputCol("title")
      .setOutputCol("words")

    StopWordsRemover.loadDefaultStopWords("spanish")

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("wordsFiltered")

    val ngram2 = new NGram()
      .setN(2)
      .setInputCol(remover.getOutputCol)
      .setOutputCol("ngrams2")

//    val vocabSize = 1500
    val vocabSize = 1000

    val cv1 = new CountVectorizer()
//    val cv = new HashingTF()
      .setBinary(true)
      .setMinDF(20)
      .setVocabSize(vocabSize)
//      .setNumFeatures(50)
      .setInputCol(remover.getOutputCol)
      .setOutputCol("features1")

    val cv2 = new CountVectorizer()
      //    val cv = new HashingTF()
      .setBinary(true)
      .setMinDF(20)
      .setVocabSize(vocabSize)
      //      .setNumFeatures(50)
      .setInputCol(ngram2.getOutputCol)
      .setOutputCol("features2")

    val va = new VectorAssembler()
      .setInputCols(Array("features1", "features2"))
      .setOutputCol("features")

    val preProcessPipeline = new Pipeline().setStages(Array(tokenizer, remover, ngram2, cv1, cv2, va)).fit(dsFiltered)

    //    savePipeline("preProcess", preProcessPipeline)

    //--------------PRE-PROCESS-------------


    val preProcessedItem = preProcessPipeline.transform(dsFiltered)

    val labelIndexer = new StringIndexer()
      .setInputCol("cat_1")
      .setOutputCol("label")
      .fit(preProcessedItem)

    val converter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("categories")
      .setLabels(labelIndexer.labels)

    val labeledItems = labelIndexer.transform(preProcessPipeline.transform(dsFiltered)).as[LeanItem3]

    val Array(trainDF, testDF) = labeledItems.randomSplit(Array(0.9, 0.1))

    val model = new NaiveBayes().setModelType("bernoulli").setFeaturesCol("features").setLabelCol("label").fit(trainDF)
//    val model = new DecisionTreeClassifier().setMaxDepth(30).setFeaturesCol("features").setLabelCol("label").fit(trainDF)
//    val model = new MultilayerPerceptronClassifier().setLayers(Array(vocabSize,250,2)).setFeaturesCol("features").setLabelCol("label").fit(trainDF)

    val predictions = model.transform(testDF)

    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Test set accuracy = " + accuracy)

    converter.transform(predictions).show(200, false)



    ds.groupByKey(_.cat_1).keys.collect().foreach(g => {

      val predictions = model.transform(testDF.filter(_.cat_1 == g))

      // Select (prediction, true label) and compute test error
      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")
      val accuracy = evaluator.evaluate(predictions)
      println(s"Test set accuracy for $g = $accuracy" )

      converter.transform(predictions).show(200, false)
    })


    val manualData = Seq(
      LeanItem3("Iphone 6S", "", "", "", None),
      LeanItem3("Iphone 6S + funda", "", "", "", None),
      LeanItem3("Samsung Galaxy S6", "", "", "", None),
      LeanItem3("Samsung Galaxy S6 + funda", "", "", "", None),
      LeanItem3("Iphone 4", "", "", "", None),
      LeanItem3("Funda unica para samsung Galaxy S6", "", "", "", None),
      LeanItem3("Film protector", "", "", "", None),
      LeanItem3("Pantalla Samsung S6", "", "", "", None),
      LeanItem3("Film protector", "", "", "", None),
      LeanItem3("Display Iphone 7", "", "", "", None),
      LeanItem3("Cargador Samsung", "", "", "", None),
      LeanItem3("Vidrio Templado Samsung iphone motorola", "", "", "", None),
      LeanItem3("Vidrio Templado barato antibalas", "", "", "", None),
      LeanItem3("Film protector ifone", "", "", "", None),
      LeanItem3("Celular barato", "", "", "", None),
      LeanItem3("galaxy j5", "", "", "", None),
      LeanItem3("lg g6", "", "", "", None),
      LeanItem3("blackberry curve", "", "", "", None),
      LeanItem3("motorola g4", "", "", "", None),
      LeanItem3("cable usb type C", "", "", "", None),
      LeanItem3("Cargador de pared", "", "", "", None)
    ).toDS()

    val manualTest = preProcessPipeline.transform(manualData).as[LeanItem3]

    val predictions2 = model.transform(manualTest)
    converter.transform(predictions2).show(200, false)
  }
}
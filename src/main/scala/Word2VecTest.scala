import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.ml.Pipeline

object Word2VecTest extends App {
  override def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .master("local[*]")
      .appName("Spark Word Count")
      .getOrCreate()

    val df = spark.read.json("/home/martin/dev/projects/scripts/item_desc_price_cat0_cat1_cat2.json")

    val tokenizer = new RegexTokenizer()
      .setInputCol("title")
      .setOutputCol("words")

    // Learn a mapping from words to Vectors.
    val word2Vec = new Word2Vec()
      .setInputCol("words")
      .setOutputCol("result")
      .setWindowSize(3)
      .setVectorSize(5) //CUIDADO CON LA MEMORIA !!
      .setMinCount(0)


    val pipeline = new Pipeline().setStages(Array(tokenizer, word2Vec))

    val model = word2Vec.fit(tokenizer.transform(df))
//    val model = pipeline.fit(df)

//    val result = model.transform(df)

    //result.take(50).foreach { case Row(_, _, _, _, _, _ ,words: Seq[_], features: Vector) => println(s"Text: [${words.mkString(", ")}] => \nVector: $features\n") }

    model.transform(tokenizer.transform(df)).select("result").show(false)

//    df.take(20).map { case Row(_, _, _, _, _, title: String) => {
//      val w = title.split(" ").head.toLowerCase
//      (w, model.findSynonyms(w, 8))
//    }}
//      .foreach{case (w, df) => println(s"$w: [${df.collect().mkString(", ")}]")}
  }
}
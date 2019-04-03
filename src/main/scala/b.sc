val txt = scala.io.Source.fromURL("https://s3.amazonaws.com/nlp.utils/cat_tree.json").mkString
val r = "\"id\":\"(\\w+)\",\"name\":\"(\\w+)\"".r
val catMap = r.findAllMatchIn(txt).map(x => x.subgroups(0) -> x.subgroups(1)).toMap

catMap.size
case class Item(title: String, description: String, price: Double, category_0: String, category_1: String, category_2: String)
case class LeanItem2(title: String, category: String)
case class LeanItem3(title: String, cat_0: String, cat_1: String, cat_2: String, price: Option[Double])
case class LeanItem(title: String, category_0: String)
case class LabeledItem(title: String, label: Double, category_0: String)
case class TxtClassItem(title: String, description: String, price: Double, category_0: String, category_1: String, category_2: String, words: Array[String], features: Array[Double], indexedLabel: Long, indexedFeatures: Array[Double])
case class CatTree(id: String, name: String, children_categories: Option[Seq[CatTree]], total_items_in_this_category: Long, parent: Option[CatTree] = None)
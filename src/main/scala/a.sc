import java.text.Normalizer

import scala.collection.parallel.{ForkJoinTaskSupport}

val rInches = "(?<=^|\\s|\\(|\\[|\\:)(\\d+)('|''|\")($|\\s|\\.$|\\.\\s|%|!|°|\\)|\\])"
val rMm = "(?<=^|\\s|x|\\(|\\[|\\:)(\\d+) ?(mm|milimetro|milimetros)($|\\s|\\.$|\\.\\s|%|!|°|\\)|\\])" //Puede empezar con x ej: 20x40cm -> 20x40 centimetros
val rCm = "(?<=^|\\s|x|\\(|\\[|\\:)(\\d+) ?(cm|centimetros|centimetro)($|\\s|\\.$|\\.\\s|%|!|°|\\)|\\])"
val rCM = "(?<=^|\\s|x|\\(|\\[|\\:)(\\d+) ?(m|metro|metros)($|\\s|\\.$|\\.\\s|%|!|°|\\)|\\])"
val rLts = "(?<=^|\\s|\\(|\\[|\\:)(\\d+) ?(lts|l|litros|litro)($|\\s|\\.$|\\.\\s|%|!|°|\\)|\\])"
val rMLts = "(?<=^|\\s|x|\\(|\\[|\\:)(\\d+) ?(ml|mililitros|mililitro)($|\\s|\\.$|\\.\\s|%|!|°|\\)|\\])"
val rGrams = "(?<=^|\\s|\\(|\\[|\\:)(\\d+) ?(gr|grs|gramos|gramo)($|\\s|\\.$|\\.\\s|%|!|°|\\)|\\])" //No usamos la G por 4g por ejemplo
val rKilo = "(?<=^|\\s|\\(|\\[|\\:)(\\d+) ?(kilo|kilos)($|\\s|\\.$|\\.\\s|%|!|°|\\)|\\])" //No usamos la K por Tv 4K por ejemplo
val rWatts = "(?<=^|\\s|\\(|\\[|\\:)(\\d+) ?(w|watt|watts)($|\\s|\\.$|\\.\\s|%|!|°|\\)|\\])"
val rVolts = "(?<=^|\\s|\\(|\\[|\\:)(\\d+) ?(v|volt|volts)($|\\s|\\.$|\\.\\s|%|!|°|\\)|\\])"
val rNumbers = "(?<=^|\\s|\\(|\\[|\\:)(\\d+|\\d+\\.\\d+|\\d+,\\d+)(?=$|\\s|\\.$|\\.\\s|,\\s|%|!|°|\\)|\\])"
val rNumbers2 = "(?<=^|\\s|\\(|\\[|\\:|,|\\.)(\\d+)(?=$|\\s|\\.$|\\.\\s|,|%|!|°|\\)|\\])" //ej: 1,2,3,4
val rNumbers3 = "(?<=^|\\s|\\(|\\[|\\:)(\\d+x\\d+)(?=$|\\s|\\.$|\\.\\s|,\\s|%|!|°|\\)|\\])" //Soluciona los 20x30
val rNumbers4 = "(?<=^|\\s|\\(|\\[|\\:)(\\d+/\\d+)(?=$|\\s|\\.$|\\.\\s|,\\s|%|!|°|\\)|\\])" //Soluciona los 20/30
val rAmpersand = "&"
val rPercentage = "%"
val rPunctuation = "[^\\w\\s\\+'\"/]"
val rSpaces = "\\s+"

val title = " 3/4 127. & 100%. 50° (20) 20x70cm samsung j7 prime 10metros 15w 1,3,4,5,6,7 4grs 4 grs 5 grs. de caja, con - ~ sin ! 14 15 16 27mm 27mm 1l 30l 3l tv + de 26\" 27'' 28\" / sin c/ interes 80"

val newTitle = Normalizer.normalize(title.toLowerCase.trim, Normalizer.Form.NFD).replaceAll("[^\\p{ASCII}]", "")
  .replaceAll(rInches, "$1 pulgadas$3")
  .replaceAll(rMm, "$1 milimetros$3")
  .replaceAll(rCm, "$1 centimetros$3")
  .replaceAll(rCM, "$1 metros$3")
  .replaceAll(rLts, "$1 litros$3")
  .replaceAll(rMLts, "$1 mililitros$3")
  .replaceAll(rGrams, "$1 gramos$3")
  .replaceAll(rKilo, "$1 kilogramos$3")
  .replaceAll(rWatts, "$1 watts$3")
  .replaceAll(rVolts, "$1 volts$3")
  .replaceAll(rNumbers4, "NUMBER / NUMBER")
  .replaceAll(rNumbers3, "NUMBER x NUMBER")
  .replaceAll(rNumbers2, "NUMBER")
  .replaceAll(rNumbers, "NUMBER")
  .replaceAll(rAmpersand, " y ")
  .replaceAll(rPercentage, " porciento ")
  .replaceAll(rPunctuation, " ")
  .replaceAll(rSpaces, " ")

val pc  = (1 to 10).par
pc.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(2))
pc.foreach(x => {
  Thread.sleep(1000)
  println(System.currentTimeMillis())
} )
#!/usr/bin/env amm

import ammonite.ops._
import scala.collection.immutable.StringOps



class Bin2D(lang: String, dim: List[Int]) {
  object Statistics {
    def average(lst: List[Double]): Double = lst.sum / lst.size.toDouble
    def median[A](lst: List[A]): A = lst.drop(lst.size/2).head
    def medianNoNull(lst: List[Int]): Int = {
      val newl = lst.sorted.filter(_ != 0)
      newl.drop(newl.size/2).head
    }

    def std(data: List[Double]): Double = {
      val n = data.size.toDouble
      val xB = average(data)
      Math.sqrt( data.map( xi => Math.pow((xi - xB), 2) ).sum / (n - 1.0) )
    }

    def corr(x: List[Double], y: List[Double]): Double = {
      val xB = average(x)
      val yB = average(y)
      val stdX = std(x)
      val stdY = std(y)

      val zx = x.map { xi => (xi - xB) / stdX }
      val zy = y.map { yi => (yi - yB) / stdY }

      zx.zip(zy).map { case (zxi: Double, zyi: Double) => zxi * zyi }.sum / (x.size.toDouble - 1.0)
    }
  }

  private var _values: List[List[Double]] =
    List.fill(dim.size)(List.fill(dim.size)(0))

  private val data =
    (read(pwd/"out"/s"${lang.toLowerCase.trim}_star_fork"): StringOps)
    .lines
    .drop(1) // header
    .map(_.split(' ').toList.filter(_ != "").map(_.toInt))
    .toList

  val stars: List[Int] = data.map(_(0))
  val forks: List[Int] = data.map(_(1))
  val total: Int = data.size


  data.foreach { case List(s,f) => update(s,f) }

  def values: List[List[Double]] = _values

  def min: Double = values.map(_.min).min

  def max: Double = values.map(_.max).max

  private def update(i1: Int, i2: Int): Unit = {
    val in1 = index(dim, i1)
    val in2 = index(dim, i2)

    _values = _values.patch(in1, {
      Seq(
        _values(in1).patch(in2, Seq(_values(in1)(in2) + 1), 1)
      )
    }, 1)
  }

  def normalize(min: Double, max: Double): Unit = {
    def norm(v: Double): Double =
      if (v == 0.0) 0.0
      else (Math.log((v - min))/(Math.log(max-min)))
    _values = _values.map(_.map(norm))
  }

  private def pround(d: Double, prec: Double): Double =
    Math.round(d * prec) / prec

  private def index(targets: List[Int], i: Int): Int = {
    targets.zipWithIndex.find { case (x, _) => i <= x } match {
      case None => targets.size - 1
      case Some((_, here)) => here
    }
  }

  def asPercentage(): Unit =
    _values = _values.map(lst => lst.map(x => (x.toDouble / total.toDouble) * 10000.0))

  def format(asInt: Boolean = false): String =
    (if (asInt) lang + "_label"
    else
      lang + "_stars_avg = \"" + pround(Statistics.average(stars.map(_.toDouble)), 100.0) + "\"\n" +
      lang + "_forks_avg = \"" + pround(Statistics.average(forks.map(_.toDouble)), 100.0) + "\"\n" +
      lang + "_stars_med = \"" + Statistics.medianNoNull(stars) + "\"\n" +
      lang + "_forks_med = \"" + Statistics.medianNoNull(forks) + "\"\n" +
      lang + "_corr = \"" + Statistics.corr(stars.map(_.toDouble), forks.map(_.toDouble)) + "\"\n" +
      lang) +
    " = np.array([\n" + values.map(lst =>
        "    [" + lst
           .map(pround(_, 10000.0))
           .map { x =>
             if (x == 0.0 && asInt) "0"
             else if (x == 0.0) "0.0000001"
             else if(asInt) x.toInt.toString
             else x.toString
           }
           .mkString(",") + "]")
      .mkString(",\n") + "\n  ])"

  def show(asInt: Boolean = false): Unit =
      write.append(pwd/"out"/"generated_grid.py", format(asInt) + "\n")
}

@main
def main(): Unit = {

  write.over(pwd/"out"/"generated_grid.py", "import numpy as np\n")

  val template = List(0,1,2,3,4,5,6)

  val sparql  = new Bin2D("sparql",  template)
  val cypher  = new Bin2D("cypher",  template)
  val gremlin = new Bin2D("gremlin", template)
  val graphql = new Bin2D("graphql", template)

  sparql.show(true)
  cypher.show(true)
  gremlin.show(true)
  graphql.show(true)

  sparql.asPercentage()
  cypher.asPercentage()
  gremlin.asPercentage()
  graphql.asPercentage()

  val min = List(sparql.min, cypher.min, gremlin.min, graphql.min).min
  val max = List(sparql.max, cypher.max, gremlin.max, graphql.max).max

  sparql.normalize(min, max)
  cypher.normalize(min, max)
  gremlin.normalize(min, max)
  graphql.normalize(min, max)

  sparql.show()
  cypher.show()
  gremlin.show()
  graphql.show()
}

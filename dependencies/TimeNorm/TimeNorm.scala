import info.bethard.timenorm.TemporalExpressionParser
import info.bethard.timenorm.Temporal
import info.bethard.timenorm.TimeSpan
import info.bethard.timenorm._
import scala.util._
import java.io._
import scala.io.Source
import scala.collection.mutable.ListBuffer

object TimeNorm4Py {

	def main(args: Array[String]) {
		
		// parse args
		val anch = args(0)
		val expression_list = args(1)

		// temporal parser
		val parser = new TemporalExpressionParser(TemporalExpressionParser.getClass.getResource("/info/bethard/timenorm/en.grammar"))

		// convert anchor string into year, day, and month
		val anch_split = anch.split("-")
		val year = anch_split(0).toInt
		val month = anch_split(1).toInt
		val day = anch_split(2).toInt

		// create anchor object
		val anchor = TimeSpan.of(year, month, day)

		// evaluate each time expression with respect to the anchor
		val expressions = expression_list.split(",")
		for( expression <- expressions ){	
			val Success(temporal) = parser.parse(expression, anchor)
			val value = temporal.timeMLValue
			println(value)
		}
	}
}
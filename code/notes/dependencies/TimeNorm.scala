
import info.bethard.timenorm.TemporalExpressionParser;
import info.bethard.timenorm.TimeSpan;
import scala.util.Success;

object TimeNorm {

      // TODO: finish code to normalize some timex relative to doc creation time..
      def main(args: Array[String]) {

        // create a new parser (using the default English grammar)
        val parser = new TemporalExpressionParser

        // establish an anchor time
        val anchor = TimeSpan.of(2013, 1, 4)

        // parse an expression given an anchor time (here, assuming it succeeds)
        val Success(temporal) = parser.parse("two weeks ago", anchor)

        // get the TimeML value ("2012-W51") from the Temporal
        val value = temporal.timeMLValue

        println(value);

    }

}



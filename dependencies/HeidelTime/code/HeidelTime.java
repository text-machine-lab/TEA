
import java.util.Date;
import java.util.Calendar;
import de.unihd.dbs.heideltime.standalone.*;
import de.unihd.dbs.uima.annotator.heideltime.resources.Language;
import de.unihd.dbs.heideltime.standalone.exceptions.DocumentCreationTimeMissingException;

public class HeidelTime {

    public static void  main(String[] args) {
        HeidelTimeStandalone heidelTime = new HeidelTimeStandalone(Language.ENGLISH,
                                                                  DocumentType.COLLOQUIAL,
                                                                  OutputType.TIMEML,
                                                                  "conf/config.props",
                                                                  POSTagger.TREETAGGER, true);

        Calendar cal = Calendar.getInstance();
        cal.set(Calendar.YEAR, 1993);
        cal.set(Calendar.MONTH, Calendar.APRIL);
        cal.set(Calendar.DAY_OF_MONTH, 26);

        /*
            TODO: Set document type before processing.
            TODO: doctime in one example i found was YEAR-MONTH-DAY
        */

        try {

            System.out.println(heidelTime.process("He went to lunch at noon, yesterday.", cal.getTime()));

        } catch(DocumentCreationTimeMissingException e) {
            System.out.println("Missing Time");
        }

        return;

    }


}


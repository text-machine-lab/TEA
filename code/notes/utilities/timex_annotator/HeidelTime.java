package heidel;

import java.util.Date;
import java.util.Calendar;
import de.unihd.dbs.heideltime.standalone.*;
import de.unihd.dbs.uima.annotator.heideltime.resources.Language;
import de.unihd.dbs.heideltime.standalone.exceptions.DocumentCreationTimeMissingException;

public class HeidelTime {

    String CURR_DIR = new String(getClass().getProtectionDomain().getCodeSource().getLocation().toString().substring(5));
    String TEA_HOME = new String(CURR_DIR + "../../../../");

    HeidelTimeStandalone heidelTime = new HeidelTimeStandalone(Language.ENGLISH,
                                                               DocumentType.NEWS,
                                                               OutputType.TIMEML,
                                                               TEA_HOME + "/dependencies/HeidelTime/heideltime-kit/conf/config.props",
                                                               POSTagger.TREETAGGER, true);

    int[] _month_consts = {0, Calendar.JANUARY, Calendar.FEBRUARY, Calendar.MARCH, Calendar.APRIL, Calendar.MAY,
                           Calendar.JUNE, Calendar.JULY, Calendar.AUGUST, Calendar.SEPTEMBER, Calendar.OCTOBER,
                           Calendar.NOVEMBER, Calendar.DECEMBER};

    public String process(String document, int year, int month, int day) {

        //WARNING: I don't check for value correctness.
        int month_const = _month_consts[month];

        Calendar cal = Calendar.getInstance();
        cal.set(Calendar.YEAR, year);
        cal.set(Calendar.MONTH, month_const);
        cal.set(Calendar.DAY_OF_MONTH, day);

        try {
            return heidelTime.process(document, cal.getTime());
        } catch(DocumentCreationTimeMissingException e) {
            System.out.println("ERROR: Missing doc time.");
            System.exit(-1);
        }

        return "NULL";

    }

    public static void  main(String[] args) {

        HeidelTime test = new HeidelTime();

        System.out.println(test.process("He ran yesterday", 1993, 4, 26));

        return;

    }


}


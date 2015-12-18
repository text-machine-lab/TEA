
/*
 *  Copyright 2015 Rodrigo Agerri

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 */

package eus.ixa.ixa.pipe.nerc;
import eus.ixa.ixa.pipe.nerc.NerCLI;

import ixa.kaflib.KAFDocument;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;

import java.util.Properties;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import net.sourceforge.argparse4j.inf.Subparser;
import net.sourceforge.argparse4j.inf.Subparsers;
import opennlp.tools.cmdline.CmdLineUtil;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.util.TrainingParameters;

import org.jdom2.JDOMException;

import com.google.common.io.Files;

import eus.ixa.ixa.pipe.nerc.eval.CrossValidator;
import eus.ixa.ixa.pipe.nerc.eval.Evaluate;
import eus.ixa.ixa.pipe.nerc.train.FixedTrainer;
import eus.ixa.ixa.pipe.nerc.train.Flags;
import eus.ixa.ixa.pipe.nerc.train.InputOutputUtils;
import eus.ixa.ixa.pipe.nerc.train.Trainer;

import java.nio.charset.StandardCharsets;


public class Ner {

    NerCLI ner_cli = null;

    public Ner(){

        String[] cli_args = { "tag",
                              "-m",
                              System.getenv("TEA_PATH") + "/code/notes/NewsReader/models/nerc-models-1.5.0/nerc-models-1.5.0/en/conll03/en-91-19-conll03.bin" };

        ner_cli = new NerCLI();

        try {

            ner_cli.parseArgs(cli_args);

        } catch (IOException e) {

            System.out.println("ioerror");

        } catch (JDOMException e) {

            System.out.println("JDOM ERROR");

        }

    }

    public String tag(String naf_tagged_text) {

        InputStream is = new ByteArrayInputStream( naf_tagged_text.getBytes( StandardCharsets.UTF_8 ) );
        OutputStream os      = new ByteArrayOutputStream();

        try {

            this.ner_cli.annotate(is, os);

        } catch (IOException e) {

            System.out.println("ioexception when tagging string");
            System.exit(1);
        } catch (JDOMException e) {

            System.out.println("ioexception when tagging string");
            System.exit(1);
        }

        return os.toString();

    }

    public static void main(String[] args) {
    }

}



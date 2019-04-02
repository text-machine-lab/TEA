
/*
 * Copyright 2014 Rodrigo Agerri

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

package eus.ixa.ixa.pipe.pos;

import eus.ixa.ixa.pipe.pos.PosCLI;

import ixa.kaflib.KAFDocument;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.Properties;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import net.sourceforge.argparse4j.inf.Subparser;
import net.sourceforge.argparse4j.inf.Subparsers;
import opennlp.tools.cmdline.CmdLineUtil;
import opennlp.tools.postag.POSModel;
import opennlp.tools.util.TrainingParameters;

import org.jdom2.JDOMException;

import com.google.common.io.Files;

import eus.ixa.ixa.pipe.pos.eval.CrossValidator;
import eus.ixa.ixa.pipe.pos.eval.Evaluate;
import eus.ixa.ixa.pipe.pos.train.FixedTrainer;
import eus.ixa.ixa.pipe.pos.train.Flags;
import eus.ixa.ixa.pipe.pos.train.InputOutputUtils;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

import java.io.BufferedReader;

/**
* newsreader parts of speech component.
*/
public class Pos {

    PosCLI pos_CLI = null;

    public Pos() {

        String CURR_DIR = new String(getClass().getProtectionDomain().getCodeSource().getLocation().toString().substring(5));

        String TEA_HOME = new String(CURR_DIR + "../../../../../");

        String[] cli_args = { "tag",
                              "-m",
                              TEA_HOME  + "/dependencies/NewsReader/models/pos-models-1.4.0/en/en-maxent-100-c5-baseline-dict-penn.bin" };

        pos_CLI = new PosCLI();

        try {

            pos_CLI.parseArgs(cli_args);

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

            this.pos_CLI.annotate(is, os);

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

        Pos p = new Pos();

        String input = "";

        String line = null;

        try{

       BufferedReader br =
        new BufferedReader(new InputStreamReader(System.in));

            while((line=br.readLine())!=null){
             input += line;
        }

        System.out.println(input);

        } catch (IOException e){


        }

        System.out.println(p.tag(input));

    }

}


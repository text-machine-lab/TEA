
/*
 *Copyright 2015 Rodrigo Agerri

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

package eus.ixa.ixa.pipe.tok;

import eus.ixa.ixa.pipe.tok.TokCLI;

import ixa.kaflib.KAFDocument;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.StringReader;
import java.util.Properties;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.impl.Arguments;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import net.sourceforge.argparse4j.inf.Subparser;
import net.sourceforge.argparse4j.inf.Subparsers;

import org.jdom2.JDOMException;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;

/**
* newsreader tokenization component.
*/
public class Tok {

    TokCLI tok_CLI = null;

    public Tok() {

        String[] cli_args = { "tok",
                          "-l",
                          "en"};

        tok_CLI = new TokCLI();

        try {

            tok_CLI.parseArgs(cli_args);

        } catch (IOException e) {

            System.out.println("ioerror");

        } catch (JDOMException e) {

            System.out.println("JDOM ERROR");

        }


    }


    public String tokenize(String naf_tagged_text) {

        InputStream is = new ByteArrayInputStream( naf_tagged_text.getBytes( StandardCharsets.UTF_8 ) );
        OutputStream os      = new ByteArrayOutputStream();

        try {

            this.tok_CLI.annotate(is, os);

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

        Tok t = new Tok();

        System.out.println(t.tokenize("hello world"));

    }


}


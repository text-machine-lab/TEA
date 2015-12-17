
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
 * ixa-pipe-tok provides several configuration parameters:
 *
 * <ol>
 * <li>lang: choose language to create the lang attribute in KAF header.
 * <li>normalize: choose normalization method.
 * <li>outputFormat: choose between oneline, conll or NAF as output.
 * <li>untokenizable: print untokenizable (\uFFFD) characters.
 * <li>notok: take already tokenized text as input and create a KAFDocument with
 * it.
 * <li>inputkaf: take a NAF Document as input instead of plain text file.
 * <li>kafversion: specify the NAF version as parameter.
 * <li>hardParagraph: never break paragraphs.
 * <li>eval: input reference corpus to evaluate a tokenizer.
 * </ol>
 *
 *
 * @author ragerri
 * @version 2015-04-08
 */

class CLI {

  /**
   * Get dynamically the version of ixa-pipe-tok by looking at the MANIFEST
   * file.
   */
  private final String version = CLI.class.getPackage()
      .getImplementationVersion();
  /**
   * Get the commit of ixa-pipe-tok by looking at the MANIFEST file.
   */
  private final String commit = CLI.class.getPackage()
      .getSpecificationVersion();
  Namespace parsedArguments = null;

  // create Argument Parser
  ArgumentParser argParser = ArgumentParsers.newArgumentParser(
      "ixa-pipe-tok-" + version + ".jar").description(
      "ixa-pipe-tok-" + version
          + " is a multilingual tokenizer developed by the IXA NLP Group.\n");
  /**
   * Sub parser instance.
   */
  private final Subparsers subParsers = argParser.addSubparsers().help(
      "sub-command help");
  /**
   * The parser that manages the tagging sub-command.
   */
  private final Subparser annotateParser;

  public CLI() {
    annotateParser = subParsers.addParser("tok").help("Tagging CLI");
    loadAnnotateParameters();
  }


  /**
   * Parse the command interface parameters with the argParser.
   *
   * @param args
   *          the arguments passed through the CLI
   * @throws IOException
   *           exception if problems with the incoming data
   * @throws JDOMException
   *           a xml exception
   */
  public final void parseCLI(final String[] args) throws IOException,
      JDOMException {
    try {
      parsedArguments = argParser.parseArgs(args);
      System.err.println("CLI options: " + parsedArguments);
      if (args[0].equals("tok")) {
        annotate(System.in, System.out);
      }
    } catch (final ArgumentParserException e) {
      argParser.handleError(e);
      System.out.println("Run java -jar target/ixa-pipe-tok-" + version
          + ".jar tok -help for details");
      System.exit(1);
    }
  }

  public final void annotate(final InputStream inputStream,
      final OutputStream outputStream) throws IOException, JDOMException {
    final String outputFormat = parsedArguments.getString("outputFormat");
    final String normalize = parsedArguments.getString("normalize");
    final String lang = parsedArguments.getString("lang");
    final String untokenizable = parsedArguments.getString("untokenizable");
    final String kafVersion = parsedArguments.getString("kafversion");
    final Boolean inputKafRaw = false;// parsedArguments.getBoolean("inputkaf");
    final Boolean noTok = parsedArguments.getBoolean("notok");
    final String hardParagraph = parsedArguments.getString("hardParagraph");
    final Properties properties = setAnnotateProperties(lang, normalize, untokenizable, hardParagraph);


    BufferedReader breader = new BufferedReader(new InputStreamReader(
        inputStream, "UTF-8"));
    BufferedWriter bwriter = new BufferedWriter(new OutputStreamWriter(
        outputStream, "UTF-8"));

    KAFDocument kaf;

    kaf = new KAFDocument(lang, kafVersion);

      final Annotate annotator = new Annotate(breader, properties);

      if (outputFormat.equalsIgnoreCase("conll")) {

        if (parsedArguments.getBoolean("offsets")) {
          bwriter.write(annotator.tokenizeToCoNLL());
        } else {
          bwriter.write(annotator.tokenizeToCoNLLOffsets());
        }
      } else if (outputFormat.equalsIgnoreCase("oneline")) {

        bwriter.write(annotator.tokenizeToText());
      } else {

        final KAFDocument.LinguisticProcessor newLp = kaf
            .addLinguisticProcessor("text", "ixa-pipe-tok-" + lang, version
                + "-" + commit);
        newLp.setBeginTimestamp();
        annotator.tokenizeToKAF(kaf);
        newLp.setEndTimestamp();
        bwriter.write(kaf.toString());
      }
      breader.close();
    bwriter.close();
  }

  private void loadAnnotateParameters() {
    // specify language (for language dependent treatment of apostrophes)
    annotateParser
        .addArgument("-l", "--lang")
        .choices("de", "en", "es", "eu", "fr", "gl", "it", "nl")
        .required(true)
        .help(
            "It is REQUIRED to choose a language to perform annotation with ixa-pipe-tok.\n");
    annotateParser
        .addArgument("-n", "--normalize")
        .choices("alpino", "ancora", "ctag", "default", "ptb", "tiger",
            "tutpenn")
        .required(false)
        .setDefault("default")
        .help(
            "Set normalization method according to corpus; the default option does not escape "
                + "brackets or forward slashes. See README for more details.\n");
    annotateParser
        .addArgument("-u","--untokenizable")
        .choices("yes", "no")
        .setDefault("no")
        .required(false)
        .help("Print untokenizable characters.\n");
    annotateParser
        .addArgument("-o", "--outputFormat")
        .choices("conll", "oneline", "naf")
        .setDefault("naf")
        .required(false)
        .help(
            "Choose output format; it defaults to NAF.\n");
    annotateParser
        .addArgument("--offsets")
        .action(Arguments.storeFalse())
        .help(
            "Do not print offset and lenght information of tokens in CoNLL format.\n");
    annotateParser
        .addArgument("--inputkaf")
        .action(Arguments.storeTrue())
        .help(
            "Use this option if input is a KAF/NAF document with <raw> layer.\n");
    annotateParser
        .addArgument("--notok")
        .action(Arguments.storeTrue())
        .help(
            "Build a KAF document from an already tokenized sentence per line file.\n");
    annotateParser
        .addArgument("--hardParagraph")
        .choices("yes", "no")
        .setDefault("no")
        .required(false)
        .help("Do not segment paragraphs. Ever.\n");
    annotateParser.addArgument("--kafversion")
         .setDefault("v1.naf")
        .help("Set kaf document version.\n");
  }

  private Properties setAnnotateProperties(final String lang, final String normalize, final String untokenizable, final String hardParagraph) {
    final Properties annotateProperties = new Properties();
    annotateProperties.setProperty("language", lang);
    annotateProperties.setProperty("normalize", normalize);
    annotateProperties.setProperty("untokenizable", untokenizable);
    annotateProperties.setProperty("hardParagraph", hardParagraph);
    return annotateProperties;
  }

    public final void parseArgs(final String[] args) throws IOException, JDOMException {

        try {

            parsedArguments = argParser.parseArgs(args);

        } catch (ArgumentParserException e) {

            argParser.handleError(e);
            System.exit(1);

        }
      }

}

public class Tok {

    CLI tok_CLI = null;

    public Tok() {

        String[] cli_args = { "tok",
                          "-l",
                          "en"};

        tok_CLI = new CLI();

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


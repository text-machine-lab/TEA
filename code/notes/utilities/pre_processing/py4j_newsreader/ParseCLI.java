
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

package eus.ixa.ixa.pipe.parse;

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
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import net.sourceforge.argparse4j.inf.Subparser;
import net.sourceforge.argparse4j.inf.Subparsers;

import org.jdom2.JDOMException;

import com.google.common.io.Files;


import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;



/**
 * This is the main class of ixa-pipe-parse a constituent shift-reduce parser
 * based on on Apache OpenNLP project, which is based on (Ratnapharki 1999).
 * Provides extra support for Collins Head Finder useful for coreference
 * resolution and dependency conversion. It also takes the tokens of a gold
 * parse file for evaluation (e.g., sec 23 of Penn Treebank) and produces the
 * test parse file ready to be evaluated with EVALB.
 *
 * @author ragerri
 * @version 2014-04-18
 */

public class ParseCLI {


  Annotate annotator = null;

  /**
   * Get dynamically the version of ixa-pipe-parse by looking at the MANIFEST
   * file.
   */
  private final String version = CLI.class.getPackage()
      .getImplementationVersion();
  /**
   * Get the git commit of the ixa-pipe-parse compiled by looking at the
   * MANIFEST file.
   */
  private final String commit = CLI.class.getPackage()
      .getSpecificationVersion();

  Namespace parsedArguments = null;

  // create Argument Parser
  ArgumentParser argParser = ArgumentParsers.newArgumentParser(
      "ixa-pipe-parse-" + this.version + ".jar").description(
      "ixa-pipe-parse is a multilingual Constituent Parsing module "
          + "developed by IXA NLP Group.\n");
  /**
   * Sub parser instance.
   */
  private final Subparsers subParsers = this.argParser.addSubparsers().help(
      "sub-command help");
  /**
   * The parser that manages the tagging sub-command.
   */
  private final Subparser annotateParser;
  /**
   * The parser that manages the training sub-command.
   */
  private final Subparser trainParser;
  /**
   * The parser that manages the evaluation sub-command.
   */
  private final Subparser evalParser;

  /**
   * Construct a CLI object with the three sub-parsers to manage the command
   * line parameters.
   */
  public ParseCLI() {
    this.annotateParser = this.subParsers.addParser("parse")
        .help("Parsing CLI");
    loadAnnotateParameters();
    this.trainParser = this.subParsers.addParser("train").help("Training CLI");
    loadTrainingParameters();
    this.evalParser = this.subParsers.addParser("eval").help("Evaluation CLI");
    loadEvalParameters();
  }

  /**
   * Parse the command interface parameters with the argParser.
   *
   * @param args
   *          the arguments passed through the CLI
   * @throws IOException
   *           exception if problems with the incoming data
   * @throws JDOMException
   *           if xml formatting exception
   */
  public final void parseCLI(final String[] args) throws IOException,
      JDOMException {
    try {
      this.parsedArguments = this.argParser.parseArgs(args);
      System.err.println("CLI options: " + this.parsedArguments);
      if (args[0].equals("parse")) {
        annotate(System.in, System.out);
      } else if (args[0].equals("eval")) {
        eval();
      } else if (args[0].equals("train")) {
        train();
      }
    } catch (final ArgumentParserException e) {
      this.argParser.handleError(e);
      System.out.println("Run java -jar target/ixa-pipe-parse-" + this.version
          + ".jar" + " (parse|train|eval) -help for details");
      System.exit(1);
    }
  }

  public final void annotate(final InputStream inputStream,
      final OutputStream outputStream) throws IOException, JDOMException {

    final BufferedReader breader = new BufferedReader(new InputStreamReader(
        inputStream, "UTF-8"));
    final BufferedWriter bwriter = new BufferedWriter(new OutputStreamWriter(
        outputStream, "UTF-8"));
    final KAFDocument kaf = KAFDocument.createFromStream(breader);

            final String model = this.parsedArguments.getString("model");
            final String headFinderOption = this.parsedArguments
                .getString("headFinder");
            final String outputFormat = this.parsedArguments.getString("outputFormat");


    final KAFDocument.LinguisticProcessor newLp = kaf.addLinguisticProcessor(
        "constituency",
        "ixa-pipe-parse-" + Files.getNameWithoutExtension(model), this.version
            + "-" + this.commit);
    newLp.setBeginTimestamp();
    String kafToString = null;
    if (outputFormat.equalsIgnoreCase("oneline")) {
      kafToString = this.annotator.parseToOneline(kaf);
    } else {
      this.annotator.parseToKAF(kaf);
      newLp.setEndTimestamp();
      kafToString = kaf.toString();
    }
    bwriter.write(kafToString);
    bwriter.close();
    breader.close();
  }

  public final void train() {
    System.err.println("Not yet ready!");
  }

  public final void eval() throws IOException {

    final String lang = this.parsedArguments.getString("language");
    final String model = this.parsedArguments.getString("model");
    final String headFinderOption = this.parsedArguments
        .getString("headFinder");
    final Properties properties = setEvaluateProperties(model, lang,
        headFinderOption);
    final Annotate annotator = new Annotate(properties);
    // special option to process treebank files adding headword marks
    if (this.parsedArguments.getString("addHeads") != null) {
      final File inputTree = new File(
          this.parsedArguments.getString("addHeads"));
      annotator.processTreebankWithHeadWords(inputTree);
    } else if (this.parsedArguments.get("test") != null) {
      final File inputTree = new File(this.parsedArguments.getString("test"));
      annotator.parseForTesting(inputTree);
    }
  }

  public void loadAnnotateParameters() {
    this.annotateParser.addArgument("-m", "--model").required(true)
        .help("Choose parsing model.\n");
    this.annotateParser
        .addArgument("-l", "--language")
        .choices("en", "es")
        .required(false)
        .help(
            "Choose language; it defaults to the language value in incoming NAF file.\n");
    this.annotateParser.addArgument("-g", "--headFinder")
        .choices("collins", "sem", Flags.DEFAULT_HEADFINDER)
        .setDefault(Flags.DEFAULT_HEADFINDER).required(false)
        .help("Choose between Collins or Semantic HeadFinder.\n");
    this.annotateParser.addArgument("-o", "--outputFormat")
        .choices("oneline", "naf").setDefault(Flags.DEFAULT_OUTPUT_FORMAT)
        .required(false).help("Choose outputFormat; it defaults to NAF.\n");
  }

  /**
   * Create the main parameters available for training parse models.
   */
  private void loadTrainingParameters() {
  }

  private void loadEvalParameters() {

    this.evalParser.addArgument("-m", "--model").required(true)
        .help("Choose parsing model.\n");
    this.evalParser.addArgument("-l", "--language").choices("en", "es")
        .required(true).help("Choose language.\n");
    this.evalParser.addArgument("-g", "--headFinder")
        .choices("collins", "sem", Flags.DEFAULT_HEADFINDER)
        .setDefault(Flags.DEFAULT_HEADFINDER).required(false)
        .help("Choose between Collins or Semantic HeadFinder.\n");
    this.evalParser
        .addArgument("--addHeads")
        .help(
            "Takes a file or a directory as argument containing a parse tree in penn treebank (one line per sentence) format; this option requires --lang and --headFinder options.\n");
    this.evalParser
        .addArgument("--test")
        .help(
            "Takes a file as argument containing the tokenized text of a gold standard Penn Treebank file to process it; It produces a test file for its parseval evaluation with EVALB.\n");
  }

  private Properties setAnnotateProperties(final String model,
      final String language, final String headFinder) {
    final Properties annotateProperties = new Properties();
    annotateProperties.setProperty("model", model);
    annotateProperties.setProperty("language", language);
    annotateProperties.setProperty("headFinder", headFinder);
    return annotateProperties;
  }

  private Properties setEvaluateProperties(final String model,
      final String language, final String headFinder) {
    final Properties annotateProperties = new Properties();
    annotateProperties.setProperty("model", model);
    annotateProperties.setProperty("language", language);
    annotateProperties.setProperty("headFinder", headFinder);
    return annotateProperties;
  }

   public final void parseArgs(final String[] args) throws IOException, JDOMException {

       try {

           parsedArguments = argParser.parseArgs(args);

            final String model = this.parsedArguments.getString("model");
            final String headFinderOption = this.parsedArguments
                .getString("headFinder");
            final String outputFormat = this.parsedArguments.getString("outputFormat");
            final Properties properties = setAnnotateProperties(model, "en",
                headFinderOption);

            this.annotator = new Annotate(properties);

       } catch (ArgumentParserException e) {

           argParser.handleError(e);
           System.exit(1);

       }
     }


}


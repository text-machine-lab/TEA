
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
 * Main class of ixa-pipe-pos, the pos tagger of ixa-pipes
 * (ixa2.si.ehu.es/ixa-pipes). The annotate method is the main entry point.
 *
 * @author ragerri
 * @version 2014-11-30
 */

public class PosCLI {

    Annotate annotator = null;

  /**
   * Get dynamically the version of ixa-pipe-pos by looking at the MANIFEST
   * file.
   */
  private final String version = CLI.class.getPackage()
      .getImplementationVersion();
  /**
   * Get the git commit of the ixa-pipe-pos compiled by looking at the MANIFEST
   * file.
   */
  private final String commit = CLI.class.getPackage()
      .getSpecificationVersion();
  /**
   * The CLI arguments.
   */
  private Namespace parsedArguments = null;
  /**
   * The argument parser.
   */
  private final ArgumentParser argParser = ArgumentParsers.newArgumentParser(
      "ixa-pipe-pos-" + this.version + ".jar").description(
      "ixa-pipe-pos-" + this.version
          + " is a multilingual POS tagger developed by IXA NLP Group.\n");
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
   * The parser that manages the cross validation sub-command.
   */
  private final Subparser crossValidateParser;
  /**
   * Default beam size for decoding.
   */
  public static final String DEFAULT_BEAM_SIZE = "3";

  /**
   * Construct a CLI object with the three sub-parsers to manage the command
   * line parameters.
   */
  public PosCLI() {
    this.annotateParser = this.subParsers.addParser("tag").help("Tagging CLI");
    loadAnnotateParameters();
    this.trainParser = this.subParsers.addParser("train").help("Training CLI");
    loadTrainingParameters();
    this.evalParser = this.subParsers.addParser("eval").help("Evaluation CLI");
    loadEvalParameters();
    this.crossValidateParser = this.subParsers.addParser("cross").help(
        "Cross validation CLI");
    loadCrossValidateParameters();
  }

  public static void main(final String[] args) throws JDOMException,
      IOException {

    final CLI cmdLine = new CLI();
    cmdLine.parseCLI(args);
  }

  /**
   * Parse the command line options.
   *
   * @param args
   *          the arguments
   * @throws IOException
   *           if io error
   * @throws JDOMException
   *           if malformed XML
   */
  public final void parseCLI(final String[] args) throws IOException,
      JDOMException {
    try {
      this.parsedArguments = this.argParser.parseArgs(args);
      System.err.println("CLI options: " + this.parsedArguments);
      if (args[0].equals("tag")) {
        annotate(System.in, System.out);
      } else if (args[0].equals("eval")) {
        eval();
      } else if (args[0].equals("train")) {
      } else if (args[0].equals("cross")) {
        crossValidate();
      }
    } catch (final ArgumentParserException e) {
      this.argParser.handleError(e);
      System.out.println("Run java -jar target/ixa-pipe-pos-" + this.version
          + ".jar (tag|train|eval|cross) -help for details");
      System.exit(1);
    }
  }

  /**
   * Main entry point for annotation. Takes system.in as input and outputs
   * annotated text via system.out.
   *
   * @param inputStream
   *          the input stream
   * @param outputStream
   *          the output stream
   * @throws IOException
   *           the exception if not input is provided
   * @throws JDOMException
   *           if malformed XML
   */
  public final void annotate(final InputStream inputStream,
      final OutputStream outputStream) throws IOException, JDOMException {

    BufferedReader breader = new BufferedReader(new InputStreamReader(
    inputStream, "UTF-8"));
    BufferedWriter bwriter = new BufferedWriter(new OutputStreamWriter(
    outputStream, "UTF-8"));

    final KAFDocument kaf = KAFDocument.createFromStream(breader);
    final String model = this.parsedArguments.getString("model");

    if (this.parsedArguments.getBoolean("nokaf")) {
      bwriter.write(annotator.annotatePOSToCoNLL(kaf));
    } else {

      final KAFDocument.LinguisticProcessor newLp = kaf.addLinguisticProcessor(
          "terms", "ixa-pipe-pos-" + Files.getNameWithoutExtension(model),
          this.version + "-" + this.commit);
      newLp.setBeginTimestamp();
      this.annotator.annotatePOSToKAF(kaf);
      newLp.setEndTimestamp();
      bwriter.write(kaf.toString());
    }
    bwriter.close();
    breader.close();
  }

  /**
   * Generate the annotation parameter of the CLI.
   */
  private void loadAnnotateParameters() {
    this.annotateParser.addArgument("-m", "--model").required(true)
        .help("It is required to provide a model to perform POS tagging.");
    this.annotateParser.addArgument("-l", "--lang")
        .choices("en", "es", "gl", "it").required(false)
        .help("Choose a language to perform annotation with ixa-pipe-pos.");

    this.annotateParser.addArgument("--beamSize").required(false)
        .setDefault(DEFAULT_BEAM_SIZE)
        .help("Choose beam size for decoding, it defaults to 3.");
    this.annotateParser
        .addArgument("--nokaf")
        .action(Arguments.storeTrue())
        .help(
            "Do not print tokens in NAF format, but conll tabulated format.\n");
    this.annotateParser.addArgument("-mw", "--multiwords")
        .action(Arguments.storeTrue())
        .help("Use to detect and process multiwords.\n");
    this.annotateParser.addArgument("-d", "--dictag")
        .action(Arguments.storeTrue())
        .help("Post process POS tagger output with a monosemic dictionary.\n");
  }

  /**
   * Loads the parameters for the training CLI.
   */
  private final void loadTrainingParameters() {
    this.trainParser.addArgument("-p", "--params").required(true)
        .help("Load the training parameters file\n");
  }

  /**
   * Main entry point for evaluation.
   *
   * @throws IOException
   *           the io exception thrown if errors with paths are present
   */
  public final void eval() throws IOException {
    final String testFile = this.parsedArguments.getString("testSet");
    final String model = this.parsedArguments.getString("model");
    final String beamSize = this.parsedArguments.getString("beamSize");

    final Evaluate evaluator = new Evaluate(testFile, model, beamSize);
    if (this.parsedArguments.getString("evalReport") != null) {
      if (this.parsedArguments.getString("evalReport").equalsIgnoreCase(
          "detailed")) {
        evaluator.detailEvaluate();
      } else if (this.parsedArguments.getString("evalReport").equalsIgnoreCase(
          "error")) {
        evaluator.evalError();
      } else if (this.parsedArguments.getString("evalReport").equalsIgnoreCase(
          "brief")) {
        evaluator.evaluate();
      }
    } else {
      evaluator.evaluate();
    }
  }

  /**
   * Load the evaluation parameters of the CLI.
   */
  private final void loadEvalParameters() {
    this.evalParser.addArgument("-m", "--model").required(true)
        .help("Choose model");
    this.evalParser.addArgument("-t", "--testSet").required(true)
        .help("Input testset for evaluation");
    this.evalParser.addArgument("--evalReport").required(false)
        .choices("brief", "detailed", "error")
        .help("Choose type of evaluation report; defaults to brief");
    this.evalParser.addArgument("--beamSize").setDefault(DEFAULT_BEAM_SIZE)
        .type(Integer.class)
        .help("Choose beam size for evaluation: 1 is faster.");
  }

  /**
   * Main access to the cross validation.
   *
   * @throws IOException
   *           input output exception if problems with corpora
   */
  public final void crossValidate() throws IOException {

    final String paramFile = this.parsedArguments.getString("params");
    final TrainingParameters params = InputOutputUtils
        .loadTrainingParameters(paramFile);
    final CrossValidator crossValidator = new CrossValidator(params);
    crossValidator.crossValidate(params);
  }

  /**
   * Create the main parameters available for training NERC models.
   */
  private void loadCrossValidateParameters() {
    this.crossValidateParser.addArgument("-p", "--params").required(true)
        .help("Load the Cross validation parameters file\n");
  }

  /**
   * Set a Properties object with the CLI parameters for annotation.
   *
   * @param model
   *          the model parameter
   * @param language
   *          language parameter
   * @param beamSize
   *          the beamsize decoding
   * @param lemmatize
   *          the lemmatization method
   * @return the properties object
   */
  private Properties setAnnotateProperties(final String model,
      final String language, final String beamSize, final String multiwords,
      final String dictag) {
    final Properties annotateProperties = new Properties();
    annotateProperties.setProperty("model", model);
    annotateProperties.setProperty("language", language);
    annotateProperties.setProperty("beamSize", beamSize);
    annotateProperties.setProperty("multiwords", multiwords);
    annotateProperties.setProperty("dictag", dictag);
    return annotateProperties;
  }

public final void parseArgs(final String[] args) throws IOException, JDOMException {

    try {

        parsedArguments = argParser.parseArgs(args);

        final String model = this.parsedArguments.getString("model");
        final String beamSize = this.parsedArguments.getString("beamSize");
        final String multiwords = Boolean.toString(this.parsedArguments
            .getBoolean("multiwords"));
        final String dictag = Boolean.toString(this.parsedArguments
            .getBoolean("dictag"));

        // always english.
        final Properties properties = setAnnotateProperties(model, "en", beamSize,
            multiwords, dictag);

        this.annotator = new Annotate(properties);

    } catch (ArgumentParserException e) {

        argParser.handleError(e);
        System.exit(1);

    }
  }



}



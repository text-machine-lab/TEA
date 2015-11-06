package ixa.srl;

import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;

import se.lth.cs.srl.Parse;
import se.lth.cs.srl.SemanticRoleLabeler;
import se.lth.cs.srl.languages.Language;
import se.lth.cs.srl.options.CompletePipelineCMDLineOptions;
import se.lth.cs.srl.options.FullPipelineOptions;
import se.lth.cs.srl.pipeline.Pipeline;
import se.lth.cs.srl.pipeline.Reranker;
import se.lth.cs.srl.pipeline.Step;
import se.lth.cs.srl.preprocessor.Preprocessor;

public class PreProcess {

	private static final Pattern JARPATH_PATTERN_BEGIN = Pattern.compile("file:");
	private static final Pattern JARPATH_PATTERN_END = Pattern.compile("[^/]+jar!.+");

	public Preprocessor pp;
	public SemanticRoleLabeler srl;
	public CompletePipelineCMDLineOptions options;
	
	private void getCompletePipeline(FullPipelineOptions options,
			String option) throws ZipException, IOException,
			ClassNotFoundException {

		Preprocessor pp = Language.getLanguage().getPreprocessor(options);
		Parse.parseOptions = options.getParseOptions();

		if (!option.equals("only-deps")) {
			SemanticRoleLabeler srl;
			if (options.reranker) {
				srl = new Reranker(Parse.parseOptions);
			} else {
				ZipFile zipFile = new ZipFile(Parse.parseOptions.modelFile);
				if (Parse.parseOptions.skipPI) {
					srl = Pipeline.fromZipFile(zipFile, new Step[] { Step.pd,
							Step.ai, Step.ac });
				} else {
					srl = Pipeline.fromZipFile(zipFile);
				}
				zipFile.close();
			}
			this.pp = pp;
			this.srl = srl;
		} else {
			this.pp = pp;
		}
	}
	
	
	public PreProcess(String lang, String option)
			throws Exception {

		String jarpath = this.getClass().getResource("").getPath();
		Matcher matcher = JARPATH_PATTERN_BEGIN.matcher(jarpath);
		jarpath = matcher.replaceAll("");		
		matcher = JARPATH_PATTERN_END.matcher(jarpath);
		jarpath = matcher.replaceAll("");

		String[] models = new String[3];
		if (lang.equals("eng")) {
		    models[0] = jarpath + "/models/eng/CoNLL2009-ST-English-ALL.anna-3.3.parser.model";
		    models[1] = jarpath + "/models/eng/srl-eng.model";
		} else if (lang.equals("spa")) {
		    models[0] = jarpath + "/models/spa/CoNLL2009-ST-Spanish-ALL.anna-3.3.parser.model";
		    models[1] = jarpath + "/models/spa/srl-spa.model";
		    models[2] = jarpath + "/models/spa/CoNLL2009-ST-Spanish-ALL.anna-3.3.morphtagger.model";
		}


		String[] arguments = null;
		if (option.equals("only-deps")) {
			if (lang.equals("eng")) {
				arguments = new String[3];
				arguments[0] = lang;
				arguments[1] = "-parser";
				arguments[2] = models[0];
			} else if (lang.equals("spa")) {
				arguments = new String[5];
				arguments[0] = lang;
				arguments[1] = "-parser";
				arguments[2] = models[0];
				arguments[3] = "-morph";
				arguments[4] = models[2];
			}
		} else if (option.equals("only-srl")) {
			if (lang.equals("eng")) {
				arguments = new String[3];
				arguments[0] = lang;
				arguments[1] = "-srl";
				arguments[2] = models[1];
			} else if (lang.equals("spa")) {
				arguments = new String[5];
				arguments[0] = lang;
				arguments[1] = "-srl";
				arguments[2] = models[1];
				arguments[3] = "-morph";
				arguments[4] = models[2];
			}
		} else {
			if (lang.equals("eng")) {
				arguments = new String[5];
				arguments[0] = lang;
				arguments[1] = "-parser";
				arguments[2] = models[0];
				arguments[3] = "-srl";
				arguments[4] = models[1];
			} else if (lang.equals("spa")) {
				arguments = new String[7];
				arguments[0] = lang;
				arguments[1] = "-parser";
				arguments[2] = models[0];
				arguments[3] = "-srl";
				arguments[4] = models[1];
				arguments[5] = "-morph";
				arguments[6] = models[2];
			}
		}

		
		// String
		// error=FileExistenceVerifier.verifyCompletePipelineAllNecessaryModelFiles(options);
		// if(error!=null){
		// System.err.println(error);
		// System.err.println();
		// System.err.println("Aborting.");
		// System.exit(1);
		// }

		this.options = new CompletePipelineCMDLineOptions();
		this.options.parseCmdLineArgs(arguments);
		this.getCompletePipeline(this.options, option);
	}
}

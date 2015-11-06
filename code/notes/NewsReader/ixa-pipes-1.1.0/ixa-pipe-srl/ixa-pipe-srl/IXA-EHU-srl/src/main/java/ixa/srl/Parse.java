package ixa.srl;

import is2.data.SentenceData09;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Pattern;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import se.lth.cs.srl.corpus.Predicate;
import se.lth.cs.srl.corpus.Sentence;
import se.lth.cs.srl.corpus.Word;
import se.lth.cs.srl.options.CompletePipelineCMDLineOptions;

public class Parse {
	
	private static final Pattern WHITESPACE_PATTERN = Pattern.compile("\\s+");

	public PreProcess preprocess;
	
	public Parse(PreProcess preprocess) {
		this.preprocess = preprocess;
	}
	
	public Document ParseDocument(List<String> annotation, String lang, String option)
			throws Exception {

		Document doc = null;
		doc = parseCoNLL09(this.preprocess.options, option, this.preprocess, annotation);

		return doc;
	}

	
	private static Document parseCoNLL09(
			CompletePipelineCMDLineOptions options, String option,
			PreProcess preprocess, List<String> in) throws IOException,
			Exception {
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		DocumentBuilder db = dbf.newDocumentBuilder();

		Document doc = db.newDocument();

		Element ROOT = doc.createElement("ROOT");
		doc.appendChild(ROOT);
		Element DEPS = doc.createElement("DEPS");
		ROOT.appendChild(DEPS);
		Element SRL = doc.createElement("SRL");
		ROOT.appendChild(SRL);

		List<String> forms = new ArrayList<String>();
		forms.add("<root>");
		List<Boolean> isPred = new ArrayList<Boolean>();
		isPred.add(false);
		String str;
		int senCount = 0;

		int sidx = 1;
		Iterator<String> iter = in.iterator();
		while (iter.hasNext()) {
			str = iter.next();
			if (str.trim().equals("")) {
				Sentence s;
				s = parse(preprocess, forms, option);
				String tag;
				for (int i = 1; i < s.size(); ++i) {
					Word w = s.get(i);

					Element d = doc.createElement("DEP");
					d.setAttribute("sentidx", String.valueOf(sidx));
					d.setAttribute("toidx", String.valueOf(i));
					d.setAttribute("fromidx", String.valueOf(w.getHeadId()));
					d.setAttribute("rfunc", w.getDeprel().toString());
					DEPS.appendChild(d);

					List<Predicate> predicates = s.getPredicates();
					for (int j = 0; j < predicates.size(); ++j) {
						Predicate pred = predicates.get(j);
						tag = pred.getArgumentTag(w);
						if (tag != null) {
							Element p = null;
							NodeList predList = doc
									.getElementsByTagName("PRED");
							for (int temp = 0; temp < predList.getLength(); temp++) {
								Node predNode = predList.item(temp);
								Element pElement = (Element) predNode;
								if (pElement.getAttribute("sentidx").equals(
										String.valueOf(String.valueOf(sidx)))
										&& pElement.getAttribute("predidx")
												.equals(String.valueOf(pred
														.getIdx()))) {
									p = pElement;
								}
							}

							if (p == null) {
								p = doc.createElement("PRED");
								p.setAttribute("sentidx", String.valueOf(sidx));
								p.setAttribute("predidx",
										String.valueOf(pred.getIdx()));
								p.setAttribute("sense", pred.getSense());
								SRL.appendChild(p);
							}

							Element n = doc.createElement("ARG");
							n.setAttribute("argument", tag);
							n.setAttribute("filleridx", String.valueOf(i));
							p.appendChild(n);
						}
					}
				}

				sidx++;

				forms.clear();
				forms.add("<root>");
				isPred.clear();
				isPred.add(false); // Root is not a predicate
				senCount++;
				if (senCount % 100 == 0) { // TODO fix output in general, don't
											// print to System.out. Wrap a
											// printstream in some (static)
											// class, and allow people to adjust
											// this. While doing this, also add
											// the option to make the output
											// file be -, ie so it prints to
											// stdout. All kinds of errors
											// should goto stderr, and nothing
											// should be printed to stdout by
											// default
					System.out.println("Processing sentence " + senCount);
				}
			} else {
				String[] tokens = WHITESPACE_PATTERN.split(str);
				forms.add(str);
				if (options.skipPI)
					isPred.add(tokens[12].equals("Y"));
			}
		}
		return doc;
	}
	
	
	public static Sentence parse(PreProcess preprocess, List<String> lines, String option) throws Exception {

		String[] words = new String[lines.size()];
		String[] lemmas = new String[lines.size()];
		String[] tags = new String[lines.size()];
		String[] morphs = new String[lines.size()];
		int[] heads = new int[lines.size() - 1];
		String[] deprels = new String[lines.size() - 1];
		int idx = 0;
		for (String line : lines) {
			if (line.equals("<root>")) {
				words[idx] = "<root>";
				lemmas[idx] = "<root-LEMMA>";
				tags[idx] = "<root-POS>";
				morphs[idx] = "<root-FEAT>";
			} else {
				String[] tokens = WHITESPACE_PATTERN.split(line);
				words[idx] = tokens[1];
				lemmas[idx] = tokens[2];
				tags[idx] = tokens[4];
				morphs[idx] = tokens[6];

				if (option.equals("only-srl")) {
					heads[idx - 1] = Integer.parseInt(tokens[8]);
					deprels[idx - 1] = tokens[10];
				}
			}
			idx++;
		}

		Sentence s;

		if (option.equals("only-srl")) {
			s = new Sentence(words, lemmas, tags, morphs);
			s.setHeadsAndDeprels(heads, deprels);
			s.buildDependencyTree();
		} else {
			SentenceData09 instance = new SentenceData09();
			instance.init(words);
			instance.setLemmas(lemmas);
			instance.setPPos(tags);
			instance.setFeats(morphs);
			s = new Sentence(preprocess.pp.preprocess(instance));
		}

		if (!option.equals("only-deps")) {
			preprocess.srl.parseSentence(s);
		}

		return s;
	}
}

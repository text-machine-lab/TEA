
package gateway;

import eus.ixa.ixa.pipe.tok.Tok;
import eus.ixa.ixa.pipe.pos.Pos;
import eus.ixa.ixa.pipe.nerc.Ner;
import eus.ixa.ixa.pipe.parse.Parse;

class EntryPoint {

    Tok tokenizer = null;
    Pos tagger    = null;
    Ner ner       = null;
    Parse parser  = null;

    public EntryPoint() {
        try {

            tokenizer = new Tok();
            tagger    = new Pos();
            ner       = new Ner();
            parser    = new Parse();

        }
        catch(Exception e){
            System.out.println(e.getMessage());
        }
    }

    public Tok getIXATokenizer() {

        return tokenizer;

    }

    public Pos getIXAPosTagger() {

        return tagger;

    }

    public Ner getIXANerTagger() {

        return ner;

    }

    public Parse getIXAParser() {

        return parser;

    }

    public static void main(String[] args) {
        EntryPoint ep = new EntryPoint();

        try {

        }
        catch (Exception e) {

            System.out.println(e.getMessage());

        }

    }

}



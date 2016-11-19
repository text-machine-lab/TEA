package gateway;

import eus.ixa.ixa.pipe.tok.Tok;
import eus.ixa.ixa.pipe.pos.Pos;
import eus.ixa.ixa.pipe.nerc.Ner;
import eus.ixa.ixa.pipe.parse.Parse;

import heidel.HeidelTime;

/**
* Defines object that willbe used to access loaded newsreader components.
*/
class EntryPoint {

    Tok tokenizer = null;
    Pos tagger    = null;
    Ner ner       = null;
    Parse parser  = null;
    HeidelTime heideler = null;

    public EntryPoint() {
        try {
            tokenizer = new Tok();
            tagger    = new Pos();
            ner       = new Ner();
            parser    = new Parse();
            heideler  = new HeidelTime();
        }
        catch(Exception e){
            System.out.println(e.getMessage());
        }
    }

    public HeidelTime getHeideler() {

        return heideler;

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


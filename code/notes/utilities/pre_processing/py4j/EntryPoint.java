
package gateway;

import eus.ixa.ixa.pipe.tok.Tok;
import eus.ixa.ixa.pipe.pos.Pos;

class EntryPoint {

    Tok tokenizer = null;
    Pos tagger    = null;

    public EntryPoint() {
        try {

            tokenizer = new Tok();
            tagger    = new Pos();

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

    public static void main(String[] args) {
        EntryPoint ep = new EntryPoint();

        try {

        }
        catch (Exception e) {

            System.out.println(e.getMessage());

        }

    }

}



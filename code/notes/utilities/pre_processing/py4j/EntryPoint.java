
package gateway;

import eus.ixa.ixa.pipe.tok.Tok;

class EntryPoint {

    Tok tokenizer = null;

    public EntryPoint() {
        try {

            tokenizer = new Tok();

        }
        catch(Exception e){
            System.out.println(e.getMessage());
        }
    }

    public Tok getIXATokenizer() {

        return tokenizer;

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



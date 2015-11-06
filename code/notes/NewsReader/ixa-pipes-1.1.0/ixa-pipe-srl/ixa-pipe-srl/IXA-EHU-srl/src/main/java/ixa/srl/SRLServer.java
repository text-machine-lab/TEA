package ixa.srl;

import ixa.kaflib.KAFDocument;

import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.StringReader;
import java.net.Socket;
import java.net.ServerSocket;

public class SRLServer {

	static int PORT;
	
	public SRLServer(String serverLang) {
	    try {
	    	PreProcess preprocess = new PreProcess(serverLang,"");
			if (serverLang.equals("eng")) {
				PORT=5005;
			} else if (serverLang.equals("spa")) {
				PORT=5007;	
			}

			
	    	BufferedReader stdInReader = null;
						
	    	System.out.println("Trying to listen port " + PORT);
			ServerSocket socketSRLServer = new ServerSocket(PORT);
	    	System.out.println("Listening port " + PORT);
	    	
	    	while(true) {
    			Socket socketSRLClient = socketSRLServer.accept();
				InputStream dataInStream = socketSRLClient.getInputStream();
				DataInput dataInFlow = new DataInputStream(dataInStream);
				OutputStream dataOutStream = socketSRLClient.getOutputStream();
				DataOutputStream dataOutFlow = new DataOutputStream(dataOutStream);

				try {
					String clientLang = dataInFlow.readUTF();
					String option = dataInFlow.readUTF();
					
					StringBuilder stdInStringBuilder = new StringBuilder();
					boolean EnOfInputFile = dataInFlow.readBoolean();
					String line = "";
					while(!EnOfInputFile){
						line = dataInFlow.readUTF();
						stdInStringBuilder.append(line);
						stdInStringBuilder.append('\n');
						EnOfInputFile = dataInFlow.readBoolean();
					}
					String stdInString = stdInStringBuilder.toString();
					
					Annotate annotator = null;
					if (clientLang.equals(serverLang)){
						annotator = new Annotate(preprocess);
					}
					
					stdInReader = new BufferedReader(new StringReader(stdInString));
					KAFDocument kaf = KAFDocument.createFromStream(stdInReader);

					if (option.equals("")) {
						annotator.SRLToKAF(kaf, clientLang, "");
					} else {
						annotator.SRLToKAF(kaf, clientLang, option);
					}
					
					BufferedReader kafReader = null;
					kafReader = new BufferedReader(new StringReader(kaf.toString()));				
					String kafLine = kafReader.readLine();
					while(kafLine != null){
						dataOutFlow.writeBoolean(false);
						dataOutFlow.writeUTF(kafLine);
						kafLine = kafReader.readLine();
					}
					dataOutFlow.writeBoolean(true);
				} catch( Exception e ) {
	    	    	System.out.println( e.getMessage() );
	    	    }
					
				dataOutFlow.flush();
				socketSRLClient.close();
			}
	    } catch( Exception e ) {
	    	System.out.println( e.getMessage() );
	    }
	}
	
	public static void main(String[] args) {
		String lang = "eng";
		if (args[0].equals("en")) {
			lang = "eng";
		} else if (args[0].equals("es")) {
			lang = "spa";
		}
	    new SRLServer(lang);
	}
}
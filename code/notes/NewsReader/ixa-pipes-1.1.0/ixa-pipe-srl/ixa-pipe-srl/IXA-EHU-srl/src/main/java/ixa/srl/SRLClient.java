package ixa.srl;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.Socket;


public class SRLClient {

	static final String HOST = "localhost";
	static int PORT;
	
	public static void main(String[] args) {
		
		try {		
			BufferedReader stdInReader = null;
			stdInReader = new BufferedReader(new InputStreamReader(System.in,"UTF-8"));
			BufferedWriter w = null;
			w = new BufferedWriter(new OutputStreamWriter(System.out, "UTF-8"));
			
/*			StringBuilder stdInStringBuilder = new StringBuilder();
			String line = stdInReader.readLine();
			while(line != null){
				stdInStringBuilder.append(line);
				stdInStringBuilder.append('\n');
				line = stdInReader.readLine();
			}*/
			
			String lang = "eng";
			if (args[0].equals("en")) {
				PORT = 5005;
				lang = "eng";
			} else if (args[0].equals("es")) {
				PORT = 5007;
				lang = "spa";
			}
			
			String options = "";
			if (args.length == 2) {
				options = args[1];
			}
			
			Socket socketSRLClient = new Socket(HOST, PORT);
			InputStream dataInStream = socketSRLClient.getInputStream();
			DataInput dataInFlow = new DataInputStream(dataInStream);
			OutputStream dataOutStream = socketSRLClient.getOutputStream();
			DataOutputStream dataOutFlow = new DataOutputStream(dataOutStream);
				
			dataOutFlow.writeUTF(lang);
			dataOutFlow.writeUTF(options);
			
			String line = stdInReader.readLine();
			while(line != null){
				dataOutFlow.writeBoolean(false);
				dataOutFlow.writeUTF(line);
				line = stdInReader.readLine();
			}
			dataOutFlow.writeBoolean(true);
			
		
			StringBuilder kafStringBuilder = new StringBuilder();
			boolean EnOfKAFFile = dataInFlow.readBoolean();
			String kafLine = "";
			while(!EnOfKAFFile){
				kafLine = dataInFlow.readUTF();
				kafStringBuilder.append(kafLine);
				kafStringBuilder.append('\n');
				EnOfKAFFile = dataInFlow.readBoolean();
			}
			String kafString = kafStringBuilder.toString();
			w.write(kafString);
			w.close();
			
			dataOutFlow.flush();
			socketSRLClient.close();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
	

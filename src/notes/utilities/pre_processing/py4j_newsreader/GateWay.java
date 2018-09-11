package gateway;

import py4j.GatewayServer;

/**
* Launches py4j gateway server. this allows a python script to create a
* gateway object and access the EntryPoint object in the jvm.
*/
class GateWay {

    public GateWay() {
    }

    public static void main(String[] args) {

        GatewayServer gatewayServer = new GatewayServer(new EntryPoint());
        gatewayServer.start();
        System.out.println("Gateway Server Started");

    }

}



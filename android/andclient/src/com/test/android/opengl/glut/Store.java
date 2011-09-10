package com.test.android.opengl.glut;

public class Store {

    private static String serverIP = "130.207.5.46";//"128.61.119.214";

    private static float[] xyzsoapdata = null;

    private static int[] bondssoapdata = null;

    private static boolean pollserver = false;

    /**
     * @return the bondssoapdata
     */
    public static int[] getBondsSoapData() {
        return bondssoapdata;
    }

    /**
     * @return the serverIP
     */
    public static String getServerIP() {
        return serverIP;
    }

    /**
     * @return the xyzsoapdata
     */
    public static float[] getXyzSoapData() {
        return xyzsoapdata;
    }

    /**
     * @return the pollserver
     */
    public static boolean isPollServer() {
        return pollserver;
    }

    /**
     * @param bondssoapdata the bondssoapdata to set
     */
    public static void setBondsSoapData(int[] bondssoapdata) {
        Store.bondssoapdata = bondssoapdata;
    }

    /**
     * @param pollserver the pollserver to set
     */
    public static void setPollServer(boolean pollserver) {
        Store.pollserver = pollserver;
    }

    /**
     * @param serverIP
     *            the serverIP to set
     */
    public static void setServerIP(String serverIP) {
        Store.serverIP = serverIP;
    }

    /**
     * @param xyzsoapdata the xyzsoapdata to set
     */
    public static void setXyzSoapData(float[] xyzsoapdata) {
        Store.xyzsoapdata = xyzsoapdata;
    }

}

package com.test.android.opengl.glut;

import java.io.IOException;

import org.ksoap2.SoapEnvelope;
import org.ksoap2.SoapFault;
import org.ksoap2.serialization.SoapObject;
import org.ksoap2.serialization.SoapSerializationEnvelope;
import org.ksoap2.transport.AndroidHttpTransport;
import org.xmlpull.v1.XmlPullParserException;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

public class SOAP_Activity extends Activity {

    public static class recvLoop implements Runnable {

        public recvLoop() {

        }

        public void run() {
            while (true) {
                try {

                    if (Store.isPollServer()) {
                        int type;
                        // get new data
                        Log.i("SOAP ", "updating xyz data...");
                        //synchronized(Store.getXyzSoapData()) {
                        type = 1;
                        final String xyz = SOAP_Activity.requestSoapData(type);
                        SOAP_Activity.parseSoapData(xyz, type);
                        //}
                        Log.i("SOAP ", "updating neighbor data...");
                        //synchronized(Store.getBondsSoapData()) {
                        type = 2;
                        final String nblist = SOAP_Activity.requestSoapData(type);
                        SOAP_Activity.parseSoapData(nblist, type);
                        //}
                        Log.i("SOAP ", "...done");
                        // update display
                        Log.i("SOAP ", "updating glut data...");
                        GLUTRenderer.updateSoapData();
                        Log.i("SOAP ", "...done");
                    }
                    final int mseconds = (int)(SECONDS_DELAY * 1000);
                    Thread.sleep(mseconds);
                } catch (final Exception e) {
                    e.printStackTrace();
                }

            }
        }
    } // recvLoop

    private static final float SECONDS_DELAY = 1.5f;

    // SOAP Parameters
    private static final String SOAP_ACTION = "";

    private static final String METHOD_NAME = "atTestArrayOfString";

    private static final String NAMESPACE = "urn:at";

    /** soap response array hacking */
    public static void parseSoapData(String result, int type) {

        // Get single Elements from the Result
        // HACK ///////////////////////////////////////////////////////
        final String hack = result.substring(8, result.length() - 3);
        Log.i("HACK ", "Parsing...");
        final String[] hack2 = hack.split("; ");
        //String disphack = "";
        float[] fhackarray;
        int[] ihackarray;

        switch (type) {
        case 1:
            fhackarray = new float[hack2.length];
            for (int i = 0; i < hack2.length; i++) {
                final String hack3 = hack2[i].substring(5);
                //Log.i("HACK3 ", hack3);
                //disphack += hack3 + ",";
                //try {
                fhackarray[i] = Float.parseFloat(hack3);
                //}catch(Exception e) {
                //	Log.i("HACK3 FAIL ", hack3);
                //e.printStackTrace();
                //}
                //Log.i("HACKVAL ", String.valueOf(fhackarray[i]));
            }
            Store.setXyzSoapData(fhackarray);
            break;
        case 2:
            ihackarray = new int[hack2.length];
            for (int i = 0; i < hack2.length; i++) {
                final String hack3 = hack2[i].substring(5);
                //Log.i("HACK3 ", hack3);
                //disphack += hack3 + ",";
                //ihackarray[i] = Integer.parseInt(hack3);
                //try {
                if (hack3.contains("nan")) {
                    continue;
                }
                ihackarray[i] = (int) Float.parseFloat(hack3);
                //}catch(Exception e) {
                //	Log.i("HACK3 FAIL ", hack3);
                //e.printStackTrace();
                //}
                //Log.i("HACKVAL ", String.valueOf(ihackarray[i]));
            }
            Store.setBondsSoapData(ihackarray);
            break;
        default:
            Store.setBondsSoapData(new int[1]);
            Store.setXyzSoapData(new float[1]);
            break;
        }

        Log.i("HACK ", "Saved Data");
        // End HACK ///////////////////////////////////////////////////////

    }

    /** SOAP interfacer
     *
     *  0=random | 1=XYZ | 2=Neighbors
     */
    public static String requestSoapData(int type) {

        // Server Address
        final String URL = "http://" + Store.getServerIP() + ":4749/";

        // SOAP object
        final SoapObject body = new SoapObject(NAMESPACE, METHOD_NAME);

        // Add Property
        final int val = type;
        body.addProperty("cmd", val);

        // Envelope
        final SoapSerializationEnvelope envelope = new SoapSerializationEnvelope(SoapEnvelope.VER11);
        envelope.dotNet = true;
        envelope.setOutputSoapObject(body);

        // Transport
        final AndroidHttpTransport androidHttpTransport = new AndroidHttpTransport(URL);
        try {
            Log.i("SOAP Request: ", "URL=" + URL + " val=" + val);
            androidHttpTransport.call(SOAP_ACTION, envelope);
        } catch (final IOException e) {
            e.printStackTrace();
            return ""; // fail
        } catch (final XmlPullParserException e) {
            e.printStackTrace();
            return ""; // fail
        } // die here

        // Get Response
        SoapObject result = null;
        try {
            result = (SoapObject) envelope.getResponse();
        } catch (final SoapFault e) {
            e.printStackTrace();
            return ""; // fail
        }

        // Get the full Result as a String
        if (result == null) {
            return ""; // fail
        } else {
            return result.toString(); // success
        }
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {

        /////////////////////////
        // App Init            //
        /////////////////////////
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        /////////////////////////
        // Request Result Here //
        /////////////////////////
        final int type = 1;
        final String response = requestSoapData(type);

        /////////////////////////
        // Handle Result Here  //
        /////////////////////////
        parseSoapData(response, type);

        /////////////////////////
        // Display in GUI      //
        /////////////////////////
        final String txtOut = "\n" + "Result: " + response + "\n" + "\n";
        final TextView tv = (TextView) findViewById(R.id.TextView01);
        tv.setText(txtOut);

    }

}

/*
   Copyright [2011] [Chris McClanahan]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

package com.fptd.sensorlogger;

public class Store {

    private static boolean thresholding = false;

    private static int numGs = 2;

    private static int xthreshAcc = 20; // percent of numGs

    private static int ythreshAcc = 30; // percent of numGs

    private static int zthreshAcc = 60; // percent of numGs

    /**
     * @return the numGs
     */
    public static int getNumGs() {
        return numGs;
    }

    /**
     * @return the xthreshAcc
     */
    public static int getXthreshAcc() {
        return xthreshAcc;
    }

    /**
     * @return the ythreshAcc
     */
    public static int getYthreshAcc() {
        return ythreshAcc;
    }

    /**
     * @return the zthreshAcc
     */
    public static int getZthreshAcc() {
        return zthreshAcc;
    }

    /**
     * @return the thresholding
     */
    public static boolean isThresholding() {
        return thresholding;
    }

    /**
     * @param numGs
     *            the numGs to set
     */
    public static void setNumGs(final int numGs) {
        Store.numGs = numGs;
    }

    /**
     * @param thresholding
     *            the thresholding to set
     */
    public static void setThresholding(final boolean thresholding) {
        Store.thresholding = thresholding;
    }

    /**
     * @param xthreshAcc
     *            the xthreshAcc to set
     */
    public static void setXthreshAcc(final int xthreshAcc) {
        Store.xthreshAcc = xthreshAcc;
    }

    /**
     * @param ythreshAcc
     *            the ythreshAcc to set
     */
    public static void setYthreshAcc(final int ythreshAcc) {
        Store.ythreshAcc = ythreshAcc;
    }

    /**
     * @param zthreshAcc
     *            the zthreshAcc to set
     */
    public static void setZthreshAcc(final int zthreshAcc) {
        Store.zthreshAcc = zthreshAcc;
    }

}

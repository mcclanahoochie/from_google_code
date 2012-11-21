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



/**
 * LiveSpectrogram
 * Takes successive FFTs and renders them onto the screen, scrolling left.
 *
 * Dan Ellis dpwe@ee.columbia.edu 2010-01-15
 *    http://www.ee.columbia.edu/~dpwe/resources/Processing/
 *
 *
 *
 * Chris McClanahan chris.mcclanahan@gatech.edu 2011-10-05
 *    http://mcclanahoochie.com/blog/portfolio/music-visualization-with-processing/
 *
 * LiveSpectrogram updates:
 * - HSV colorspace
 * - file-chooser dialog
 * - performance optimizations
 *
 */

import ddf.minim.analysis.*;
import ddf.minim.*;

Minim minim;
AudioPlayer in;
FFT fft;
int colmax = 512;
int rowmax = 256;
int[][] sgram = new int[colmax][rowmax];
int col;
int leftedge = 0;


void setup() {
    // window
    size(512, 256, P3D);
    background(0);
    smooth();

    // object creation
    textMode(SCREEN);
    minim = new Minim(this);

    // file choose
    String fname = "";
    fname = getMp3File();
    if (fname.length() < 3) { exit(); }

    // fft creation
    in = minim.loadFile(fname, 2048);
    in.loop();
    fft = new FFT(in.bufferSize(), in.sampleRate());
    fft.window(FFT.HAMMING);
}


void draw() {

    // colors
    background(0);
    colorMode(HSB, 255);
    int sval = 0;

    // perform a forward FFT on the samples in the input buffer
    fft.forward(in.mix);
    for (int i = 0; i < rowmax /* fft.specSize() */; ++i) {
        // fill in the new column of spectral values (and scale)
        sgram[col][i] = (int)Math.round(Math.max(0, 52 * Math.log10(1000 * fft.getBand(i))));
    }

    // next time will be the next column
    col = col + 1;
    // wrap back to the first column when we get to the end
    if (col == colmax) { col = 0; }

    // Draw points.
    // leftedge is the column in the ring-filled array that is drawn at the extreme left
    // start from there, and draw to the end of the array
    for (int i = 0; i < colmax - leftedge; ++i) {
        for (int j = 0; j < rowmax; ++j) {
            sval = Math.min(255, sgram[i + leftedge][j]);
            stroke(255 - sval, sval, sval);
            point(i, height - j);
        }
    }

    // Draw the rest of the image as the beginning of the array (up to leftedge)
    for (int i = 0; i < leftedge; ++i) {
        for (int j = 0; j < rowmax; ++j) {
            sval = Math.min(255, sgram[i][j]);
            stroke(255 - sval, sval, sval);
            point(i + colmax - leftedge, height - j);
        }
    }

    // Next time around, move the left edge over by one, to have the whole thing scroll left
    leftedge = leftedge + 1;
    // Make sure it wraps around
    if (leftedge == colmax) { leftedge = 0; }
}


void stop() {
    // always close Minim audio classes when you finish with them
    in.close();
    minim.stop();
    super.stop();
}



import javax.swing.*;
/**
filechooser taken from http://processinghacks.com/hacks:filechooser
@author Tom Carden
*/
String getMp3File() {
    try {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
    } catch (Exception e) {
        e.printStackTrace();
    }
    final JFileChooser fc = new JFileChooser();
    int returnVal = fc.showOpenDialog(this);
    String returnstring = "";
    // load
    if (returnVal == JFileChooser.APPROVE_OPTION) {
        File file = fc.getSelectedFile();
        if (file.getName().endsWith("mp3")) {
            returnstring = (file.getPath());
        } else {
            String lines[] = loadStrings(file);
            for (int i = 0; i < lines.length; i++) {
                println(lines[i]);
            }
        }
    } else {
        println("Open command cancelled by user.");
    }
    return returnstring;
}

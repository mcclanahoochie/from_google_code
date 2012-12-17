/*
   Copyright [2012] [Chris McClanahan]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Credit also goes to: http://blurtime.blogspot.com/2010_11_01_archive.html
*/

#include <ks0108.h>
#include "Arial14.h"
#include "SystemFont5x7.h"
#include <fix_fft.h>

char im[128], data[128], lastpass[64];
char x = 30, ylim = 60;
int i = 0, val = 0, j = 0;
char toggle = 0;

const int colmax = 12;//128;    // memory limitation!!!
const int rowmax = 64;
const int width = colmax;
const int height = rowmax;
int col = 0;
int leftedge = 0;
int sgram[colmax][rowmax];

int thresh = 2;

void setup() {
    GLCD.Init(NON_INVERTED);      // non inverted writes pixels onto a clear screen
    analogReference(DEFAULT);     // use default (5v) aref voltage.
    GLCD.SelectFont(System5x7);   // switch to fixed width system font
    GLCD.ClearScreen();
    countdown(4);
    GLCD.ClearScreen();
    for (int z = 0; z < 64; z++) {
        lastpass[z] = 0;
    };
};

void countdown(int count) {
    while (count-- > 0) {
        GLCD.CursorTo(2, 2);
        GLCD.PutChar(count + '0');
        delay(1000);
    }
}

void loop() {

    // -----------------------------------------------------------------
    if (toggle == 0) { GLCD.ClearScreen(); toggle = 1; }
    else { toggle = toggle - 1; }
    int sval = 0;

    // -----------------------------------------------------------------
    for (i = 0; i < 128; i++) {
        val = analogRead(5);     // pin 5
        data[i] = val / 4 - 128; // ?
        im[i] = 0;
    };

    fix_fft(data, im, 7, 0);

    // -----------------------------------------------------------------
    char d;
    for (i = 1; i < rowmax; i++) {
        d = sqrt(data[i] * data[i] + im[i] * im[i]) * 1.33;
        if(d < thresh>>1) d = 0;
        if(d >= ylim) data[i] = ylim - 1;
        GLCD.DrawLine(i + x, lastpass[i], i + x, ylim,     BLACK);
        GLCD.DrawLine(i + x, ylim,        i + x, ylim - d, BLACK);
        data[i] = d;
        lastpass[i] = ylim - d;
    };

    // -----------------------------------------------------------------
    for (i = 0; i < rowmax; i++) {
        sgram[col][i] = data[i];//sqrt(data[i] * data[i] + im[i] * im[i]);
    };

    col = col + 1;
    if (col == colmax) { col = 0; }

    for (i = 0; i < colmax - leftedge; i++) {
        for (j = 0; j < rowmax; ++j) {
            if (sgram[i + leftedge][j] > thresh) {
                GLCD.SetDot(i, height - j, BLACK);
            }
        }
    }

    for (i = 0; i < leftedge; i++) {
        for (j = 0; j < rowmax; ++j) {
            if (sgram[i][j] > thresh) {
                GLCD.SetDot(i + colmax - leftedge, height - j, BLACK);
            }
        }
    }

    leftedge = leftedge + 1;
    if (leftedge == colmax) { leftedge = 0; }

};



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

import android.app.Activity;
import android.os.Bundle;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.CompoundButton.OnCheckedChangeListener;

/**
 * Demonstrates how to use a seek bar
 */
public class Settings extends Activity implements SeekBar.OnSeekBarChangeListener, OnCheckedChangeListener {

    CheckBox mCheckBox_1;

    SeekBar mSeekBar_1;

    SeekBar mSeekBar_2;

    SeekBar mSeekBar_3;

    TextView mProgressText1;

    TextView mProgressText2;

    TextView mProgressText3;

    TextView mTrackingText1;

    TextView mTrackingText2;

    TextView mTrackingText3;

    String percentOf = "";

    public void onCheckedChanged(final CompoundButton buttonView, final boolean isChecked) {
        int selected = buttonView.getId();

        if (selected == mCheckBox_1.getId()) {
            Store.setThresholding(isChecked);
        }

    }

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.settings);

        // configure thresholding
        mCheckBox_1 = (CheckBox) findViewById(R.id.thresholding_1);
        mCheckBox_1.setChecked(Store.isThresholding());
        mCheckBox_1.setOnCheckedChangeListener(this);

        // max acceleration value
        percentOf = " % of " + Store.getNumGs() + "G";

        // X
        mSeekBar_1 = (SeekBar) findViewById(R.id.seek_1);
        mSeekBar_1.setProgress(Store.getXthreshAcc());
        mSeekBar_1.setOnSeekBarChangeListener(this);
        mProgressText1 = (TextView) findViewById(R.id.progress_1);
        mProgressText1.setText("value: " + Store.getXthreshAcc() + percentOf);
        mTrackingText1 = (TextView) findViewById(R.id.tracking_1);

        // Y
        mSeekBar_2 = (SeekBar) findViewById(R.id.seek_2);
        mSeekBar_2.setProgress(Store.getYthreshAcc());
        mSeekBar_2.setOnSeekBarChangeListener(this);
        mProgressText2 = (TextView) findViewById(R.id.progress_2);
        mProgressText2.setText("value: " + Store.getYthreshAcc() + percentOf);
        mTrackingText2 = (TextView) findViewById(R.id.tracking_2);

        // Z
        mSeekBar_3 = (SeekBar) findViewById(R.id.seek_3);
        mSeekBar_3.setProgress(Store.getZthreshAcc());
        mSeekBar_3.setOnSeekBarChangeListener(this);
        mProgressText3 = (TextView) findViewById(R.id.progress_3);
        mProgressText3.setText("value: " + Store.getZthreshAcc() + percentOf);
        mTrackingText3 = (TextView) findViewById(R.id.tracking_3);

    }

    public void onProgressChanged(final SeekBar seekBar, final int progress, final boolean fromTouch) {
        int selected = seekBar.getId();

        if (selected == mSeekBar_1.getId()) {
            mProgressText1.setText("value: " + progress + percentOf);
            Store.setXthreshAcc(progress);
        } else if (selected == mSeekBar_2.getId()) {
            mProgressText2.setText("value: " + progress + percentOf);
            Store.setYthreshAcc(progress);
        } else if (selected == mSeekBar_3.getId()) {
            mProgressText3.setText("value: " + progress + percentOf);
            Store.setZthreshAcc(progress);
        }

    }

    public void onStartTrackingTouch(final SeekBar seekBar) {
        int selected = seekBar.getId();

        if (selected == mSeekBar_1.getId()) {
            // mTrackingText1.setText("tracking on");
        } else if (selected == mSeekBar_2.getId()) {
            // mTrackingText2.setText("tracking on");
        } else if (selected == mSeekBar_3.getId()) {
            // mTrackingText3.setText("tracking on");
        }

    }

    public void onStopTrackingTouch(final SeekBar seekBar) {
        int selected = seekBar.getId();

        if (selected == mSeekBar_1.getId()) {
            // mTrackingText1.setText("tracking off");
        } else if (selected == mSeekBar_2.getId()) {
            // mTrackingText2.setText("tracking off");
        } else if (selected == mSeekBar_3.getId()) {
            // mTrackingText3.setText("tracking off");
        }

    }

}

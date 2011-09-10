/*
 * Copyright (C) 2007 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.test.android.opengl.glut;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.CompoundButton.OnCheckedChangeListener;
import android.widget.TextView.OnEditorActionListener;

/**
 *
 */
public class Settings extends Activity implements OnEditorActionListener, OnCheckedChangeListener {

    EditText mEditText_ip;

    CheckBox mCheckBox_poll;

    @Override
    public void finish() {
        // Save settings
        Log.i("Settings:", "setting IP to: " + mEditText_ip.getText().toString());
        Store.setServerIP(mEditText_ip.getText().toString());
        //Log.i("Settings:", "setting polling to: " + mCheckBox_poll.isChecked());
        //Store.setPollServer(mCheckBox_poll.isChecked());
        super.finish();
    }

    public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
        final int selected = buttonView.getId();

        if (selected == mCheckBox_poll.getId()) {
            Log.i("Settings:", "setting polling to: " + mCheckBox_poll.isChecked());
            Store.setPollServer(isChecked);
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.settings);

        // ip address
        mEditText_ip = (EditText) findViewById(R.id.edit_text_1);
        mEditText_ip.setOnEditorActionListener(this);
        mEditText_ip.setText(Store.getServerIP());

        // polling
        mCheckBox_poll = (CheckBox) findViewById(R.id.check_poll_server);
        mCheckBox_poll.setOnCheckedChangeListener(this);
        mCheckBox_poll.setChecked(Store.isPollServer());

    }

    public boolean onEditorAction(TextView v, int actionId, KeyEvent event) {
        //final int selected = v.getId();

        //if (selected == mEditText_ip.getId()) {
        //	Log.i("Settings:", "setting IP to: " + mEditText_ip.getText().toString());
        //	Store.setServerIP(mEditText_ip.getText().toString());
        //	return true;
        //}
        return false;
    }

}

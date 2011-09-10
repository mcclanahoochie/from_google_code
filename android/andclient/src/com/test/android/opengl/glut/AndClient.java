/*
 * Copyright (C) 2008
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

import android.app.ListActivity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.ListView;

public class AndClient extends ListActivity {
    private final String[] titles = { "Cube", "XYZ", "Settings", "SOAP Test", };

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setTitle("Android SmartP Client");

        setListAdapter(new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1, titles));
        getListView().setTextFilterEnabled(true);
    }

    @Override
    protected void onListItemClick(ListView l, View v, int position, long id) {
        super.onListItemClick(l, v, position, id);

        Intent i = null;
        if (position == 0) {
            i = new Intent(this, CubeActivity.class);
        } else if (position == 1) {
            i = new Intent(this, XYZActivity.class);
        } else if (position == 2) {
            i = new Intent(this, Settings.class);
        } else if (position == 3) {
            i = new Intent(this, SOAP_Activity.class);
        }

        if (i != null) {
            startActivity(i);
        }
    }
}

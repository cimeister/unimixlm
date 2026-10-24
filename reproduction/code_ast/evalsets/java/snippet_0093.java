package com.example.malte.app;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import com.mindorks.placeholderview.PlaceHolderView;

import org.apache.http.client.ResponseHandler;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.BasicResponseHandler;
import org.apache.http.impl.client.DefaultHttpClient;
import org.json.*;

import java.io.*;
import java.net.URL;
import java.net.URLConnection;
import java.text.SimpleDateFormat;
import java.util.*;

public class MainActivity extends AppCompatActivity {

    PlaceHolderView mGalleryView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mGalleryView = (PlaceHolderView) findViewById(R.id.galleryView);

        try {
            InputStream in = getResources().openRawResource(R.raw.tweets);
            InputStreamReader inputreader = new InputStreamReader(in);
            BufferedReader buffreader = new BufferedReader(inputreader);
            String line;
            StringBuilder text = new StringBuilder();
            try {
                while (( line = buffreader.readLine()) != null) {
                    text.append(line);
                    text.append('\n');
                }
            } catch (IOException e) {
                out("Error");
            }
            String result = text.toString();
            JSONArray jArray = new JSONArray(result);
            for (int i=0; i < jArray.length(); i++) {
                JSONObject oneObject = jArray.getJSONObject(i);
                String profilname = oneObject.getString("profilname");
                String tweet = oneObject.getString("tweet");
                String tweetid = oneObject.getString("tweetid");
                String tweetlink = oneObject.getString("tweetlink");
                String profilimage = oneObject.getString("profilimage");
                String date = oneObject.getString("date");
                int yes = oneObject.getInt("yes");
                int no = oneObject.getInt("no");
                int total = yes+no;
                ArrayList<Integer> list = new ArrayList<>();
                list.add(total);
                list.add(yes);

                String pattern = "HH:mm:ss";
                SimpleDateFormat sdf = new SimpleDateFormat(pattern);
                Date daten = sdf.parse(date);

                SimpleDateFormat dateFormat = new SimpleDateFormat("HH:mm:ss");
                mGalleryView.addView(new Item(this.getApplicationContext(), mGalleryView,
                        profilimage, profilname, tweet, "", dateFormat.format(daten), list, mGalleryView));
            }
        } catch (Exception e){
            e.printStackTrace();
        }
    }

    private void out(String x) {
        Log.d("DatenParty", x);
    }
}
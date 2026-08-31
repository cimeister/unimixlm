//
//  MyApplication.java
//  EC-sample
//
//  Created by Trial: <NAME> on 2019/08/09.
//  Copyright © 2019 Repro CRE. All rights reserved.
//

package com.company.ecsample;

import android.app.Activity;
import android.app.Application;
import android.os.Handler;
import android.util.Log;

import com.company.ecsample.Utils.Token;

import io.repro.android.Repro;


public class MyApplication extends Application {

    @Override
    public void onCreate() {
        super.onCreate();

        /**
         * Use the following method for debaug your application.
         *
         * Refer to the documentation for more detail information.
         * (JP) https://docs.repro.io/ja/dev/sdk/log.html#id1
         * (EN) https://docs.repro.io/en/dev/sdk/log.html#id1
         */
        Repro.setLogLevel(Log.DEBUG);


        /**
         * Using the follwing method, You can deley showing the inApp message.
         * This method is useful when you fire inApp message as the application boot up. For it sometimes happens that inApp message doesn't
         * appear because the splash screen prevents it.
         *
         * Refer to the documentation for more detail information.
         * (JP) https://docs.repro.io/ja/dev/sdk/in-app-message.html#id3
         * (EN) https://docs.repro.io/en/dev/sdk/in-app-message.html#id3
         */
        Repro.disableInAppMessageOnActive();


        /**
         * Set up Repro
         * Refer to the documentation for more detail information.
         * (JP) https://docs.repro.io/ja/dev/sdk/getstarted/ios.html#ios-setup
         * (EN) https://docs.repro.io/en/dev/sdk/getstarted/ios.html#ios-setup
         */
        Repro.setup(this, Token.SDK_TOKEN);


        /**
         * Push notification setup
         * You need a FCM setting to send push notifications
         * Refer to the documentation for more detail information.
         * (JP) https://docs.repro.io/ja/dev/sdk/push-notification/setup-android.html
         * (EN) https://docs.repro.io/en/dev/sdk/push-notification/setup-android.html
         */
        Repro.enablePushNotification();
    }

    private Activity mCurrentActivity = null;

    public void showInAppMessage(Activity CurrentActivity) {

        mCurrentActivity = CurrentActivity;

        /**
         * Show inApp message.
         * The following method is only used when you use splash screen in the app.
         *
         * Refer to the documentation for more detail information.
         * (JP) https://docs.repro.io/ja/dev/sdk/in-app-message.html#id3
         * (EN) https://docs.repro.io/en/dev/sdk/in-app-message.html#id3
         */
        // If it does not appear, add delay processing.
        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                Repro.showInAppMessage(mCurrentActivity);
            }
        }, 500);


    }
}
/*
 * Copyright (C) 2019 <NAME> (<EMAIL>)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package be.rufer.babybottlecalculator;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.SeekBar;
import android.widget.TextView;

import be.rufer.babybottlecalculator.calculation.MilkPowderCalculator;
import be.rufer.babybottlecalculator.calculation.WaterCalculator;

public class MainActivity extends AppCompatActivity
{
    private EditText weightInput;
    private EditText daysOnEarthInput;
    private SeekBar mealsSeekBar;
    private TextView mealsSeekBarValue;
    private Button calculateButton;
    private TextView resultView;

    private static final String EMPTY_STRING = "";
    private static final String SEEK_BAR_DELIMITER = "/";

    private TextWatcher textWatcher = new TextWatcher()
    {
        @Override
        public void beforeTextChanged(CharSequence charSequence, int i, int i2, int i3)
        {
            // intentionally do nothing
        }

        @Override
        public void onTextChanged(CharSequence charSequence, int i, int i2, int i3)
        {
            checkInputFieldsForEmptyValues();
        }

        @Override
        public void afterTextChanged(Editable editable)
        {
            // intentionally do nothing
        }
    };

    private SeekBar.OnSeekBarChangeListener onSeekBarChangeListener = new SeekBar.OnSeekBarChangeListener() {
        int pval = 0;

        @Override
        public void onProgressChanged(SeekBar seekBar, int progress, boolean b) {
            resultView.setText(EMPTY_STRING);
            pval = progress;
        }

        @Override
        public void onStartTrackingTouch(SeekBar seekBar) {
            // intentionally do nothing
        }

        @Override
        public void onStopTrackingTouch(SeekBar seekBar) {
            mealsSeekBarValue.setText(pval + SEEK_BAR_DELIMITER + seekBar.getMax());
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        weightInput = (EditText) findViewById(R.id.weightInput);
        daysOnEarthInput = (EditText) findViewById(R.id.daysOnEarthInput);
        mealsSeekBar = (SeekBar) findViewById(R.id.mealsSeekBar);
        mealsSeekBarValue = (TextView) findViewById(R.id.mealsSeekBarValue);
        calculateButton = (Button) findViewById(R.id.calculateButton);
        resultView = (TextView) findViewById(R.id.resultView);

        weightInput.addTextChangedListener(textWatcher);
        daysOnEarthInput.addTextChangedListener(textWatcher);
        mealsSeekBar.setOnSeekBarChangeListener(onSeekBarChangeListener);

        checkInputFieldsForEmptyValues();
    }

    private void checkInputFieldsForEmptyValues()
    {
        resultView.setText(EMPTY_STRING);

        String weightInputText = weightInput.getText().toString();
        String daysOnEarthInputText = daysOnEarthInput.getText().toString();

        if (weightInputText.equals(EMPTY_STRING) || daysOnEarthInputText.equals(EMPTY_STRING))
        {
            calculateButton.setEnabled(false);
        }
        else
        {
            calculateButton.setEnabled(true);
        }
    }

    /** Called when the user taps the Send button */
    public void calculate(View view)
    {
        String weightInputText = weightInput.getText().toString();
        double birthWeight = Double.parseDouble(weightInputText);
        String daysOnEarthInputText = daysOnEarthInput.getText().toString();
        int daysOnEarth = Integer.parseInt(daysOnEarthInputText);
        int numberOfMeals = mealsSeekBar.getProgress();

        double amountOfWater = WaterCalculator.calculateAmountOfWater(birthWeight, daysOnEarth, numberOfMeals);
        double spoons = MilkPowderCalculator.calculateSpoons(amountOfWater);

        resultView.setText(String.format(
                "Amount of water: %.2fml%nSpoons: %.2f",
                amountOfWater,
                spoons));
    }
}
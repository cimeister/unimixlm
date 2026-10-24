<?php

namespace udokmeci\yii2payment\models;

use Yii;
use yii\behaviors\TimestampBehavior;
use yii\db\Expression;
/**
 * This is the model class for table "exchange_rates".
 *
 * @property string $currency_code
 * @property string $rate
 * @property string $updated
 */
class ExchangeRates extends \yii\db\ActiveRecord
{

   

    public function behaviors()
    {
        return [
            [
                'class' =>TimestampBehavior::className(),
                'createdAtAttribute' => false,
                'updatedAtAttribute' => 'updated',
                'value' => new Expression('NOW()'),
            ]

        ];
    }
    /**
     * @inheritdoc
     */
    public static function tableName()
    {
        return 'exchange_rates';
    }

    /**
     * @inheritdoc
     */
    public function rules()
    {
        return [
            [['currency_code', 'rate'], 'required'],      
            [['updated'], 'safe'],
            [['currency_code'], 'string', 'max' => 5]
        ];
    }

    public static function convert($value,$from,$to,$fixed=false){
        $from_model = self::findOne($from);
        $to_model = self::findOne($to);

        if (!$from_model || !$to_model) 
            throw new \Exception("Unknown Currency Convertion", 1);
        $converted=$value * $to_model->rate / $from_model->rate;
        if(!$fixed)
            return floatval($converted);
        if($fixed)
            return number_format($converted,$to_model->currency->E);

    }

    public function beforeSave($insert)
    {
        if($insert)
            return parent::beforeSave($insert);

        if (parent::beforeSave($insert) ) {
            $change=($this->getOldAttribute("rate")/$this->rate)*100;


            if( abs($change) < 90 && abs($change) > 110  ){
                Yii::warning($this->currency_code . "value has changed over the limits %10 so did not saved. It was ".$this->getOldAttribute("rate"). " now ".$this->rate,"exchangeRates" );
                return false;
            }
                
            return true;
        } else {
            return false;
        }
    }

    /**
     * @inheritdoc
     */
    public function attributeLabels()
    {
        return [
            'currency_code' => 'Currency Code',
            'rate' => 'Rate',
            'updated' => 'Updated',
        ];
    }

    public function getCurrency()
    {
        return $this->hasOne(Currency::className(), ['code' => 'currency_code']);
    }
}
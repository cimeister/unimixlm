<?php

use yii\helpers\Html;
use yii\helpers\Url;
use yii\widgets\ActiveForm;
use kartik\date\DatePicker;

/* @var $this yii\web\View */
/* @var $payment app\models\Payments */
/* @var $form yii\widgets\ActiveForm */
?>
<?php 

    $payments = $model->payments;
    $total_paid = 0;
    if($payments){
        foreach ($payments as $p) {
            $total_paid += $p->amount ;
        }
    }
    $remaining = $model->amount - $total_paid ;

    $outstandings = $model->outstanding;
    $credit = 0;
    if($outstandings){
        foreach ($outstandings as $o) {
            $credit += $o->amount ;
        }
    }

    $creditMax = $remaining - $credit ;
?>
<div class="cheque-form">

    <?php $form = ActiveForm::begin(
             [   
            'id' => 'cheque-form',
            'options'=>['method' => 'post'],
            'action' => Url::to(['invoices/cheque', 'id' => $model->id]),
            
            ]);
    ?>

    	<?= $form->field($outstanding, "bank")
	        ->textInput(
	            [	  
	                'placeholder'=> Yii::t('invo', 'Bank Name'),
	            ])
	        ->label(false)
        ?>

        <?= $form->field($outstanding, "cheque_no")
	        ->textInput(
	            [	  
	                'placeholder'=> Yii::t('invo', 'Cheque No'),
	            ])
	        ->label(false)
        ?>

        <?= $form->field($outstanding, "cheque_date")
        	->widget(DatePicker::classname(), 
        		[
                    'type' => DatePicker::TYPE_INPUT,
                    'options' => ['placeholder' => Yii::t('invo', 'Cheque Date')],
                    'pluginOptions' => [
                        'format' => 'yyyy-mm-dd',
                        'todayHighlight' => true
                	]    
                ])
        	->label(false);
        ?>     
    	<?= $form->field($outstanding, 'amount')
	        ->textInput(
	            [
	                'type' => 'number', 
	                'max' => $creditMax, 
	                'min' => 1,
	                'placeholder'=> Yii::t('invo', 'Amount'),
	            ])
	        ->label(false);
    	?>


    <div class="form-group">
        <?= Html::submitButton(Yii::t('invo', 'Save'), ['class' => 'btn btn-block bg-navy']) ?>
    </div>

    <?php ActiveForm::end(); ?>

</div>
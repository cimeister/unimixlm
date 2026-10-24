<?php
/* @var $this yii\web\View */
/* @var $model \common\models\ExportWasteReminderForm */
?>
<table width="100%" cellpadding="0" cellspacing="0">
    <tr>
        <td class="content-block">
            <strong>Уважаемый клиент!</strong>
        </td>
    </tr>
    <tr>
        <td class="content-block">
            <p>Напоминаем вам, что <strong><?= Yii::$app->formatter->asDate($model->date, 'php:d.m.Y г.') ?></strong> состоится вывоз отходов.</p>
            <p>Просим с вашей стороны не забывать выдавать водителю 3 экз. Транспортной накладной  и 2 экз. Акта приема-передачи в заполненном  виде!</p>
            <p>Просим также ставить печать в п.6 и п.16 в ТН, во избежание проблем транспортировки отходов на наше производство.</p>
        </td>
    </tr>
    <tr>
        <td class="content-block">
            <strong>Автомобиль</strong>: <?= $model->transport_info ?>

        </td>
    </tr>
    <tr>
        <td class="content-block">
            <strong>Водитель</strong>: <?= $model->driver_info ?>
        </td>
    </tr>
    <tr>
        <td class="content-block">
            <p>Негабаритным грузом считается груз превышающим 2,55 м в ширину, 20 м в длину от капота авто (включая прицеп) и 4 м в высоту от проезжей части с учётом груза.</p>
            <p>Заранее благодарим</p>
        </td>
    </tr>
</table>
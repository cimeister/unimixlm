<?php

use yii\helpers\Html;
use yii\widgets\ActiveForm;
use app\modules\correspondence\controllers;

$this->registerCss("
li.active{
background: #C6C4C4
}
");
$this->title = Html::encode($this->title) . 'ตั้งค่าประเภทเอกสาร';
$this->registerJsFile('@web/../modules/correspondence/style/js/setting-document.js', ['depends' => [\yii\web\JqueryAsset::className()]]);
?>

    <section id="middle" style="color: black">
        <div id="content" class="padding-30">
            <div id="panel-ui-tan-l1" class="panel panel-default">

                <div class="panel-heading">

									<span class="elipsis"><!-- panel title -->
                                        <?php

                                        ?>
                                        <b><?= controllers::t('menu', 'Settings') ?></b> <!-- panel title -->
									</span>
                </div>
                <!-- panel content -->
                <div class="panel-body padding-40">
                    <div class="tabs nomargin table-responsive">

                        <!-- tabs nav -->
                        <ul class="nav nav-tabs nav-justified tabsetting">
                            <li class="" id="firstTab">
                                <a href="#ttab3_nobg" data-toggle="tab"
                                   aria-expanded="false"><?= controllers::t('menu', 'Document type') ?></a>
                            </li>
                            <li class="">
                                <a href="#ttab5_nobg" data-toggle="tab"
                                   aria-expanded="false"><?= controllers::t('menu', 'Category') ?></a>
                            </li>
                            <li class="">
                                <a href="#ttab4_nobg" data-toggle="tab"
                                   aria-expanded="false"><?= controllers::t('menu', 'Organization') ?></a>
                            </li>
                            <li class="">
                                <a href="#ttab6_nobg" data-toggle="tab"
                                   aria-expanded="false"><?= controllers::t('menu', 'Folder') ?></a>
                            </li>
                            <li class="">
                                <a href="#ttab1_nobg" data-toggle="tab"
                                   aria-expanded="true"><?= controllers::t('menu', 'Speed') ?></a>
                            </li>
                            <li class="">
                                <a href="#ttab2_nobg" data-toggle="tab"
                                   aria-expanded="false"><?= controllers::t('menu', 'Secret') ?></a>
                            </li>

                        </ul>
                        <!-- /tabs nav -->
                        <!-- tabs content -->
                        <div class="tab-content transparent" style="margin-top: 20px">

                            <div id="ttab1_nobg" class="tab-pane"><!-- TAB 1 CONTENT -->
                                <table class="table table-bordered table-striped example">
                                    <thead>
                                    <tr>
                                        <th><?= controllers::t('menu', 'Lists') ?></th>
                                        <th><?= controllers::t('menu', 'Edit') ?></th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    <?php
                                    foreach ($model_speed as $rows) {
                                        $checkDoc = \app\modules\correspondence\models\CmsDocSpeed::find()
                                            ->from(['cms_document', 'cms_doc_speed'])
                                            ->where('cms_document.speed_id = ' . $rows['speed_id'])
                                            ->one();
                                        echo "<tr>";
                                        echo "<td>" . $rows['speed_name'] . "</td>";
                                        if ($checkDoc) {
                                            echo "<td>" . Html::a("<i class=\"fa fa-edit\"></i>
                                            <span>" . controllers::t('menu', 'Edit') . "</span>",
                                                    ['speed/update?id=' . $rows['speed_id']],
                                                    ['class' => 'btn btn-3d btn-sm btn-reveal btn-warning']) . ""
                                                . "</td>";
                                        } else {
                                            echo "<td>" . Html::a("<i class=\"fa fa-edit\"></i>
                                            <span>" . controllers::t('menu', 'Edit') . "</span>",
                                                    ['speed/update?id=' . $rows['speed_id']],
                                                    ['class' => 'btn btn-3d btn-sm btn-reveal btn-warning']) . "
                                         <a href='#ttab1_nobg' onclick='redirectDeleteRoll(" . $rows['speed_id'] . ")'
                                               class='btn btn-3d btn-sm btn-reveal btn-danger confirmDeleteSpeed'>
                                                <i class='fa fa-trash'></i>
                                                  <span>" . controllers::t('menu', 'Delete') . "</span></a>"
                                                . "</td>";
                                        }
                                        echo "</tr>";
                                    }
                                    ?>
                                    </tbody>
                                </table>
                                <div>
                                    <a href="#NewSpeedModal" class="btn btn-success" style="font-size:22px; width: 100%"
                                       data-toggle="modal" data-whatever="@mdo"><?= controllers::t('menu', 'Add') ?>
                                    </a>
                                </div>
                            </div><!-- /TAB 1 CONTENT -->

                            <div id="ttab2_nobg" class="tab-pane"><!-- TAB 2 CONTENT -->
                                <table class="table table-bordered table-striped example">
                                    <thead>
                                    <tr>
                                        <th><?= controllers::t('menu', 'Lists') ?></th>
                                        <th><?= controllers::t('menu', 'Edit') ?></th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    <?php
                                    foreach ($model_secret as $rows) {
                                        $checkDoc = \app\modules\correspondence\models\CmsDocSecret::find()
                                            ->from(['cms_document', 'cms_doc_secret'])
                                            ->where('cms_document.secret_id = ' . $rows['secret_id'])
                                            ->one();
                                        echo "<tr>";
                                        echo "<td>" . $rows['secret_name'] . "</td>";
                                        if ($checkDoc) {
                                            echo "<td>" . Html::a("<i class=\"fa fa-edit\"></i>
                                            <span>" . controllers::t('menu', 'Edit') . "</span>",
                                                    ['secret/update?id=' . $rows['secret_id']],
                                                    ['class' => 'btn btn-3d btn-sm btn-reveal btn-warning']) . ""
                                                . "</td>";
                                        } else {
                                            echo "<td>" . Html::a("<i class=\"fa fa-edit\"></i>
                                            <span>" . controllers::t('menu', 'Edit') . "</span>",
                                                    ['secret/update?id=' . $rows['secret_id']],
                                                    ['class' => 'btn btn-3d btn-sm btn-reveal btn-warning']) . "
                                         <a href='#ttab2_nobg' onclick='redirectDeleteRoll(" . $rows['secret_id'] . ")'
                                               class='btn btn-3d btn-sm btn-reveal btn-danger confirmDeleteSecret'>
                                                <i class='fa fa-trash'></i>
                                                  <span>" . controllers::t('menu', 'Delete') . "</span></a>"
                                                . "</td>";
                                        }
                                        echo "</tr>";
                                    }
                                    ?>
                                    </tbody>
                                </table>
                                <div>
                                    <a href="#NewSecretModal" class="btn btn-success"
                                       style="font-size:22px; width: 100%"
                                       data-toggle="modal" data-whatever="@mdo"><?= controllers::t('menu', 'Add') ?>
                                    </a>
                                </div>
                            </div><!-- /TAB 2 CONTENT -->
                            <div id="ttab3_nobg" class="tab-pane active"><!-- TAB 3 CONTENT -->
                                <table class="table table-bordered table-striped example">
                                    <thead>
                                    <tr>
                                        <th><?= controllers::t('menu', 'Lists') ?></th>
                                        <th><?= controllers::t('menu', 'Edit') ?></th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    <?php
                                    foreach ($model_type as $rows) {
                                        $checkDoc = \app\modules\correspondence\models\CmsDocType::find()
                                            ->from(['cms_document', 'cms_doc_type'])
                                            ->where('cms_document.type_id = ' . $rows['type_id'])
                                            ->one();
                                        echo "<tr>";
                                        echo "<td>" . $rows['type_name'] . "</td>";
                                        if ($checkDoc) {
                                            echo "<td>" . Html::a("<i class=\"fa fa-edit\"></i>
                                            <span>" . controllers::t('menu', 'Edit') . "</span>",
                                                    ['type/update?id=' . $rows['type_id']],
                                                    ['class' => 'btn btn-3d btn-sm btn-reveal btn-warning']) . ""
                                                . "</td>";
                                        } else {
                                            echo "<td>" . Html::a("<i class=\"fa fa-edit\"></i>
                                            <span>" . controllers::t('menu', 'Edit') . "</span>",
                                                    ['type/update?id=' . $rows['type_id']],
                                                    ['class' => 'btn btn-3d btn-sm btn-reveal btn-warning']) . "
                                         <a href='#ttab3_nobg' onclick='redirectDeleteRoll(" . $rows['type_id'] . ")'
                                               class='btn btn-3d btn-sm btn-reveal btn-danger confirmDeleteType'>
                                                <i class='fa fa-trash'></i>
                                                  <span>" . controllers::t('menu', 'Delete') . "</span></a>"
                                                . "</td>";
                                        }
                                        echo "</tr>";
                                    }
                                    ?>
                                    </tbody>
                                </table>
                                <div>
                                    <a href="#NewTypeModal" class="btn  btn-success" style="font-size:22px; width: 100%"
                                       data-toggle="modal" data-whatever="@mdo"><?= controllers::t('menu', 'Add') ?>
                                    </a>
                                </div>
                            </div><!-- /TAB 3 CONTENT -->
                            <div id="ttab4_nobg" class="tab-pane"><!-- TAB 4 CONTENT -->
                                <b style="color: red">*หน่วยงานที่เคยบันทึกในระบบ</b><br>
                                <br>
                                <table class="table table-bordered table-striped example">
                                    <thead>
                                    <tr>
                                        <th><?= controllers::t('menu', 'Lists') ?></th>
                                        <th><?= controllers::t('menu', 'Edit') ?></th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    <?php

                                    foreach ($models_from_dept as $model) {
                                        $checkDoc = \app\modules\correspondence\models\CmsDocDept::find()
                                            ->from(['cms_document', 'cms_doc_dept'])
                                            ->where('cms_document.doc_dept_id = ' . $model['doc_dept_id'])
                                            ->one();
                                        //
                                        echo "<tr>";
                                        echo "<td>" . $model['doc_dept_name'] . "</td>";
                                        if ($checkDoc) {
                                            echo "<td>" . Html::a("<i class=\"fa fa-edit\"></i>
                                            <span>" . controllers::t('menu', 'Edit') . "</span>",
                                                    ['doc-dept/update?id=' . $model['doc_dept_id']],
                                                    ['class' => 'btn btn-3d btn-sm btn-reveal btn-warning']); ?>
                                            <?php
                                            echo "</td>";
                                        } else {
                                            echo "<td>" . Html::a("<i class=\"fa fa-edit\"></i>
                                            <span>" . controllers::t('menu', 'Edit') . "</span>",
                                                    ['doc-dept/update?id=' . $model['doc_dept_id']],
                                                    ['class' => 'btn btn-3d btn-sm btn-reveal btn-warning']); ?>
                                            <a href='#ttab4_nobg'
                                               onclick='redirectDeleteRoll("<?= $model['doc_dept_id'] ?>")'
                                               class='btn btn-3d btn-sm btn-reveal btn-danger confirmDeleteFromDept'>
                                                <i class='fa fa-trash'></i>
                                                <span><?= controllers::t('menu', 'Delete') ?></span></a>
                                            <?php
                                            echo "</td>";
                                        }
                                    }
                                    // display pagination

                                    ?>
                                    </tbody>
                                </table>
                                <div>
                                    <a href="#NewOrganModal" class="btn btn-success" style="font-size:22px; width: 100%"
                                       data-toggle="modal" data-whatever="@mdo"><?= controllers::t('menu', 'Add') ?>
                                    </a>
                                </div>
                            </div><!-- /TAB 4 CONTENT -->
                            <div id="ttab5_nobg" class="tab-pane "><!-- TAB 5 CONTENT -->
                                <b style="color: red">*การเงิน , พัสดุ</b><br>
                                <table class="table table-bordered table-striped example">
                                    <thead>
                                    <tr>
                                        <th><?= controllers::t('menu', 'Lists') ?></th>
                                        <th><?= controllers::t('menu', 'Edit') ?></th>
                                    </tr>
                                    </thead>
                                    <tbody>
                                    <?php
                                    foreach ($model_sub_type as $rows) {
                                        $checkDoc = \app\modules\correspondence\models\CmsDocSubType::find()
                                            ->from(['cms_document', 'cms_doc_sub_type', 'cms_address'])
                                            ->where('cms_document.sub_type_id =' . $rows['sub_type_id'])
                                            ->orWhere('cms_address.sub_type_id = ' . $rows['sub_type_id'])
                                            ->groupBy('cms_doc_sub_type.sub_type_id')
                                            ->one();
                                        echo "<tr>";
                                        echo "<td>" . $rows['sub_type_name'] . "</td>";
                                        if ($checkDoc) {
                                            echo "<td>" . Html::a("<i class=\"fa fa-edit\"></i>
                                            <span>" . controllers::t('menu', 'Edit') . "</span>",
                                                    ['sub-type/update?id=' . $rows['sub_type_id']],
                                                    ['class' => 'btn btn-3d btn-sm btn-reveal btn-warning']) . ""
                                                . "</td>";
                                        } else {
                                            echo "<td>" . Html::a("<i class=\"fa fa-edit\"></i>
                                            <span>" . controllers::t('menu', 'Edit') . "</span>",
                                                    ['sub-type/update?id=' . $rows['sub_type_id']],
                                                    ['class' => 'btn btn-3d btn-sm btn-reveal btn-warning']) . "
                                         <a href='#ttab5_nobg' onclick='redirectDeleteRoll(" . $rows['sub_type_id'] . ")'
                                               class='btn btn-3d btn-sm btn-reveal btn-danger confirmDeleteSubType'>
                                                <i class='fa fa-trash'></i>
                                                  <span>" . controllers::t('menu', 'Delete') . "</span></a>"
                                                . "</td>";
                                        }
                                        echo "</tr>";
                                    }
                                    ?>
                                    </tbody>
                                </table>
                                <div>
                                    <a href="#NewSubTypeModal" class="btn btn-success"
                                       style="font-size:22px; width: 100%"
                                       data-toggle="modal" data-whatever="@mdo"><?= controllers::t('menu', 'Add') ?>
                                    </a>
                                </div>
                            </div><!-- /TAB 5 CONTENT -->
                            <div id="ttab6_nobg" class="tab-pane "><!-- TAB 6 CONTENT -->
                                <table class="table table-bordered table-striped example">
                                    <thead>
                                    <tr>
                                        <th><?= controllers::t('menu', 'Category') ?></th>
                                        <th><?= controllers::t('menu', 'Address ID') ?></th>
                                        <th><?= controllers::t('menu', 'Year') ?></th>
                                        <th><?= controllers::t('menu', 'Address Name') ?></th>
                                        <th><?= controllers::t('menu', 'Edit') ?></th>
                                    </tr>
                                    </thead>

                                    <tbody><?php
                                    foreach ($model_address as $rows) {
                                        $checkDoc = \app\modules\correspondence\models\CmsAddress::find()
                                            ->from(['cms_document', 'cms_address'])
                                            ->where('cms_document.address_id = "' . $rows['address_id'] . '"')
                                            ->one();
                                        echo "<tr>";
                                        echo "<td>" . $rows->subType->sub_type_name . "</td>";
                                        echo "<td>" . $rows['address_id'] . "</td>";
                                        echo "<td>" . $rows['address_year'] . "</td>";
                                        echo "<td>" . $rows['address_name'] . "</td>";
                                        if ($checkDoc) {
                                            echo "<td>" . Html::a("<i class=\"fa fa-edit\"></i>
                                            <span>" . controllers::t('menu', 'Edit') . "</span>",
                                                    ['address/update?id=' . $rows['address_id']],
                                                    ['class' => 'btn btn-3d btn-sm btn-reveal btn-warning']) . ""
                                                . "</td>";
                                        } else {
                                            echo "<td>" . Html::a("<i class=\"fa fa-edit\"></i>
                                            <span>" . controllers::t('menu', 'Edit') . "</span>",
                                                    ['address/update?id=' . $rows['address_id']],
                                                    ['class' => 'btn btn-3d btn-sm btn-reveal btn-warning']) . "
                                         <a href='#ttab6_nobg' onclick='passParamFromDept(\"" . $rows['address_id'] . "\")'
                                               class='btn btn-3d btn-sm btn-reveal btn-danger confirmDeleteAddress'>
                                                <i class='fa fa-trash'></i>
                                                  <span>" . controllers::t('menu', 'Delete') . "</span></a>"
                                                . "</td>";
                                        }
                                        echo "</tr>";
                                    }
                                    ?>
                                    </tbody>
                                </table>
                                <div>
                                    <a href="#NewAddressModal" class="btn btn-success"
                                       style="font-size:22px; width: 100%"
                                       data-toggle="modal" data-whatever="@mdo"><?= controllers::t('menu', 'Add') ?>
                                    </a>
                                </div>
                            </div><!-- /TAB 6 CONTENT -->
                        </div>
                        <!-- /tabs content -->

                    </div>
                </div>
                <!-- /panel content -->

            </div>

    </section>
    <div class="modal fade" id="NewSpeedModal" role="dialog">
        <div class="modal-dialog" style="width: 50%">
            <div class="modal-content">
                <div class="modal-header">
                    <button type="button" class="close" data-dismiss="modal" id="close1">&times;</button>
                    <h4 class="modal-title"><?= controllers::t('menu', 'Add Speed') ?></h4>
                </div>
                <div class="modal-body">
                    <?php
                    $speed = new \app\modules\correspondence\models\CmsDocSpeed();
                    $id_speed = \app\modules\correspondence\models\CmsDocSpeed::find()->orderBy(['speed_id' => SORT_DESC])->one();

                    $form = ActiveForm::begin(['action' => '../speed/create', 'method' => 'post'], ['options' => ['id' => 'form', 'class' => 'form-horizontal', 'enctype' => 'multipart/form-data']]) ?>
                    <?= $form->field($speed, 'speed_id')->hiddenInput(['value' => $id_speed->speed_id + 1])->label(false) ?>
                    <?= $form->field($speed, 'speed_name')->textInput(['maxlength' => true]) ?>
                    <div class="form-group">
                        <?= Html::submitButton($speed->isNewRecord ? controllers::t('menu', 'Create') : controllers::t('menu', 'Edit'), ['class' => $speed->isNewRecord ? 'btn btn-success' : 'btn btn-primary']) ?>
                    </div>

                    <?php ActiveForm::end(); ?>
                </div>
            </div>
        </div>
    </div>
    <div class="modal fade" id="NewSecretModal" role="dialog">
        <div class="modal-dialog" style="width: 80%">
            <div class="modal-content">
                <div class="modal-header">
                    <button type="button" class="close" data-dismiss="modal" id="close1">&times;</button>
                    <h4 class="modal-title"><?= controllers::t('menu', 'Add Secret') ?></h4>
                </div>
                <div class="modal-body">
                    <?php
                    $secret = new \app\modules\correspondence\models\CmsDocSecret();
                    $id_secret = \app\modules\correspondence\models\CmsDocSecret::find()->orderBy(['secret_id' => SORT_DESC])->one();

                    $form = ActiveForm::begin(['action' => '../secret/create'],['options' => ['autocomplete' => 'off']]);
                    ?>
                    <?= $form->field($secret, 'secret_id')->hiddenInput(['value' => $id_secret->secret_id + 1])->label(false) ?>
                    <?= $form->field($secret, 'secret_name')->textInput(['maxlength' => true]) ?>

                    <div class="form-group">
                        <?= Html::submitButton($secret->isNewRecord ? controllers::t('menu', 'Create') : controllers::t('menu', 'Edit'), ['class' => $secret->isNewRecord ? 'btn btn-success' : 'btn btn-primary']) ?>
                    </div>

                    <?php ActiveForm::end(); ?>
                </div>
            </div>
        </div>
    </div>
    <div class="modal fade" id="NewTypeModal" role="dialog">
        <div class="modal-dialog" style="width: 80%">
            <div class="modal-content">
                <div class="modal-header">
                    <button type="button" class="close" data-dismiss="modal" id="close1">&times;</button>
                    <h4 class="modal-title"><?= controllers::t('menu', 'Add Document type') ?></h4>
                </div>
                <div class="modal-body">
                    <?php
                    $type = new \app\modules\correspondence\models\CmsDocType();
                    $form = ActiveForm::begin(['action' => '../type/create', 'method' => 'post'], ['options' => ['id' => 'form',
                        'class' => 'form-horizontal', 'enctype' => 'multipart/form-data']]); ?>
                    <div class="form-group">
                        <?= $form->field($type, 'type_name')->textInput(['maxlength' => true]) ?>
                    </div>
                    <div class="form-group">
                        <?= Html::submitButton($type->isNewRecord ? controllers::t('menu', 'Create') : controllers::t('menu', 'Edit'), ['class' => $type->isNewRecord ? 'btn btn-success' : 'btn btn-primary']) ?>
                    </div>

                    <?php ActiveForm::end(); ?>
                </div>
            </div>
        </div>
    </div>
    <div class="modal fade" id="NewOrganModal" role="dialog">
        <div class="modal-dialog" style="width: 80%">
            <div class="modal-content">
                <div class="modal-header">
                    <button type="button" class="close" data-dismiss="modal" id="close1">&times;</button>
                    <h4 class="modal-title"><?= controllers::t('menu', 'Add Organization') ?></h4>
                </div>
                <div class="modal-body">
                    <div id="docFromDept">
                        <?php
                        $fromdept = new \app\modules\correspondence\models\CmsDocDept();
                        $id_dept = \app\modules\correspondence\models\CmsDocDept::find()->orderBy(['doc_dept_id' => SORT_DESC])->one();
                        $form = ActiveForm::begin(['options' => ['autocomplete' => 'off']],['action' => '../doc-dept/create']); ?>
                        <?= $form->field($fromdept, 'doc_dept_id')->hiddenInput(['value' => $id_dept->doc_dept_id + 1])->label(false) ?>

                        <?= $form->field($fromdept, 'doc_dept_name')->textInput(['maxlength' => true]) ?>

                        <?= $form->field($fromdept, 'user_id')->hiddenInput(['maxlength' => true, 'value' => null])->label(false) ?>

                        <div class="form-group">
                            <?= Html::submitButton($fromdept->isNewRecord ? controllers::t('menu', 'Create') : controllers::t('menu', 'Edit'), ['class' => $fromdept->isNewRecord ? 'btn btn-success' : 'btn btn-primary']) ?>
                        </div>

                        <?php ActiveForm::end(); ?>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="modal fade" id="NewSubTypeModal" role="dialog">
        <div class="modal-dialog" style="width: 80%">
            <div class="modal-content">
                <div class="modal-header">
                    <button type="button" class="close" data-dismiss="modal" id="close1">&times;</button>
                    <h4 class="modal-title"><?= controllers::t('menu', 'Add Subtype') ?></h4>
                </div>
                <div class="modal-body">
                    <div id="docFromDept">
                        <?php
                        $model_sub = new \app\modules\correspondence\models\CmsDocSubType();
                        $id_sub = \app\modules\correspondence\models\CmsDocSubType::find()->orderBy(['sub_type_id' => SORT_DESC])->one();
                        $form = ActiveForm::begin(['options' => ['autocomplete' => 'off']],['action' => '../sub-type/create']); ?>
                        <div class="form-group">
                            <?= $form->field($model_sub, 'sub_type_id')->hiddenInput(['value' => $id_sub->sub_type_id + 1])->label(false) ?>
                        </div>
                        <div class="form-group">
                            <?= $form->field($model_sub, 'sub_type_name')->textInput(['maxlength' => true]) ?>
                        </div>
                        <div class="form-group">
                            <?= Html::submitButton($model_sub->isNewRecord ? controllers::t('menu', 'Create') : controllers::t('menu', 'Edit'), ['class' => $model_sub->isNewRecord ? 'btn btn-success' : 'btn btn-primary']) ?>
                        </div>

                        <?php ActiveForm::end(); ?>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div class="modal fade" id="NewAddressModal" role="dialog">
        <div class="modal-dialog" style="width: 80%">
            <div class="modal-content">
                <div class="modal-header">
                    <button type="button" class="close" data-dismiss="modal" id="close1">&times;</button>
                    <h4 class="modal-title"><?= controllers::t('menu', 'Add Folder') ?></h4>
                </div>
                <div class="modal-body">
                    <div id="docFromDept">
                        <?php
                        $model_address = new \app\modules\correspondence\models\CmsAddress();
                        $year = array();
                        $yearStar = date("Y") + 543;

                        foreach (range(2500, $yearStar + 3) as $letter) {
                            $year[$letter] = $letter;
                        }
                        $form = ActiveForm::begin(['options' => ['autocomplete' => 'off']]); ?>
                        <div class="form-group">
                            <?php
                            $addressId = substr(\app\modules\correspondence\models\CmsAddress::find()->select('address_id')
                                ->orderBy(['address_id' => SORT_DESC])->limit(1)->one()->address_id,-3);
                            $addressId = (integer)$addressId+1;
                            ?>
                            <?= $form->field($model_address, 'address_id')->textInput()
                                ->input('text', ['placeholder' => 'รหัสของสันแฟ้มเช่น A101','value'=> (string)$addressId]) ?>
                            <span id="address-id-error"></span>
                        </div>
                        <div class="form-group">
                            <?= $form->field($model_address, 'address_name')->textInput(['maxlength' => true])->input('text', ['placeholder' => 'ชื่อแฟ้ม']) ?>
                        </div>
                        <div class="form-group">
                            <?= $form->field($model_address, 'address_year')
                                ->textInput(['class' => 'form-control', 'id' => 'address_year']) ?>

                        </div>
                        <div class="form-group">
                            <?= $form->field($model_address, 'sub_type_id')->dropDownList(\yii\helpers\ArrayHelper::
                            map(\app\modules\correspondence\models\CmsDocSubType::find()->asArray()->all()
                                , 'sub_type_id', 'sub_type_name'), ['prompt' => '--- กรุณาเลือกหมวดหมู่ ---'])->error(false);
                            ?>
                        </div>
                        <div class="form-group">
                            <button type="submit"
                                    class="btn btn-success" id="address-submit"><?= controllers::t('menu', 'Create') ?></button>
                        </div>

                        <?php ActiveForm::end(); ?>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        const addressIdError = "<?=controllers::t('menu','File IDs must be unique.')?>";
        function redirectDeleteRoll(id) {
            pass_id = id;

        }

        function passParamFromDept(idDoc) {
            pass_id = idDoc;

        }
    </script>

<?php

$this->registerJs(<<<JS
   $('.example').DataTable();
$(function(){
    $('#address_year').datepicker({
        autoclose: true,
        startView: 2,
        minViewMode: 2,
        maxViewMode: 2,
        format: "yyyy"
    });
});

 $('.modal').on('hidden.bs.modal', function(e)
    {
         $(':input', this).val('');
    }) ;
JS
);
?>
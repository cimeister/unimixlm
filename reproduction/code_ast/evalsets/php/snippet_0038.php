<?php
/**
 * Created by PhpStorm.
 * User: sir
 * Date: 2018/5/22
 * Time: 13:36
 */
namespace app\modelmanagement\model;

use think\exception\PDOException;
use think\Model;
class CompleteModel extends Model
{
    protected $name = 'model_complete';


    public function insertTb($param)
    {
        try {
            $result = $this->allowField(true)->save($param);
            if (false === $result) {
                return ['code' => -1, 'msg' => $this->getError()];
            } else {
                return ['code' => 1, 'data' => [], 'msg' => '添加成功'];
            }
        } catch (PDOException $e) {
            return ['code' => -1, 'msg' => $e->getMessage()];
        }
    }

    public function editTb($param)
    {
        try {
            $result = $this->allowField(true)->save($param, ['id' => $param['id']]);
            if (false === $result) {
                return ['code' => -1, 'msg' => $this->getError()];
            } else {
                return ['code' => 1, 'msg' => '编辑成功'];
            }
        } catch (PDOException $e) {
            return ['code' => 0, 'msg' => $e->getMessage()];
        }
    }

    public function deleteTb($id)
    {
        try {
            $this->where('id', $id)->delete();
            return ['code' => 1, 'msg' => '删除成功'];
        } catch (PDOException $e) {
            return ['code' => -1, 'msg' => $e->getMessage()];
        }
    }

    public function getOne($id)
    {
        $data = $this->find($id);
        return $data;
    }

    public function attributeArr($model_number_arr)
    {
        /**
         * 需求说明 ---
         * 当数组里 只有 一个值的时候
         *          返回 属于该分组的所有模型编号 并且返回该组的所有属性
         *
         * 当数组里 存在多个值的时候
         *          只返回所有分组的模型编号
         */
        $data['model_id'] = $data['attribute'] = [];
        // 获取当前启用的模型
        $version = new VersionsModel();
        $version_number = $version->statusOpen(1);

        if(sizeof($model_number_arr) == 1){
            // 获取当前模型的分组
            $group_name = $this->where(['version_number'=>$version_number,'model_id'=>$model_number_arr[0]])->value('group_name');
            // 属于该分组的所有模型编号
            $data['model_id'] = $this->where(['version_number'=>$version_number,'group_name'=>$group_name])->column('model_id');
            // 该组的所有属性
            $attribute = new CompleteGroupModel();
            $data['attribute'] = $attribute->attributeArr($group_name);
        }else{
            // 获取当前模型的所有分组
            $group_name_arr = $this->where(['version_number'=>$version_number,'model_id'=>['in',$model_number_arr]])->column('group_name');
            // 属于该分组的所有模型编号
            $data['model_id'] = $this->where(['version_number'=>$version_number,'group_name'=>['in',$group_name_arr]])->column('model_id');
        }
        return $data;
    }

    public function removeVersionsRelevance($version_number)
    {
        $this->where(['version_number'=>['eq',$version_number]])->delete();
        return ['code'=>1,'msg'=>'删除成功'];
    }

}
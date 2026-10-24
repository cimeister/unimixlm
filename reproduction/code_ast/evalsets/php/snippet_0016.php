<?php
/**
 * 工具类
 * User: cf8
 * Date: 18-3-22
 * Time: 上午10:34
 */

namespace common\components;


class Tools
{
    /**
     * 用户输入安全过滤函数
     * @param     $content
     * @param int $role
     * @return mixed
     */
    public static function check_input($content, $role = 0)
    {
        if($role){
            $content = trim(self::filter_sql($content));
        } else {
            $content = strip_tags(trim(self::filter_sql($content)));
        }
        return $content;
    }

    /**
     * 过滤SQL注入字符
     * @param $content
     * @return mixed
     */
    public static function filter_sql($content) {
        $sql = array("select", 'insert', 'from', "update", "delete", "union", "into", "load_file", "outfile", "SELECT", 'INSERT', 'FROM', "UPDATE", "DELETE", "UNION", "INTO", "LOAD_FILE", "OUTFILE", "\'", "\/\*", "\.\.\/", "\.\/");
        $content = str_replace($sql, "", $content);
        return $content;
    }

    /**
     * 检查手机号
     * @param $mobile
     * @return bool
     */
    public static function check_mobile($mobile)
    {
        if(preg_match("/^1\d{10}$/",$mobile)){
            return true;
        }else{
            return false;
        }
    }
    /**
     * 获取用户IP
     * @return string
     */
    public static function get_ip()
    {
        $ip = 'unknown';
        $headers = array('HTTP_X_REAL_FORWARDED_FOR', 'HTTP_X_FORWARDED_FOR', 'HTTP_CLIENT_IP', 'REMOTE_ADDR');
        foreach ($headers as $h){
            $ip = $_SERVER[$h];
            // 有些ip可能隐匿，即为unknown
            if ( isset($ip) && strcasecmp($ip, 'unknown') ){
                break;
            }
        }
        if( $ip ){
            // 可能通过多个代理，其中第一个为真实ip地址
            list($ip) = explode(', ', $ip, 2);
        }
        return $ip;
    }

    /**
     * 生成验证码
     * @return int
     */
    public static function genVerifyCode()
    {
        $code = rand(100000,999999);
        return $code;
    }
}
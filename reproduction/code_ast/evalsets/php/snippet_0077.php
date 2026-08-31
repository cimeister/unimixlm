<?php
namespace Admin\Controller;
use Think\Controller;
class IndexController extends Controller {
    public function indexAction(){
        if(session('loginuser')==''){
            $this->redirect('Login/index');exit;
        }
    	$this->display();
    }
    public function loginoutAction(){
        $_SESSION["loginuser"]="";
        $this->redirect('Login/index');
    }
}
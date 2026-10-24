<?php

namespace Horus\App\Controllers;

use Horus\Core\View;
use Horus\Core\Controller;

class Pagescontroller extends Controller{

    public function login(){

        View::render('Templates/header.php',[
            'title' => 'Login'
        ]);
        View::render('Partials/navbar.php');
        View::render('Welcome/index.php');
        View::render('Templates/footer.php');

    }

    public function register(){

        View::render('Templates/header.php',[
            'title' => 'Registro'
        ]);
        View::render('Partials/navbar.php');
        View::render('Welcome/index.php');
        View::render('Templates/footer.php');

    }

}
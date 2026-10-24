<?php

use Illuminate\Support\Facades\Route;

use App\Http\Controllers\ContactController;

Route::get('/', function () {
    return view('welcome');
});

Route::post("contactar" , [ContactController::class , "sendMail"])->name("sendMail");
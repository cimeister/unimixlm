<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\API\v1\WelcomeController;
use App\Http\Controllers\API\v1\CountryController;
use App\Http\Controllers\API\v1\GardenerController;
use App\Http\Controllers\API\v1\CustomerController;
use App\Http\Controllers\API\v1\LocationAreaController;
use App\Http\Controllers\API\v1\GardenersPerCountryController;


Route::resource('', WelcomeController::class, ['only' => ['index']]);

Route::resource('countries', CountryController::class, ['only' => ['index', 'show']]);

Route::resource('locations', LocationAreaController::class, ['only' => ['index', 'show']]);

Route::resource('gardenersByCountry', GardenersPerCountryController::class, ['only' => ['index', 'show']]);

Route::resource('gardeners', GardenerController::class, ['only' => ['index', 'show', 'store', 'update', 'destroy']]);

Route::resource('customers', CustomerController::class, ['only' => ['index', 'show', 'store', 'update', 'destroy']]);
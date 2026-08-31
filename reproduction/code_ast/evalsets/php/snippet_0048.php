<?php

namespace App\Http\Controllers\Backend;

use App\Models\Artistas\Artista;
use App\Models\Usuarios\Usuario;
use Illuminate\Foundation\Bus\DispatchesJobs;
use Illuminate\Routing\Controller as BaseController;
use Illuminate\Foundation\Validation\ValidatesRequests;
use Illuminate\Foundation\Auth\Access\AuthorizesRequests;
use App\Traits\Backend\Handler;

class BackController extends BaseController
{
  use AuthorizesRequests, DispatchesJobs, ValidatesRequests, Handler;

  protected function geraErros($validator){
    foreach( $validator->errors()->all() as $e ){
      $erros[] = $e;
    }
    return response()->json($erros);
  }
}
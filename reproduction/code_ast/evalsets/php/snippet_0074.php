<?php

use Illuminate\Database\Seeder;

class CategoriasTableSeeder extends Seeder
{
    /**
     * Run the database seeds.
     *
     * @return void
     */
    public function run()
    {
        $data = array (
			array ("nombre"=>"Belleza", "ruta_img"=>"http://'.config('app.subdomain').'.redlinks.com.ec/img/categorias/", 'created_at'=>Carbon\Carbon::now()->todatetimestring(), 'updated_at'=>Carbon\Carbon::now()->todatetimestring()),
			array ("nombre"=>"Restaurantes", "ruta_img"=>"http://'.config('app.subdomain').'.redlinks.com.ec/img/categorias/", 'created_at'=>Carbon\Carbon::now()->todatetimestring(), 'updated_at'=>Carbon\Carbon::now()->todatetimestring()),
			array ("nombre"=>"Compras", "ruta_img"=>"http://'.config('app.subdomain').'.redlinks.com.ec/img/categorias/", 'created_at'=>Carbon\Carbon::now()->todatetimestring(), 'updated_at'=>Carbon\Carbon::now()->todatetimestring()),
			array ("nombre"=>"Entretenimiento", "ruta_img"=>"http://'.config('app.subdomain').'.redlinks.com.ec/img/categorias/", 'created_at'=>Carbon\Carbon::now()->todatetimestring(), 'updated_at'=>Carbon\Carbon::now()->todatetimestring()),
			array ("nombre"=>"Fitness", "ruta_img"=>"http://'.config('app.subdomain').'.redlinks.com.ec/img/categorias/", 'created_at'=>Carbon\Carbon::now()->todatetimestring(), 'updated_at'=>Carbon\Carbon::now()->todatetimestring()),
			array ("nombre"=>"Servicios", "ruta_img"=>"http://'.config('app.subdomain').'.redlinks.com.ec/img/categorias/", 'created_at'=>Carbon\Carbon::now()->todatetimestring(), 'updated_at'=>Carbon\Carbon::now()->todatetimestring()),
			array ("nombre"=>"Salud", "ruta_img"=>"http://'.config('app.subdomain').'.redlinks.com.ec/img/categorias/", 'created_at'=>Carbon\Carbon::now()->todatetimestring(), 'updated_at'=>Carbon\Carbon::now()->todatetimestring())
		);
		DB::table('categorias')->insert($data);
    }
}
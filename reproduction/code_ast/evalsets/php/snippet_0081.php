<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('Clientes', function (Blueprint $table) {
            //Aqui deve colocar todos os campos que deseja
            $table->id();
            $table->string('nome');
            $table->string('endereco');
            $table->string('email');
            $table->bigInteger('telefone');
            $table->charset = 'utf8';
            $table->collation = 'utf8_unicode_ci';
            $table->timestamps();
            //[executa as migrações] php artisan migrate
            //[Carrega todo o laravel na memoria] php artisan tinker
            //[Adiciona uma nova linha na tabela] App/Models/[nomeTabela]::create('coluna1' => 'valor1', 'colunaX' => 'valorX')
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('Clientes');
    }
};
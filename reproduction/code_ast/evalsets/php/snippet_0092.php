<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Emails extends Model
{
    protected $table        = 'emails';
    protected $primaryKey   = 'id';
    protected $fillable     = [
        'code', 
        'host', 
        'port', 
        'username', 
        'password', 
        'encryptation', 
        'from', 
        'fromname', 
        'subject', 
        'template', 
        'status',
        'created_at', 
        'updated_at',
        'user_id',
        'id_user'
    ];
    protected $hidden = [
        'created_at', 
        'updated_at',
        'user_id',
        'id_user'
    ];

    public function user() {
        return $this->hasOne('App\Models\Settings\User');
    }

}
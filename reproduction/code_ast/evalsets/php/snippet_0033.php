<?php

namespace App;

use Illuminate\Database\Eloquent\Model;

class Grade extends Model
{

    /**
     * The attributes that are mass assignable.
     *
     * @var array
     */
    protected $fillable = [
        'user_id', 'subject', 'grade',
    ];

    /**
     * Relations
     */
    public function user() {
        return $this->belongsTo('App\User');
    }
    public function subject() {
        return $this->belongsTo('App\Subject');
    }
}
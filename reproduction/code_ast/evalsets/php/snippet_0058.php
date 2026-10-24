<?php namespace App;

use Illuminate\Database\Eloquent\Model;

class jadwal_pemeriksaan extends Model {

	protected $table = 'tbl_permohonan';
	protected $primaryKey = 'no_permohonan';
	public $timestamps = false;

	public static $rules = array(
		'layername'=>'required|min:3',
		'layerurl' => 'required',
		'orderlayer'=>'numeric|required',
		'level' => 'required',
	);

}
<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class Investigacion extends Model
{
    use HasFactory;

    protected $table = "investigaciones";
    public $fillable = ['titulo', 'resumen', 'fecha_publicacion', 'escuela_id', 'sublinea_id', 'estado_id'];

    public $dates = [
        'fecha_publicacion'
    ];

    public function escuela()
    {
        return $this->belongsTo(Escuela::class);
    }

    public function estado()
    {
        return $this->belongsTo(EstadoInvestigacion::class);
    }

    public function presupuestos()
    {
        return $this->belongsToMany(Financiador::class, 'financiador_investigacion')
            ->withPivot(['presupuesto']);
    }

    public function responsables()
    {
        return $this->belongsToMany(Investigador::class, 'investigador_investigacion')
            ->with('docente', 'estudiante')
            ->withPivot(['es_responsable']);
    }

    public function sublinea()
    {
        return $this->belongsTo(SublineaInvestigacion::class);
    }

}
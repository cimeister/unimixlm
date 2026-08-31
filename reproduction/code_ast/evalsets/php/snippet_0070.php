@extends('layouts.common')
@section('headers')
<style>
    td
    {
        text-align:center;
    }
    .oversized-col
    {
        max-width:150px;
        overflow:hidden;
    }
</style>
@endsection
@section('content')
<!-- Page Heading -->
<header class="bg-white shadow">
    <div class="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <h2 class="font-semibold text-xl text-gray-800 leading-tight">
            Transportistas.
        </h2>
    </div>
</header>

<!-- Page Content -->

<div class="py-12">
<div class="max-w-2xl mx-auto sm:px-6 lg:px-8">
<div class="bg-white overflow-hidden shadow-sm sm:rounded-lg">
<div class="p-6 bg-white border-b border-gray-200">

        <div class="row">
            <div class="col-lg-6 controlDiv">
                <label class="form-label">Nombre:</label>
                <input type="text" class="form-control" id="txtNombre" name="txtNombre" value="" placeholder="transportista">       
            </div>
            <div class="col-lg-3 controlDiv">
                <button class="btn btn-success" style="position:relative; top:30px" type='button' onclick="agregar()">Agregar <i class="fas fa-plus"></i></button>
            </div>
        </div>
            
        <h5 class="separtor">Lista:</h5>

        <table class="table table-sm table-striped table-bordered table-hover">
            <thead>
                <tr>
                    <th style = "text-align:center" scope="col">#</th>
                    <th style = "text-align:center" scope="col">Nombre</th>
                </tr>
            </thead>
            <tbody>
                @foreach ($carriers as $carrier)
                <tr>
                    <td>{{ $loop->iteration }}</td>
                    <td>{{ $carrier->name }}</td>
                </tr>
                @endforeach
            </tbody>
        </table>
</div>
</div>
</div>
</div>

@endsection
@section('scripts')
<script>

function agregar()
{
    if($("#txtNombre").val().trim() == "")
    {
        showModal("Validación", "Transportista vacío.");
        return;
    }
    $.ajax({url: "/int/catalog/carriers_add/"+$("#txtNombre").val().trim(),context: document.body}).done(function() 
        {
            showModal("Notificación","Agregado!");
            location.reload();
        });
}

</script>
@endsection
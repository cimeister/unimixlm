@extends('app')
 
@section('content-miembros')
<br>
<div class="container">
  <div class="row">
    <div class="col-xs-12">
      <h1 class="text-center color-azul"><i class="fa fa-money" aria-hidden="true"></i>&nbsp; Listado de Mis Encuestas Pendientes&nbsp; <i class="fa fa-money" aria-hidden="true"></i></h1>
            <ul class="nav nav-pills">
                <li class="active"><a data-toggle="pill" href="#atencion"><i class="fa fa-list" aria-hidden="true"></i>&nbsp;Encuestas Pendientes</a></li>
            </ul>
          
          <div class="tab-content">
           
             <div id="atencion" class="tab-pane fade in active">
                <div class="panel panel-default">

                  <div class="panel-heading fondo-naranja-oscuro color-blanco">
                    <h2 class="panel-title text-center" style="font-weight:bold;">
                        <i class="fa fa-list" aria-hidden="true"></i>
                        &nbsp; Mis Encuestas Pendientes&nbsp; <i class="fa fa-list" aria-hidden="true"></i>
                    </h2>
                  </div>
                      <div class="panel-body">

                        @if(COUNT($encuestas_pendientes) > 0) 
                          <div class="table-responsive" id="lista-encuestas-pendientes">
                            @include('encuesta.encuestas-pendientes-table')
                          </div> 
                        @else 
                          {{-- <ul class="list-group">
                            <li class="list-group-item active"><i class="fa fa-lock" aria-hidden="true"></i>&nbsp;Compras en Atención&nbsp;<span class="badge">0</span></li>
                            <li class="list-group-item"> --}}
                                <center>
                                    <img src="/images/cero.png" title="No Existen Encuestas Pendientes">
                                    <p class="color-azul cuenta-usuario-row"><b>No posee ninguna encuesta pendiente por realizar.</b></p>
                                    <p class="color-azul cuenta-usuario-row">Cuando Usted tenga encuestas pendientes; éstas aparecerán aquí.</p>
                                </center>
                           {{--  </li> --}}
                          {{-- </ul> --}}
                        @endif 
                      </div>
                </div>
             </div>
          </div>
      </div>
    </div>
  </div>
</div>
@endsection
{{-- $compras_encuestas;
        $compras_cerradas;
        $compras_anuladas;
                       <div class="table-responsive" id="lista-encuesta">
                          @include('miscompras.compras-encuesta-table')
                       </div>
                        <div class="table-responsive" id="lista-cerradas">
                          @include('miscompras.compras-cerradas-table')
                       </div>  
                        <div class="table-responsive" id="lista-anuladas">
                          @include('miscompras.compras-anuladas-table')
                       </div>  --}}
@section('scripts')
<script>
 $(window).on('hashchange', function() {
        if (window.location.hash) {
            var page = window.location.hash.replace('#', '');
            if (page == Number.NaN || page <= 0) {
                return false;
            }else{
                getData(page);
            }
        }
    });
$(document).ready(function()
{
      $(document).on('click', '#tbl-encuestas-pendientes .pagination a',function(event)
    { 

        //alert('hizo click');

        $('#tbl-encuestas-pendientes li').removeClass('active');
        $(this).parent('li').addClass('active');
        event.preventDefault();
        var myurl = $(this).attr('href');
       var page=$(this).attr('href').split('page=')[1];
       //alert(page);
       getData(page);
    });
});




 function getData(page){
        $.ajax(
        {
            url: '?page=' + page,
            type: "get",
            //data : "tipo=1",
            datatype: "html",
            // beforeSend: function()
            // {
            //     you can show your loader 
            // }
        })
        .done(function(data)
        {
            //console.log(data);
            
            $("#lista-encuestas-pendientes").empty().html(data);

            //location.hash = page;
        })
        .fail(function(jqXHR, ajaxOptions, thrownError)
        {
              alert('No response from server');
        });
}
</script>
@endsection
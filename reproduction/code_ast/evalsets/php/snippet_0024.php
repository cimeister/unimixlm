<h1><a href="{{ route('Create') }}">Posta publicação</a></h1>

<hr>
<h2>Publicações</h2>

@foreach($publicacao as $publicacao)
    <div style="background-color: #cdcdcd;">
        <label></label>{{$publicacao->nome}}</label><br><hr>
        <label><img src="/uploads/{{ $publicacao->image }}"></label><br>
        <label>{{$publicacao->descricao}}</label><br>

        <a href="{{ route('Delete',$publicacao->id) }}">Remover publicação</a><br>
    </div>
@endforeach
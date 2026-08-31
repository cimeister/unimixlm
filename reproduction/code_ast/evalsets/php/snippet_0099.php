

 @extends('adminlte::page')

@section('title','Data Anggota')

@section('content_header')

<br>
@endsection


@section('content')
<div class="container">
    <div class="row justify-content-center">
        <div class="col-md-12">
            <div class="card">
                <div class="card-header">
                    @include('layouts._flash')
                   <b>Data Anggota</b>
                    <a href="{{route('anggota.create')}}" class="btn btn-sm btn-outline-primary float-right"><i>Tambah Angggota</i></a>
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table" id="petugas">

                            <thead>
                            <tr>
                                <th><i>ID Petugas</i></th>
                                <th><i>Nama Petugas</i></th>
                                <th><i>Jabatan</i></th>
                                <th><i>No Telepon</i></th>
                                <th><i>Alamat</i></th>
                                <th><i>Action</i></th>


                            </tr>
                            </thead>
                            <tbody>
                            @php $no=1; @endphp
                            @foreach ($petugas as $data)
                             <tr>
                                <td>{{$no++}}</td>
                                <td>{{$data->nama_petugas}}</td>
                                <td>{{$data->jabatan_petugas}}</td>
                                <td>{{$data->no_telp}}</td>
                                <td>{{$data->alamat}}</td>



                                <td>
                                    <form action="{{route('petugas.destroy',$data->id)}}" method="post">
                                       @method('delete')
                                       @csrf
                                       <a href="{{route('petugas.edit',$data->id)}}" class="btn btn-outline-info">Edit</a>
                                       <button type="submit" class="btn btn-outline-danger" onclick="return confirm('Apakah anda yakin menghapusnya')">HAPUS</button>
                                       </form>
                                </td>
                             </tr>
                            @endforeach
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>



@endsection

@section('css')
    <link rel="stylesheet" type="text/css" href="{{asset('DataTables/datatables.min.css')}}">
    
@endsection

@section('js')
<script  type="text/javascript" src="{{asset('DataTables/datatables.min.js')}}"> </script>
<script>
    $(document).ready(function(){
        $('#petugas').DataTable();
    });
</script>
<script src="//cdn.jsdelivr.net/npm/sweetalert2@11"></script>
<script src="{{asset('js/sweetalert2.js')}}"></script>
<script src="{{asset('js/delete.js')}}"></script>

@endsection
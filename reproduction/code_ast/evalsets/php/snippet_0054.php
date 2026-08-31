@extends('admin.layouts.dashboard')

@section('content')



    {{-- VALIDACIONES-RESPUESTA--}}
    @include('admin.layouts.success')   {{-- SAVE --}}
    @include('admin.layouts.danger')    {{-- EDITAR --}}
    @include('admin.layouts.delete')    {{-- DELELTE --}}
    {{-- VALIDACIONES-RESPUESTA --}}
    
    @if ($errors->any())
        <div class="alert alert-danger">
            <ul>
                @foreach ($errors->all() as $error)
                    <li>{{$error}}</li>
                @endforeach
            </ul>
        </div>
    @endif
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-10">
                <div class="card">
                    <div class="card-header">Editar Movimiento</div>
    
                    <div class="card-body">
                    <form  method="POST"   action="{{ route('detailvouchers.update',$var->id) }}" enctype="multipart/form-data" >
                        @method('PATCH')
                        @csrf()
                        
                        <input type="hidden" name="id_account" value="{{$id_account ?? -1}}" readonly>
                       
                        <div class="form-group row">
                            <label for="account" class="col-md-3 col-form-label text-md-right">Cuenta:</label>

                            <div class="col-md-6">
                                <input id="account" type="text" class="form-control @error('account') is-invalid @enderror" readonly name="account" value="{{ $var->accounts['code_one'] }}.{{ $var->accounts['code_two'] }}.{{ $var->accounts['code_three'] }}.{{ $var->accounts['code_four'] }}.{{ $var->accounts['code_five'] }} - {{ $var->accounts['description'] }}" required autocomplete="account" >

                                @error('account')
                                    <span class="invalid-feedback" role="alert">
                                        <strong>{{ $message }}</strong>
                                    </span>
                                @enderror
                            </div>
                            <a href="{{ route('detailvouchers.selectaccount',[$coin,$var->id_header_voucher,$var->id]) }}" title="Editar"><i class="fa fa-eye"></i></a>  
                                   
                        </div>
                       
                        <div class="form-group row">
                            <label for="rate" class="col-md-3 col-form-label text-md-right">Tasa:</label>
                            <div class="col-md-4">
                                <input id="rate" type="text" class="form-control @error('rate') is-invalid @enderror" name="rate" value="{{ $var->tasa ?? $bcv }}" required autocomplete="rate">
                                @error('rate')
                                    <span class="invalid-feedback" role="alert">
                                        <strong>{{ $message }}</strong>
                                    </span>
                                @enderror
                            </div>
                            <!--<a href="#" onclick="refreshrate()" title="tasaactual"><i class="fa fa-redo-alt"></i></a> --> 
                            <label  class="col-md-2 col-form-label text-md-right h6">Tasa actual:</label>
                            <div class="col-md-2 col-form-label text-md-left">
                                <label for="tasaactual" id="tasaacutal">{{ number_format($bcv, 2, ',', '.')}}</label>
                            </div>
                        </div>


                        <div class="form-group row">
                            <label id="coinlabel" for="coin" class="col-md-3 col-form-label text-md-right">Tipo:</label>

                            <div class="col-md-3">
                                <select class="form-control" name="type" id="type">
                                    <option value="debe">debe</option>
                                    @if($var->haber != 0)
                                        <option selected value="haber">haber</option>
                                    @else 
                                        <option value="haber">haber</option>
                                    @endif
                                </select>
                            </div>
                            <label id="coinlabel" for="coin" class="col-md-3 col-form-label text-md-right">Moneda:</label>

                            <div class="col-md-3">
                                <select class="form-control" name="coin" id="coin">
                                    <option value="bolivares">Bolívares</option>
                                    @if($coin == 'dolares')
                                        <option selected value="dolares">Dolares</option>
                                    @else 
                                        <option value="dolares">Dolares</option>
                                    @endif
                                </select>
                            </div>
                        </div>
                       <div class="form-group row">
                            <label for="amount" class="col-md-3 col-form-label text-md-right">Monto:</label>
                            <div class="col-md-4">
                                @if ($var->debe != 0)
                                    <input id="amount" type="text" class="form-control @error('amount') is-invalid @enderror" name="amount" value="{{ number_format($var->debe ?? 0, 2, ',', '.') }}" required autocomplete="amount">
                                @else
                                    <input id="amount" type="text" class="form-control @error('amount') is-invalid @enderror" name="amount" value="{{ number_format($var->haber ?? 0, 2, ',', '.') }}" required autocomplete="amount">
                                @endif
                              @error('amount')
                            <span class="invalid-feedback" role="alert">
                                <strong>{{ $message }}</strong>
                            </span>
                            @enderror
                            </div>
                        </div>

                        <br>
                        
                        <div class="form-group row">
                            <div class="form-group col-sm-2">
                            </div>
                            <div class="form-group col-sm-4">
                                <button type="submit" class="btn btn-success btn-block"><i class="fa fa-send-o"></i>Actualizar</button>
                            </div>
                            <div class="form-group col-sm-4">
                                <a href="{{ route('detailvouchers.create',[$coin,$var->id_header_voucher]) }}" name="danger" type="button" class="btn btn-danger btn-block">Cancelar</a>
                            </div>
                        </div>

                    </form>
               
                </div>
            </div>
        </div>
    </div>
</div>
@endsection
@section('validacion')
    
    <script>
       
        $(document).ready(function () {
            $("#reference").mask('000000000000000', { reverse: true });
            
        });
        $(document).ready(function () {
            $("#rate").mask('000.000.000.000.000.000,0000', { reverse: true });
            
        });
        $(document).ready(function () {
            $("#amount").mask('000.000.000.000.000.000,00', { reverse: true });
            
        });
       

    </script>
@endsection
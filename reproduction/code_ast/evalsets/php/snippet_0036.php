@extends('layout.master')
@section('title')
    {{translateUrl()}}
@endsection

@section('content')
<div class='row'>
    <div class='col-lg-1 col-md-1 col-xs-12 pull-right'>
        <button class="btn btn-primary searchButton">
            <span class="fa fa-search" aria-hidden="true"></span>
        </button>
    </div>
    <div class='col-lg-6 col-md-6 col-xs-12'>
        <div class="control-label col-md-3 col-sm-3 col-xs-12">
            <select name='search[field]' class="form-control col-md-7 col-xs-12">
                <option value=''>Field</option>
                {!! $field !!}
            </select>
        </div>
        <div class="control-label col-md-2 col-sm-2 col-xs-12">
            <select name='search[operator]' class="form-control col-md-7 col-xs-12">
                <option value="equal">{{trans('form.equal')}}</option>
                <option value="not_equal">{{trans('form.not_equal')}}</option>
            </select>
        </div>
        <div class="control-label col-md-5 col-sm-5 col-xs-12">
            <input name="search[keyword]" placeholder="Keyword" type='text' class="form-control col-md-7 col-xs-12">
        </div>
    </div>
</div>
<table id="dTable" class="table table-striped table-responsive table-bordered">
<div class='row'>
    <div class='col-lg-1 col-md-1 col-xs-12 pull-right'>
        <a href="{{route('admin.management.menu.form')}}">
            <button id='addModalButton' class="btn btn-primary">
                <span class="fa fa-plus-square" aria-hidden="true"></span>
            </button>
        </a>
    </div>    
</div>
    <thead>
        <tr>
            <th>Name</th>
            <th>Url</th>
            <th>Action</th>
        </tr>
    </thead>
    <tbody>
        
    </tbody>
</table>
@endsection

@section('script')
<script>
    var target = '#dTable';    
    $(document).ready(function(){
        dt = new DatatableCustomClass(`{{route('admin.management.menu.datatables')}}`,`{{route('admin.management.menu.datatablesColumn')}}`,target,$('[name*=search]'));
        dt.renderDatatables()
        $('.searchButton').on('click',function(){
            dt.renderDatatables();
        });
        $('[name*=search]').on('change',function(){
            dt.renderDatatables();
        });
        $('#confrimationUpdateYes').on('click',function(){
            var uuid = $(this).attr('data-id')
            window.location.href = `{{route('admin.management.menu.form')}}/${uuid}`
        })
        $('#confrimationDeleteYes').on('click',function(){
            var deleteMenu = new ajax(`{{route('admin.management.menu.delete')}}`,{uuid : $(this).attr('data-id')},{},'DELETE').execAjax();            
                deleteMenu.getResult().then(function(res){
                PNotify.removeAll();
                var message = res.body.content.message;
                var notice = new PNotify({
                    title: 'Click to Close Notice',
                    text: `${message}`,
                    buttons: {
                        closer: false,
                        sticker: false
                    },
                    type : 'success'
                });
                notice.get().click(function() {
                    notice.remove();
                });                
            },function(err){
                PNotify.removeAll();
                var message = '';
                    $.each(err.responseJSON,function(key,item){
                    message = message.concat(item[0],' <br>')
                })
                var notice = new PNotify({
                    title: 'Click to Close Notice',
                    text: `${message}`,
                    buttons: {
                        closer: false,
                        sticker: false
                    }
                });
                notice.get().click(function() {
                    notice.remove();
                });
            })
            dt.renderDatatables();            
        })
})

</script>
@endsection
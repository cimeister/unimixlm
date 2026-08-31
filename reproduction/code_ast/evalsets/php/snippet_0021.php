@extends ('admin.master')

@section("content")
    <div class="content-header">
        <div class="container-fluid">
            <div class="row mb-2">
                <h1 class="m-0 text-dark">CẬP NHẬT TRƯỞNG PHÒNG</h1>
            </div>
        </div>
    </div>
<div class="card card-primary">
  <div class="card-header">
    <a class="btn btn-primary" href="{{route('room.index')}}">
        <i class="fas fa-arrow-circle-left">
        </i>Trở Lại</a>
  </div>
  <form role="form" action="{{route('room.handleUpdate')}}" method="POST">
    @csrf
    <div class="card-body">
    <input type="hidden" class="form-control" id="id" name="id" value='{{$edit->id}}'>
        <div class="form-group">
                <label for="exampleInputPassword1">Nhân Viên</label>
                <select id="id_employee" name="employee" class="form-control">
                @foreach($employee as $employee)
                    <option value='{{$employee ->id}}'>{{$employee->name}} </option>
                @endforeach
                </select>
        </div>
    </div>
    <div class="card-footer">
          <button type="submit" class="btn btn-primary">Cập Nhật</button>
    </div>
  </form>
</div>
@endsection
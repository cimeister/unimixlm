@extends('user-views.layouts.app')


@section('name')
Detail Data Pelatihan
@endsection

@section('content')


<main class="page-content">
    <div class="container-fluid">
        <div class="page-breadcrumb d-none d-sm-flex align-items-center mb-3">
            <div class="breadcrumb-title pe-3">Program Pelatihan</div>
            <div class="ps-3">
                <nav aria-label="breadcrumb">
                    <ol class="breadcrumb mb-0 p-0">
                        <li class="breadcrumb-item"><a href="javascript:;"><i class="lni lni-database"></i></a>
                        </li>
                        <li class="breadcrumb-item active" aria-current="page">Detail Data</li>
                    </ol>
                </nav>
            </div>

        </div>
        <hr>
    </div>

    <div class="container-fluid">
        <div class="row">
            <div class="col-lg-10 mx-auto">
                <div class="card">
                    <div class="card-body">
                        <div class="border p-3 rounded">
                                <div class="row g-3">
                                    <div class="col-6">
                                        <label class="form-label mr-1" for="nama_pelatihan">Nama Pelatihan</label>
                                        <input type="text" class="form-control" placeholder="Nama Program Pelatihan"
                                            name="nama_pelatihan" value="{{ $item->nama_pelatihan }}" readonly>
                                    </div>
                                    <div class="col-6">
                                        <label class="form-label mr-1" for="jenis_pelatihan">Jenis Pelatihan</label>
                                            <input type="text" class="form-control" placeholder="Jenis Program Pelatihan"
                                            name="jenis_pelatihan" value="{{ $item->jenis_pelatihan }}" readonly>
                                    </div>
                                    <div class="col-4">
                                        <label class="form-label mr-1" for="penyelenggara">Penyelenggara</label>
                                        <input type="text" class="form-control" placeholder="Penyelenggara Program Pelatihan"
                                            name="penyelenggara" value="{{ $item->penyelenggara }}" readonly>
                                    </div>
                                    <div class="col-4">
                                        <label class="form-label mr-1" for="contact_info">Contact Info</label>
                                        <input type="text" class="form-control" placeholder="Penyelenggara Contact"
                                            name="contact_info" value="{{ $item->contact_info }}" readonly>
                                    </div>
                                    <div class="col-4">
                                        <label class="small mb-1 mr-1" for="status_wajib">Status Wajib</label>
                                        <input type="text" class="form-control" placeholder="Penyelenggara Contact"
                                            name="contact_info" value="{{ $item->status_wajib }}" readonly>
                                    </div>
                                    <div class="col-4">
                                        <label class="small mb-1 mr-1" for="periode_awal">Periode Mulai</label>
                                        <input type="text" class="form-control" placeholder="Penyelenggara Contact"
                                            name="contact_info" value="{{ $item->periode_awal }}" readonly>
                                    </div>
                                    <div class="col-4">
                                        <label class="small mb-1 mr-1" for="periode_akhir">Periode Selesai</label>
                                        <input type="text" class="form-control" placeholder="Penyelenggara Contact"
                                        name="contact_info" value="{{ $item->periode_akhir }}" readonly>
                                    </div>
                                    <div class="col-4">
                                        <label class="small mb-1 mr-1" for="biaya">Biaya</label>
                                        <input type="text" class="form-control" placeholder="Penyelenggara Contact"
                                        name="contact_info" value="Rp. {{ number_format($item->biaya) }}" readonly>
                                    </div>
                                    <div class="col-12">
                                        <label class="small mb-1 mr-1" for="keterangan">Keterangan</label>
                                        <textarea type="text" class="form-control" placeholder="Penyelenggara Contact"
                                        name="contact_info" value="{{ $item->keterangan }}" readonly>{{ $item->keterangan }}</textarea>
                                    </div>
                                    <div class="col-12">
                                        <a href="{{ route('program-pelatihan.index') }}" class="btn btn-primary px-4" type="button">Kembali</a>
                                    </div>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    </div>
</main>




@endsection
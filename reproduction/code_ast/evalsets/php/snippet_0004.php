<section id="inner-title" class="inner-title">
  <div class="container">
    <div class="row">
      <div class="col-md-6 col-lg-6">
        <h2>Lihat Booking Pemeriksaan</h2>
      </div>
      <div class="col-md-6 col-lg-6">
        <div class="breadcrumbs">
          <ul>
            <li>Current Page:</li>
            <li><a href="index.php">Home</a></li>
            <li><a href="lihat_booking.php">Lihat Booking Pemeriksaan</a></li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</section>

<section id="section16" class="section16">
  <div class="container">
    <div class="col-lg-12">
          <a href="<?php echo base_url('pemeriksaan/create')?>">
              <button class="btn btn-primary btn-sm"> <i class="fa fa-plus"></i> Tambah </button>
          </a>
            <p></p>
     <br>
      <table class="table table-bordered table-striped">
        <thead>
          <tr class="bg-primary">
       
            <th>No</th>
            <th>Nama Pasien</th>
            <th>No Handphone</th>
            <th>Email</th>
            <th>Keluhan</th>
            <th>Spesialis</th>
            <th>Dokter</th>
            <th>Tanggal Booking</th>
            <th>Jam Booking</th>
            <th>Status</th>
            <th>Action</th>
          </tr>
        </thead>

        <tbody>
       
          <tr>
          
            <td> </td>
            <td> </td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
          
							<td> </td>
              <td>
					<a href="#">
						<button class="btn btn-success btn-sm"> <i class="fa fa-edit"></i> Edit </button>
					</a>
					<a href="#"onclick="return confirm('Yakin hapus data?')">
						<button class="btn btn-danger btn-sm"> <i class="fa fa-trash"></i> Hapus </button>
					</a>
				</td>
          </tr>
  
        </tbody>
     
      </table>
    </div>
  </div>
</section>

<?php
$template_header;
?>
<body>
	<div class="wrapper h-100">
		<div class="container-fluid">
			<div class="row">
				<?php $this->load->view("admin/template/a_t_sidebar"); ?>
				<?php $this->load->view("admin/template/a_t_navbar", $nav); ?>
				<div class="col-12 text-center">
					<div class="container-fluid p-2 py-5 p-sm-5">
						<?php if ($this->session->flashdata("alert")): ?>
							<?php $alert = $this->session->flashdata("alert"); ?>
							<div class="alert alert-<?=$alert[0]?> alert-dismissible">
								<?=$alert[1]?>
								<button type="button" class="close" data-dismiss="alert">
									&times;
								</button>
							</div>
						<?php endif; ?>
						<div class="row">
							<div class="col-12 col-sm-6 text-left">
								<h2>View Type #<?=$row_info["type_id"]?></h2>
							</div>
							<div class="col-12 col-sm-6 text-right">
								<a class="btn btn-primary" href="<?=base_url()?>admin/types_edit?id=<?=$row_info['type_id']?>">
									<i class="fa fa-pencil p-1" aria-hidden="true"></i> Update
								</a>
							</div>
						</div>
						<div class="row view_container">
							<div class="col-12 col-md-3">
								<img class="img-responsive img_view img_zoomable" src="<?php
								if (!empty($row_info["img"])) {
									echo base_url(). 'uploads/types/type_'. $row_info["type_id"] .'/'. $row_info["img"];
								} else {
									echo base_url(). "assets/img/no_img.png";
								}
								?>">
							</div>
							<div class="col-12 col-md-9">
								<div class="row mt-2">
									<div class="col-12">
										<label>Type Name:</label><br>
										<?=$row_info["name"]?>
									</div>
									<div class="col-12">
										<label>Description:</label><br>
										<?=$row_info["description"]?>
									</div>
									<div class="col-12 col-md-6">
										<label>Price Range:</label><br>
										PHP <?=$row_info["price_range"]?>
									</div>
									<div class="col-12 col-md-6">
										<label>Featured:</label><br>
										<?=($row_info["featured"] == 1 ? "YES" : "NO")?>
									</div>
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
</body>
<script type="text/javascript">
</script>
</html>
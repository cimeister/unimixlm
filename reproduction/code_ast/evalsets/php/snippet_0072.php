<?php $this->load->view('includes/header'); ?>	

	<!-- Main container start -->

	<section id="main-container">
		<div class="container">
			<div class="error-page text-center">
				<div class="error-code">
					<strong>404</strong>
				</div>
				<div class="error-message">
					<h3>Oops... Page Not Found!</h3>
				</div>
				<div class="error-body">
					Try using the button below to go to main page of the site <br/>
					<a href="<?php echo base_url();?>" class="btn btn-primary solid blank"><i class="fa fa-arrow-circle-left">&nbsp;</i> Go to Home</a>
				</div>
			</div>
		</div>
	</section><!--/ Main container end -->	
<?php $this->load->view('includes/footer');?>
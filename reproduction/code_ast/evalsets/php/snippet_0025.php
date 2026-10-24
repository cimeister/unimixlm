<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <title>Premlal Jewelleries | Login Screen</title>
  <!-- Tell the browser to be responsive to screen width -->
  <meta content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" name="viewport">
  <!-- Bootstrap 3.3.6 -->
  <link rel="stylesheet" href="<?php echo base_url('backendassets'); ?>/bootstrap/css/bootstrap.min.css">
  <!-- Font Awesome -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.5.0/css/font-awesome.min.css">
  <!-- Ionicons -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/ionicons/2.0.1/css/ionicons.min.css">
  <!-- Theme style -->
  <link rel="stylesheet" href="<?php echo base_url('backendassets'); ?>/dist/css/AdminLTE.min.css">

  <!-- HTML5 Shim and Respond.js IE8 support of HTML5 elements and media queries -->
  <!-- WARNING: Respond.js doesn't work if you view the page via file:// -->
  <!--[if lt IE 9]>
  <script src="https://oss.maxcdn.com/html5shiv/3.7.3/html5shiv.min.js"></script>
  <script src="https://oss.maxcdn.com/respond/1.4.2/respond.min.js"></script>
  <![endif]-->
</head>
<body class="hold-transition lockscreen">
<!-- Automatic element centering -->
<div class="lockscreen-wrapper">
	<div id="LoginMsg" style="display:none; border-color:#fff ; background-color:#fff !important; " class="alert alert-success text-center"></div>
  <div class="lockscreen-logo">
    <!--<a href="javascript:void(0);"><b>Premlal Jewellery</b></a>-->
     <img src="<?php echo base_url('frontendassets');?>/images/plogo.png">
  </div>
  <!-- User name -->
  <div class="lockscreen-name" style="color:white">Staff Login</div>

  <!-- START LOCK SCREEN ITEM -->
	<div class="lockscreen-item" id="EmailScreen">
		<form class="lockscreen-credentials" onsubmit="return false;">
			<div class="input-group">
				<input type="Email" class="form-control" placeholder="Enter Your Login Email" id="LoginEmail" style="padding:0px !important">
				<div class="input-group-btn">
					<button type="button" class="btn" id="EmailSend"><i class="fa fa-arrow-right text-muted"></i></button>
					<img style="height:25px; display:none" src="<?php echo base_url(); ?>/backendassets/dist/img/tinyloading.gif" id="EmailLoading" />
				</div>
			</div>
		</form>
	</div>
  <div class="lockscreen-item" style="display:none" id="PasswordScreen">
    <!-- lockscreen image -->
    <div class="lockscreen-image">
      <img src="" alt="User Image" id="UserImage" style="height:128;width:128;">
    </div>
    <!-- /.lockscreen-image -->

    <!-- lockscreen credentials (contains the form) -->
    <form class="lockscreen-credentials" onsubmit="return false;">
      <div class="input-group">
        <input type="password" class="form-control" placeholder="Enter Your Password" id="LoginPassword">

        <div class="input-group-btn">
          <button type="button" class="btn" id="PasswordSend"><i class="fa fa-arrow-right text-muted"></i></button>
		  <img style="height:25px; display:none" src="<?php echo base_url(); ?>/backendassets/dist/img/tinyloading.gif" id="PassLoading" />
        </div>
      </div>
    </form>
    <!-- /.lockscreen credentials -->

  </div>
  <!-- /.lockscreen-item -->
  <div class="help-block text-center" style="color:white">
    Enter your Email ID to promot for password
  </div>
  <div class="text-center">
    <!--<a href="login.html">Forgot Your Password ?</a>-->
  </div>
  <div class="lockscreen-footer text-center" style="color:white">
    Copyright &copy; 2020 <b><a href="http://localhost/Premlal/EcoDashAdmin" class="" style="color:orange"><NAME></a></b><br>
    All rights reserved
  </div>
</div>
<!-- /.center -->

<!-- jQuery 2.2.3 -->
<script src="<?php echo base_url('backendassets'); ?>/plugins/jQuery/jquery-2.2.3.min.js"></script>
<!-- Bootstrap 3.3.6 -->
<script src="<?php echo base_url('backendassets'); ?>/bootstrap/js/bootstrap.min.js"></script>
<script>var base_url = "<?php echo base_url(); ?>";</script>
</body>
</html>
<script>
	$(function(){
		$('#EmailSend').click(function(){
			$('#LoginMsg').hide();
			if($.trim($('#LoginEmail').val())!='')
			{
				$.ajax({
					url:"",
					method:'POST',
					data:{'LoginEmail':$.trim($('#LoginEmail').val())},
					beforesend:function(){
						$('#EmailSend').hide();
						$('#EmailLoading').show();
					},
					complete:function(){
						$('#EmailLoading').hide();
						$('#EmailSend').show();
					},
					success:function(data){
						data = $.parseJSON(data);
						if(data!='')
						{
							if(data.response == 1)
							{
								data.Img = $.trim(data.Img);
								if(data.Img!='')
									$('#UserImage').attr('src',base_url+data.Img);
								else
									$('#UserImage').hide();
								$("#EmailScreen").animate({left:800, opacity:"hide"}, 800);
								setTimeout(function(){$('#PasswordScreen').animate({right:10, opacity:"show"},200);},700)
							}
							else
							{
								$('#LoginMsg').html('<b style="color:#444444">Invalid Email ID</b>');
								$('#LoginMsg').show();
							}
						}
					}
				});
			}
		});
		
		$('#PasswordSend').click(function(){
			if($.trim($('#LoginPassword').val())!='' && $.trim($('#LoginEmail').val())!='')
			{
				$.ajax({
					url:"",
					method:'POST',
					data:{'Email':$.trim($('#LoginEmail').val()),'Password':$.trim($('#LoginPassword').val())},
					beforesend:function(){
						$('#PasswordSend').hide();
						$('#PassLoading').show();
					},
					complete:function(){
						$('#PassLoading').hide();
						$('#PasswordSend').show();
					},
					success:function(data){
						data = $.parseJSON(data);
						if(data == 1)
							window.location = base_url+'EcoDashAdmin/StaffDashboard';
						else
						{
							resetAnim();
							$('#LoginMsg').html('<b style="color:#444444">Invalid Password.</b>');
							$('#LoginMsg').show();
						}
					}
				});
			}
		});
	});
	
function resetAnim()
{
	$('#EmailScreen').css('left','0px');
	$('#PasswordScreen').hide();
	$("#EmailScreen").show();
	$('#PasswordSend').val('');
}
</script>
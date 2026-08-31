<?php
session_start();
$_SESSION['stats']=0;

if(!isset($_SESSION['img']))
header('location:signup.php');

include('engine/e.signup.m.php');
?>
<html>
<title>Quiz Portal</title>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.0/jquery.min.js"></script>
<script type="text/javascript" >

tp1=55+"%";
tp2=10+"%";

$(document).ready(function(){
    $("#fac").click(function(){
$("#main1").fadeIn();
$("#wel").hide();
$("#mn1").hide();
$("#mn2").hide();
$("#stu").hide();



$("#fac").animate({left:tp1},700);
$("#fac").animate({top:tp2},700);



});
});


$(document).ready(function(){
    $("#stu").click(function(){
$("#main2").fadeIn();
$("#wel").hide();
$("#mn1").hide();
$("#mn2").hide();
$("#fac").hide();


$("#stu").animate({left:tp1},700);
$("#stu").animate({top:tp2},700);

});
});


</script>

<head>

<link rel="stylesheet" type="text/css" href="prop/p.signup.css" media="all" />

</head>
<body>

<div id="head" >

<a href="index.php"><img src="images/4.jpg" id="logo1" ></a>
<a href="index.php"><p id="logo" >Quiz Portal</p></a>

</div>


<div id="main"  >


<div id="main5"  style="display:;">

<p id="txt51"  color="black" face="corbel" >Enter a 4-digit pin (for security purpose)</p>
<p id="txt52"  color="black" face="corbel" >Verify you are not a robot</p>

<span id="cap"><?php print $img; ?></span>


<p id="err1" style="color:dimgrey;font-size:12px;font-family:Century Gothic,sans-serif;position:absolute;top:0px;left:160px;line-height:125px;" >
<?php print $err1; ?>
<br><?php print $err2; ?>
</p>


<form method="post"
         action="" >


<input id="field51"
           type="password"
           name="pin"
           value="" >



<input id="field52"
           type="text"
           name="ver"
           value="" >
            
<input type="submit"
           name="verify"
           value="VERIFY"
             id="sb2"  >

</form>

</div>





</div>



<div id="foot" ></div>

</body>
</html>
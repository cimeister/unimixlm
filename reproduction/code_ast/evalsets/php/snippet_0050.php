<?php

/* home.php Revision 3 10-5-15 7:00pm */

/* ================================================================ */

// HEADER

// 1. Update the Header information below (all 3 lines).

/*
File Name: home.php
Date: 10/10/2015
Programmer: <NAME>
*/

// ==========================================================
// VARIABLES

/* 2. CHANGE the programmer name to your name, and the image name to the name of your image that you used in Assignment 2.
      COPY the new $link_2_page and link_2_text variables from the guestbook_add.php file into the space below, under the 
      $link_1_page and $link_1_text variables. */
 
$programmer_name = "<NAME>";
$home_image_name = "homePageImage.jpg";
$image_path = "images";
$styles_path = "styles";
$styles_page = "home.css";

$link_1_page = "home.php";
$link_1_text = "Home";

$link_2_page = "guestbook_add.php";
$link_2_text = "Guestbook: Add";

$heading = "CAS 225: PHP and MySQL for Designers";

/* ======================================= */

?>

<!DOCTYPE html>

<html>

<head>

<meta charset = "UTF-8">

<title><?php echo $heading; ?></title>

<link rel = "stylesheet" href = "<?php echo $styles_path . '/' . $styles_page; ?>" /> 

</head>

<body>

<div class = "shade">

<br />

<!-- We will code the line below later in the class, when we learn how to do logins. -->

Logged in as: <strong>(no login code yet)</strong>

<p>

<!-- ======================================
NAVIGATION BAR 
-->

<!-- 3. COPY the line for the second link from your guestbook_add.php page. This line must use PHP for credit. -->

     <a href="<?php echo $link_1_page; ?>"><?php echo $link_1_text; ?></a> |
     <a href="<?php echo $link_2_page; ?>"><?php echo $link_2_text; ?></a>


</p>

</div>

<h1><?php echo $heading; ?></h1>

<h2><?php echo $programmer_name; ?></h2>

<h3><?php echo date("F j, Y g:d a") ?></h3>

<img id="homeImage" src="<?php echo $image_path . '/' . $home_image_name; ?>" alt="<?php echo $home_image_name; ?>">

<!-- =============================================== -->

Please make a choice from the menu above.

</body>

</html>
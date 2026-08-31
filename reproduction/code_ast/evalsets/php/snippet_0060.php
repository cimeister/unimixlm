<?php
	include('dbDictTree.php');
	
	$id = $_POST['id'];
	
	deleteTreenode($id);
	
	//redirect to list
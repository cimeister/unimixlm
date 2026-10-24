<?php
$name=$_POST['name'];
$visitor_email=$POST['email']
$subject=$_POST['subject']
$message=$_POST['message']

$email_from = 'BhetarSchool.com';

$email_subject='New Form Submisssion';

$email_body="User name: $name.\n".
            "User Email: $visitor_email.\n".
            "Subject: $subject.\n".
            "User Message: $message.\n";

$to = '<EMAIL>';

$headers="From: $email_from \r\n";

$headers.="Reply-To: $visitor_email \r\n";

mail($to,$email_subject,$email_body,$headers);

header("Location: contact.html");
?>
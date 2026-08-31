<?php


function changetourl($string){
 
        $url = '~(?:(https?)://([^\s<]+)|(www\.[^\s<]+?\.[^\s<]+))(?<![\.,:])~i'; 
        $string = preg_replace($url, '<a href="$0" target="_blank" title="$0">$0</a>', $string);
        echo $string;
}
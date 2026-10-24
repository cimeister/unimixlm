$(document).ready(function(){

  var $body       = $('body');
  var $btnMenu    = $('.toggle-menu');

  $btnMenu.click(function() {
    $body.toggleClass('menu-open');
  });
});
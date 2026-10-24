
	function animup() {
	    $('.scroll_down').animate({ top: '-=10px' }, 300);
	    $('.scroll_down').animate({ top: '+=10px' }, 500);
	    $('.scroll_down').animate({ top: '-=10px' }, 500);
	    $('.scroll_down').animate({ top: '+=10px' }, 500);
	    $('.scroll_down').animate({ top: '-=10px' }, 500);
	    $('.scroll_down').animate({ top: '+=10px' }, 500);
	    $('.scroll_down').animate({ top: '-=10px' }, 500);
	    $('.scroll_down').animate({ top: '+=10px' }, 500);
		$('.scroll_down').animate({  borderSpacing: -360 }, {
		    step: function(now,fx) {
		      $(this).css('-webkit-transform','rotate('+now+'deg)'); 
		      $(this).css('-moz-transform','rotate('+now+'deg)');
		      $(this).css('transform','rotate('+now+'deg)');
		    },
		    duration: 'slow'
		},'linear');	
	    animdown();
	}

	function animdown() {
	$('.scroll_down').animate({ top: '+=0px' }, 200, animup);
		}

	$(document).ready(function() {
	    animup();
	});
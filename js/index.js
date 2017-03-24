var Handlebars = require('handlebars');
var marked = require('marked');
var helper = require('./helper');

Handlebars.registerHelper("markdown", function(array, sep, options) {
	var s = array.join('\n');
	return marked(s);
});

$(document).ready(function() {
	$.getJSON('./data.json', function(data) {

		var source = $("#template").html();
		var template = Handlebars.compile(source);

		console.log(helper.getLocal());

		var d = data[helper.getLocal()] || data.en;
		$('body').html(template(d));


		$('section').each(function() {
			if($(this).height() < window.innerHeight)
				$(this).height(window.innerHeight);
		});

		$('#fullpage').fullpage({
			//Navigation
	        menu: '#menu',
	        lockAnchors: false,
	        anchors:['firstPage', 'secondPage'],
	        navigation: false,
	        navigationPosition: 'right',
	        navigationTooltips: ['firstSlide', 'secondSlide'],
	        showActiveTooltip: false,
	        slidesNavigation: false,
	        slidesNavPosition: 'bottom',

	        //Scrolling
	        css3: true,
	        scrollingSpeed: 700,
	        autoScrolling: true,
	        fitToSection: true,
	        fitToSectionDelay: 1000,
	        scrollBar: false,
	        easing: 'easeInOutCubic',
	        easingcss3: 'ease',
	        loopBottom: false,
	        loopTop: false,
	        loopHorizontal: true,
	        continuousVertical: false,
	        continuousHorizontal: false,
	        scrollHorizontally: false,
	        interlockedSlides: false,
	        dragAndMove: false,
	        offsetSections: false,
	        resetSliders: false,
	        fadingEffect: false,
	        normalScrollElements: '#element1, .element2',
	        scrollOverflow: false,
	        scrollOverflowReset: false,
	        scrollOverflowOptions: null,
	        touchSensitivity: 15,
	        normalScrollElementTouchThreshold: 5,
	        bigSectionsDestination: null,

	        //Accessibility
	        keyboardScrolling: true,
	        animateAnchor: true,
	        recordHistory: true,

	        //Design
	        verticalCentered: true,
	        responsiveWidth: 0,
	        responsiveHeight: 0,
	        responsiveSlides: false
		});
		
		$('#works .owl-carousel').owlCarousel({
		    loop:true,
		    margin: 20,
		    responsive:{
		    	0: {
		    		margin: 0,
		    		items: 1
		    	},
		    	600: {
		    		margin: 60,
		    		items: 2
		    	},
		        1200: {
		            items: 3
		        },
		        1900: {
		            items: 4
		        }
		    }
		});
		$('#skills .owl-carousel').owlCarousel({
		    autoWidth:true,
		    responsive:{
		    	0: {
		    		items: 2,
		    		center:true
		    	},
		    	600: {
		   			center: false,
		    		items: 4
		    	},
		        1200: {
		        	center: false,
		            items: 6
		        },
		        1900: {
		        	center: false,
		            items: 8
		        }
		    }
		});
		$('#projects .owl-carousel').owlCarousel({
		    loop:true,
		    responsive:{
		    	0: {
		    		items: 1
		    	}
		    }
		});
	});
});
var Handlebars = require('handlebars');
var marked = require('marked');
var helper = require('./helper');

Handlebars.registerHelper("markdown", function(array, sep, options) {
	var s = array.join('\n');
	return marked(s);
});
Handlebars.registerHelper("size", function(desktop, mobile, options) {
	if(window.innerWidth > 600) return desktop; 
	else return mobile
});

$(document).ready(function() {
	$.getJSON('./data.json', function(data) {

		var source = $("#template").html();
		var template = Handlebars.compile(source);

		var d = data[helper.getLocal()] || data.ru;
		$('body').html(template(d));

		$('#fullpage').fullpage({
			//Navigation
	        menu: '#menu',
	        lockAnchors: false,
	        navigation: false,
	        navigationPosition: 'right',
	        showActiveTooltip: false,
	        slidesNavigation: false,
	        slidesNavPosition: 'bottom',

			//Scrolling
			css3: true,
			scrollBar: false,
			easing: 'easeInOutCubic',
			easingcss3: 'ease',
			fadingEffect: true,
			touchSensitivity: 30,

			//Accessibility
			keyboardScrolling: true,
			recordHistory: true,

			//Design
			verticalCentered: true,
			responsiveWidth: 600
		});
		
		$('#works .owl-carousel').owlCarousel({
			loop:true,
			dots: true,
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
			autoWidth: true,
			dots: true,
			center:true,
			responsive:{
				0: {
					items: 2,
					center: false
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
			dots: true,
			responsive:{
				0: {
					items: 1
				}
			}
		});
		$('#achievements .owl-carousel').owlCarousel({
			loop:true,
			dots: true,
			margin: 20,
			responsive:{
				0: {
					margin: 30,
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
	});
});
module.exports.getLocal = function() {
	return (navigator.language || navigator.systemLanguage || navigator.userLanguage).substr(0, 2).toLowerCase();
}
const gulp = require('gulp');
const browserify = require('gulp-browserify');
const sass = require('gulp-sass');
const autoprefixer = require('gulp-autoprefixer');
const connect = require('gulp-connect');
const plumber = require('gulp-plumber');
const notify = require('gulp-notify');
const uglify = require('gulp-uglify');
const uglifycss = require('gulp-uglifycss');

var errorMessage = () => {
	return plumber({errorHandler: notify.onError((err) => {
		return {
			title: err.name,
			message: err.message
		}
	})});
}

gulp.task('server', () => {
	return connect.server({
		port: 1338,
		livereload: true,
		root: './'
	})
});

gulp.task('styles', () => {
	gulp.src('./styles/**/*.scss')
		.pipe(errorMessage())
		.pipe(sass().on('error', sass.logError))
		.pipe(autoprefixer())
		.pipe(uglifycss())
		.pipe(gulp.dest('./'))
		.pipe(connect.reload());
});

gulp.task('js', () => {
	gulp.src('./js/index.js')
		.pipe(errorMessage())
		.pipe(browserify())
		.pipe(uglify())
		.pipe(gulp.dest('./'))
		.pipe(connect.reload());
});

gulp.task('watch', () => {
	gulp.watch('styles/**/*.scss', ['styles']);
	gulp.watch('js/**/*.js', ['js']);
});


gulp.task('default', ['styles', 'js', 'server', 'watch']);

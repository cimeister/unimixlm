var gulp = require('gulp');
var uglify = require("gulp-uglify");
var concat = require("gulp-concat");


gulp.task('uglify', function() {

    return gulp.src("src/*.*")
        .pipe(uglify())
        .pipe(concat("mandarin.min.js"))
        .pipe(gulp.dest('bin'));
});
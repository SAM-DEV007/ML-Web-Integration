$(document).on('click', '#restart', function(){
    if ($.cookie("restart") != null) {
        $.cookie("restart") = 1;
    }
});
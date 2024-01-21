$(document).on('click', '#predict', function() {
    $.ajax({
        url: $('#url').val(),
        data: {
            sentence: $('#sentence').val(),
            csrfmiddlewaretoken: $('input[name=csrfmiddlewaretoken]').val()
        },
        type: 'POST',
        error: function(error) {
            console.log(error);
        }
    });
});
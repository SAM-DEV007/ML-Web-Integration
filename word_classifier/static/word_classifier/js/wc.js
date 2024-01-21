$(document).on('click', '#predict', function() {
    $.ajax({
        url: $('#url').val(),
        data: {
            sentence: $('#sentence').val(),
            csrfmiddlewaretoken: $('input[name=csrfmiddlewaretoken]').val()
        },
        type: 'POST',
        success: function(response) {
            $('#predictsen').prop('readonly', false);
            $('#predictsen').val(response);
            $('#predictsen').prop('readonly', true);
        },
        error: function(error) {
            console.log(error);
        }
    });
});
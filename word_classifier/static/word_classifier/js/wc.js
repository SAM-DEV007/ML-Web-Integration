$(document).ready(function(){
    $('#limit').text(100);
    
    $("#sentence").on("input", function(){
        $('#limit').text(100 - $(this).val().length);
    });

    $('#predict').click(function(){
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
});
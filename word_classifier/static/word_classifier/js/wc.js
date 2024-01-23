$(document).ready(function(){
    let limit = 50;

    $('#limit').text(limit);
    
    $("#sentence").on("input", function(){
        if ($(this).val().length > limit){
            $(this).val($(this).val().substring(0, limit));
        }
        if ((limit - $(this).val().length) <= 30){
            $('#limit').css('color', 'red');
        } else {
            $('#limit').css('color', 'white');
        }
        $('#limit').text(limit - $(this).val().length);
    });

    $('#predict').click(function(){
        writeReadOnly($('#predictsen'), 'WAITING...');
        $.ajax({
            url: $('#url').val(),
            data: {
                sentence: $('#sentence').val(),
                csrfmiddlewaretoken: $('input[name=csrfmiddlewaretoken]').val()
            },
            type: 'POST',
            success: function(response) {
                if (response.includes('<!DOCTYPE html>')){
                    writeReadOnly($('#predictsen'), 'ERROR: FAILED TO FETCH DATA');
                    console.log('Model not found!');
                    return;
                }
                writeReadOnly($('#predictsen'), response);
            },
            error: function(error) {
                writeReadOnly($('#predictsen'), 'ERROR: FAILED TO FETCH DATA');
                console.log(error);
            }
        });
    });

    function writeReadOnly(element, text){
        element.prop('readonly', false);
        element.val(text);
        element.prop('readonly', true);
    }
});
$(document).ready(function(){
    $('#limit').text(100);
    
    $("#sentence").on("input", function(){
        $('#limit').text(100 - $(this).val().length);
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
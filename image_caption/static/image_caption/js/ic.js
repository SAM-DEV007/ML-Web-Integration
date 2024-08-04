$(document).ready(function(){
    function readURL(input){    
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#image1').attr('src', e.target.result);
            };

            reader.readAsDataURL(input.files[0]);
        }
    }

    function validateFileType(){
        var fileName = $('#file').val();
        var idxDot = fileName.lastIndexOf(".") + 1;
        var extFile = fileName.substr(idxDot, fileName.length).toLowerCase();
        
        if ((extFile=="jpg" || extFile=="jpeg" || extFile=="png") && (sizeText() <= 4)){
            readURL($('#file')[0]);
        } else {
            $('#file').val('');
            $('#image1').attr('src', $('#default').val());
            $('#imagesize').text('');
        }
    }

    function sizeText(){
        var _size = $('#file')[0].files[0].size;
        var fSExt = new Array('Bytes', 'KB', 'MB', 'GB')
        
        var mb_size = (_size / (1024*1024));

        var i=0;
        while(_size > 900){
            _size /= 1024;
            i++;
        }

        var exactSize = (Math.round(_size * 100) / 100);
        $('#imagesize').text('Uploaded Image Size: ' + exactSize + ' ' + fSExt[i]);

        return mb_size;
    }

    $('#file').change(function(){
        validateFileType();
    });

    $('#predict').click(function(){
        writeReadOnly($('#predictsen'), 'WAITING...');

        var file = $('#file')[0].files[0];
        if (file === undefined){
            writeReadOnly($('#predictsen'), 'ERROR: NO FILE SELECTED');
            return;
        }

        var formData = new FormData();
        formData.append('image', file);
        formData.append('csrfmiddlewaretoken', $('input[name=csrfmiddlewaretoken]').val());

        $.ajax({
            url: "model",
            data: formData,
            type: 'POST',
            cache: false,
            contentType: false,
            processData: false,
            success: function(response) {
                if (response.includes('<!DOCTYPE html>')){
                    writeReadOnly($('#predictsen'), 'ERROR: FAILED TO FETCH DATA');
                    console.log('Model not found!');
                    return;
                }
                var response = JSON.parse(response);
                writeReadOnly($('#predictsen'), response.caption);
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

    $('#default').val($('#image1').attr('src'));
});
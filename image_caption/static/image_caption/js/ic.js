$(document).ready(function(){
    let download_available = false;
    let width, height;
    let original_image;

    function readURL(input){    
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                var image = new Image();
                image.src = e.target.result;
                image.onload = function(){
                    width = this.width;
                    height = this.height;
                };
                delete image;
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
            reset_download();
            return;
        }

        var formData = new FormData();
        formData.append('image', file);
        formData.append('width', width);
        formData.append('height', height);
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
                    reset_download();
                    console.log('Model not found!');
                    return;
                }
                var response = JSON.parse(response);
                writeReadOnly($('#predictsen'), response.caption);
                $('#image2').attr('src', response.image);
                original_image = response.original_image;
                download_available = true;
            },
            error: function(error) {
                reset_download();
                writeReadOnly($('#predictsen'), 'ERROR: FAILED TO FETCH DATA');
                console.log(error);
            }
        });
    });

    $('#download').click(function(){
        if (download_available) {
            if ($('#originalsize')[0].checked) downloadImage(original_image);
            else downloadImage($('#image2').attr('src'));
        }
    });

    async function downloadImage(imageSrc) {
        const image = await fetch(imageSrc)
        const imageBlog = await image.blob()
        const imageURL = URL.createObjectURL(imageBlog)
      
        const link = document.createElement('a')
        link.href = imageURL
        link.download = 'image.jpg'
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
      }

    function writeReadOnly(element, text){
        element.prop('readonly', false);
        element.val(text);
        element.prop('readonly', true);
    }

    function reset_download(){
        $('#image2').attr('src', $('#default').val());
        download_available = false;
    }

    $('#default').val($('#image1').attr('src'));
});
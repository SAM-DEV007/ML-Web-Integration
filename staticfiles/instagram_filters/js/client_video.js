$(document).ready(function(){
    let video = document.getElementById('tempvid');
    let streaming = false;

    navigator.mediaDevices.getUserMedia({video: true, audio: false})
    .then(function(stream){
        video.srcObject = stream;
        video.play();
        streaming = true;
    })
    .catch(function(error){
        console.log(error);
    });

    video.addEventListener('canplay', function(ev){
        video.setAttribute('width', 640);
        video.setAttribute('height', 480);
        $('#canvas').attr('width', 640);
        $('#canvas').attr('height', 480);
    }, false);

    $('#vid').submit(function(e){
        e.preventDefault();
        $.ajax({
            type: 'POST',
            url: $('#urlpost').val(),
            data:
            {
                'webimg': $('#webimg').val(),
                'csrfmiddlewaretoken': $('input[name=csrfmiddlewaretoken]').val(),
            }
        })
    });

    setInterval(main, 0.1);

    function main(){
        if (streaming){
            getframe();
            $('#vid').submit();
        }
    }

    function getframe(){
        let context = $('#canvas')[0].getContext('2d');
        context.drawImage(video, 0, 0, 640, 480);

        let data = $('#canvas')[0].toDataURL('image/jpeg');
        $('#webimg').attr('value', data);
    }
});
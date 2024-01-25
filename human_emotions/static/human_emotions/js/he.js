$(document).ready(function(){
    let video = document.getElementById('tempvid');
    let img = document.getElementById('video');
    let ws_scheme = window.location.protocol == "https:" ? "wss" : "ws";
    let streaming = false;

    let ws = new WebSocket(
        ws_scheme + '://' + window.location.host + '/human_emotions'
    );

    ws.onmessage = (event) => {
        frameUpdate = event.data;
        img.src = "data:image/jpeg;base64," + frameUpdate; 
    };

    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({video: true, audio: false})
        .then(function(stream){
            video.srcObject = stream;
            video.play();
        })
        .catch(function(error){
            console.log(error);
        });
    }

    video.addEventListener('canplay', function(ev){
        video.setAttribute('width', 640);
        video.setAttribute('height', 480);
        $('#canvas').attr('width', 640);
        $('#canvas').attr('height', 480);

        streaming = true;
    }, false);

    setInterval(main, 100);

    function main(){
        if (streaming){
            getframe();
        }
    }

    function getframe(){
        let context = $('#canvas')[0].getContext('2d');
        context.drawImage(video, 0, 0, 640, 480);

        $('#canvas')[0].toBlob(function(blob) {
            if (ws.readyState == WebSocket.OPEN) {
                ws.send(blob);
            }
        }, 'image/jpeg');
    }
});
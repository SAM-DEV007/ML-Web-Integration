let igstat = 0

function igext() {
    $('.window').removeClass('ignormal');
    $('.window').addClass('igextend');
    $('.igtxt').removeClass('igsubhide');
    $('.igtxt').addClass('igsubshow');
    igstat = 1;
}

function ignorm() {
    $('.window').removeClass('igextend');
    $('.window').addClass('ignormal');
    $('.igtxt').removeClass('igsubshow');
    $('.igtxt').addClass('igsubhide');
    igstat = 0;
}

function igwindow() {
    if (igstat == 0) igext(); else ignorm();
}

$(document).on('click', '#igwindow_text', function() {
    igwindow();
});

let pages = ['#igwindow', '#wclwindow'];
let headings = ['#igwindow_text', '#wclwindow_text'];
let ig_force_change = false;
let curr = 0;

function hidearrow(arrow) {
    $(arrow).removeClass('.arrowshow');
    $(arrow).addClass('.arrowhide');
}

function showarrow(arrow) {
    $(arrow).removeClass('.arrowhide');
    $(arrow).addClass('.arrowshow');
}

function rightarrow() {
    if (curr == 0 && igstat == 1) {
        ignorm();
        ig_force_change = true;
    }
    $(pages[curr]).css('visibility', 'hidden');
    $(headings[curr]).fadeTo(1000, 0);
    curr += 1;
    if (curr >= pages.length) curr = pages.length - 1;
    if (curr == 1) {
        $('#left').css('visibility', 'visible').fadeTo(500, 1);
    }
    if (curr == pages.length - 1) {
        $('#right').fadeTo(500, 0).css('visibility', 'hidden');
    }
    $(pages[curr]).css('visibility', 'visible');
    $(headings[curr]).fadeTo(1000, 1);
}

$(document).on('click', '#right', function() {
    rightarrow();
});

function leftarrow() {
    $(pages[curr]).css('visibility', 'hidden');
    $(headings[curr]).fadeTo(1000, 0);
    curr -= 1;
    if (curr < 0) curr = 0;
    if (curr == pages.length - 2) {
        $('#right').css('visibility', 'visible').fadeTo(500, 1);
    }
    if (curr == 0) {
        $('#left').fadeTo(500, 0).css('visibility', 'hidden');
        if (ig_force_change) {
            igext();
            ig_force_change = false;
        }
    }
    $(pages[curr]).css('visibility', 'visible');
    $(headings[curr]).fadeTo(1000, 1);
}

$(document).on('click', '#left', function() {
    leftarrow();
});
$(document).ready(function(){
    $('#user-input').keypress(function(event){
        if(event.which == 13){
            event.preventDefault();
            $('#message-form').submit();
        }
    });

    $('#message-form').on('submit', function(event){
        event.preventDefault();
        var user_message = $('#user-input').val();
        $('#chat-messages').append('<div class="message"><p class="user-message">' + user_message + '</p></div>');
        $('#user-input').val('');
        $.ajax({
            url: '/get_response',
            type: 'POST',
            data: {user_message: user_message},
            success: function(response){
                $('#chat-messages').append('<div class="message"><p class="bot-message">' + response + '</p></div>');
                $('#chat-messages').scrollTop($('#chat-messages')[0].scrollHeight);
            }
        });
    });
});
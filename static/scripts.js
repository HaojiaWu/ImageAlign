document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('alignForm').addEventListener('submit', function(e) {
        e.preventDefault();
        var formData = new FormData(this);
        fetch('/align', {
            method: 'POST',
            body: formData
        })
        .then(response => response.blob())
        .then(blob => {
            var imageUrl = URL.createObjectURL(blob);
            var outputImage = document.getElementById('outputImage');
            outputImage.src = imageUrl;
        })
        .catch(error => console.error('Error:', error));
    });
});

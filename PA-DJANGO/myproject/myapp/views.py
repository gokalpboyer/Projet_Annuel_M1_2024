from django.shortcuts import render

def upload_photo(request):
    if request.method == 'POST':
        photo = request.FILES['photo']
        return render(request, 'myapp/upload_success.html')
    return render(request, 'myapp/upload_photo.html')
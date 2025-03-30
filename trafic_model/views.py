from  django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import io
import logging
from .modelLoad import predict #Import the prediction function 

# setup logging
logger = logging.getLogger(__name__)

def sam_model(request):
    '''simple test view'''
    return HttpResponse("here is API is running !")


@csrf_exempt
def predict_trafic(request):
    if request.method == 'GET':
        return JsonResponse({"message": "Send a POST request with an image to get predictions."})
    
    if request.method == 'POST' and 'image' in request.FILES:
        try:
            image_file = request.FILES['image']
            image = Image.open(io.BytesIO(image_file.read()))  # Open the image correctly
            
            # Call the model prediction function
            result = predict(image)
            
            return JsonResponse({"prediction": result})

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return JsonResponse({"error": f"Internal server error: {str(e)}"}, status=500)

    return JsonResponse({"error": "Invalid request. Please upload an image."}, status=400)
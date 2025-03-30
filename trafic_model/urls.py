from django.contrib import admin
from django.urls import path
from .views import predict_trafic

urlpatterns = [
    path('predict/', predict_trafic, name="model"),
]
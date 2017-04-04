from django.conf import settings
from django.conf.urls import url
from django.conf.urls.static import static
from django.contrib import admin
from django.shortcuts import redirect

from redditlsa import views

admin.autodiscover()

urlpatterns = [
    url(r'^$', lambda request: redirect('algebra'), name='home'),
    url(r'^algebra/$', views.algebra_view, name='algebra'),
    url(r'^maps/$', views.map_view, name='map'),
    url(r'^about/$', views.about_view, name='about'),
    url(r'^search/$', views.search_view, name='search'),
    url(r'^refine/$', views.refine_view, name='refine')
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

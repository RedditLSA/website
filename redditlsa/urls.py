from django.conf import settings
from django.conf.urls import url
from django.conf.urls.static import static
from django.contrib import admin
from django.shortcuts import redirect
from django.urls import reverse

from redditlsa import views

admin.autodiscover()

query_string = '?insubs=Feminism&insubs=politics&insubs=altright&insubs=antivax&outsubs=hillaryclinton&outsubs' \
               '=SandersForPresident&outsubs=The_Donald&outsubs=GaryJohnson&outsubs=jillstein&method=optimal'

handler500 = 'redditlsa.views.server_error'
urlpatterns = [
    url(r'^$', lambda request: redirect(reverse('map') + query_string), name='home'),
    url(r'^algebra/$', views.algebra_view, name='algebra'),
    url(r'^maps/$', views.map_view, name='map'),
    url(r'^about/$', views.about_view, name='about'),
    url(r'^search/$', views.search_view, name='search'),
    url(r'^refine/$', views.refine_view, name='refine')
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

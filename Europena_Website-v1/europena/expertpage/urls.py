from django.conf.urls import url

from . import views

app_name = 'expertpage'

urlpatterns = [
    url(r'^$', views.index, name='index'),
    # url(r'^(?P<document_id>[0-9]+[\/\]+[A-za-z0-9]+)/$', views.increase, name='increase'),
    url(r'^(?P<document_id>[0-9]+[\/\]+[A-za-z0-9]+)/(?P<value>[+-]?[0-9])/$', views.eval, name='eval'),
]

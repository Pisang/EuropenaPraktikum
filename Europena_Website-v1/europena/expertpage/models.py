from django.db import models


class Document(models.Model):
    id = models.CharField(max_length=150, primary_key=True)
    contributor = models.CharField(max_length=500, null=True)
    country = models.CharField(max_length=30, null=True)
    created = models.CharField(max_length=50, null=True)
    creator = models.CharField(max_length=100, null=True)
    dataProvider = models.CharField(max_length=50, null=True)
    date = models.CharField(max_length=20, null=True)
    description = models.CharField(max_length=3000, null=True)
    format = models.CharField(max_length=20, null=True)
    identifier = models.CharField(max_length=100, null=True)
    language = models.CharField(max_length=10, null=True)
    medium = models.CharField(max_length=100, null=True)
    provider = models.CharField(max_length=100, null=True)
    publisher = models.CharField(max_length=100, null=True)
    relation = models.CharField(max_length=500, null=True)
    spatial = models.CharField(max_length=100, null=True)
    subject = models.CharField(max_length=100, null=True)
    title = models.CharField(max_length=1000, null=True)
    type = models.CharField(max_length=100, null=True)
    year = models.CharField(max_length=100, null=True)
    known_genre = models.CharField(max_length=20, null=True)
    approved = models.IntegerField(null=True)

    def __str__(self):
        return self.id + ' : ' + self.title

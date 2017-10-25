from django.http import HttpResponseRedirect
from django.shortcuts import render, get_object_or_404

from .models import Document


def index(request):
    all_documents = Document.objects.all()
    context = {'all_documents': all_documents}
    return render(request, 'expertpage/index.html', context)


def increase(request, document_id):
    all_documents = Document.objects.all()
    context = {'all_documents': all_documents}

    document_id = '/' + document_id;
    doc = get_object_or_404(Document, pk=str(document_id))

    doc.approved += 1
    doc.save()

    return render(request, 'expertpage/index.html', context)


def eval(request, document_id, value):
    all_documents = Document.objects.all()
    context = {'all_documents': all_documents}

    document_id = '/' + document_id;
    doc = get_object_or_404(Document, pk=str(document_id))

    doc.approved += int(value)
    doc.save()

    return HttpResponseRedirect('/expertpage')

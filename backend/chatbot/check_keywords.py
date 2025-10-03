from .models import DocumentMetadata

docs = DocumentMetadata.objects.all()
for doc in docs:
    print(f"{doc.filename}: keywords='{doc.keywords}'")

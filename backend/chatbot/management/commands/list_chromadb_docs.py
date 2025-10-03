from django.core.management.base import BaseCommand
from chatbot.chroma_connection import ChromaService

class Command(BaseCommand):
    help = 'List all documents in ChromaDB'

    def handle(self, *args, **kwargs):
        try:
            collection = ChromaService.get_client().get_or_create_collection(name="documents")
            
            # Get all documents
            results = collection.get(
                include=["metadatas"],
                where={"source": "pdf_scrape"}
            )
            
            ids = results.get('ids', [])
            metadatas = results.get('metadatas', [])
            
            self.stdout.write(self.style.SUCCESS(f'\n📊 Found {len(ids)} documents in ChromaDB:\n'))
            
            # Filter for architecture-related
            arch_docs = []
            
            for i, (doc_id, metadata) in enumerate(zip(ids, metadatas)):
                filename = metadata.get('filename', 'N/A')
                folder = metadata.get('folder_name', 'N/A')
                keywords = metadata.get('keywords', 'N/A')
                
                # Check if architecture related
                filename_lower = filename.lower()
                if 'cs' in filename_lower or 'computer science' in filename_lower:
                    arch_docs.append((filename, keywords, folder))
                
                # Print all (first 50)
                if i < 50:
                    self.stdout.write(f'{i+1}. {filename}')
                    if keywords:
                        self.stdout.write(f'   Keywords: {keywords[:80]}')
                    self.stdout.write(f'   Folder: {folder}\n')
            
            # Show Computer Science docs
            if arch_docs:
                self.stdout.write(self.style.SUCCESS('\n🏛️ Computer Science-related documents found:'))
                for filename, keywords, folder in arch_docs:
                    self.stdout.write(self.style.SUCCESS(f'  ✅ {filename}'))
                    self.stdout.write(f'     Keywords: {keywords}')
                    self.stdout.write(f'     Folder: {folder}\n')
            else:
                self.stdout.write(self.style.ERROR('\n❌ NO Computer Science documents found in ChromaDB!'))
                self.stdout.write(self.style.WARNING('This means the BS Computer Science curriculum was never uploaded or synced properly.'))
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error: {e}'))
            import traceback
            traceback.print_exc()

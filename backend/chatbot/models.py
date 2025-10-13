from django.db import models

# Create your models here.

class TestItem(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class Topic(models.Model):
    """Predefined topics for guided conversation"""
    TOPIC_CHOICES = [
        ('admissions_enrollment', 'General Admissions and Enrollment'),
        ('programs_courses', 'Programs and Courses'),
        ('fees', 'Fees'),
    ]
    
    topic_id = models.CharField(max_length=50, choices=TOPIC_CHOICES, unique=True)
    label = models.CharField(max_length=255)
    description = models.TextField()
    retrieval_strategy = models.CharField(max_length=50, default='generic')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.label} ({self.topic_id})"
    
    def get_keywords_list(self):
        """Get all keywords for this topic as a list"""
        return [keyword.keyword for keyword in self.keywords.filter(is_active=True)]
    
    def get_keywords_string(self):
        """Get all keywords for this topic as a comma-separated string"""
        return ', '.join(self.get_keywords_list())

class TopicKeyword(models.Model):
    """Keywords associated with topics for document filtering"""
    topic = models.ForeignKey(Topic, on_delete=models.CASCADE, related_name='keywords')
    keyword = models.CharField(max_length=255)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.CharField(max_length=255, blank=True, help_text="Admin who added this keyword")
    
    class Meta:
        unique_together = ['topic', 'keyword']
        ordering = ['keyword']
    
    def __str__(self):
        return f"{self.topic.topic_id}: {self.keyword}"
    
class Conversation(models.Model):
    """Represents a conversation session"""
    session_id = models.CharField(max_length=255, unique=True)
    title = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    # Token budgeting fields
    total_exchanges = models.IntegerField(default=0)
    current_token_count = models.IntegerField(default=0)
    max_token_budget = models.IntegerField(default=3000)  # Safe margin under 4K
    
    def __str__(self):
        return f"Conversation {self.session_id} ({self.total_exchanges} exchanges)"

class ConversationTurn(models.Model):
    """Represents a single user query and bot response"""
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='turns')
    turn_number = models.IntegerField()
    user_query = models.TextField()
    bot_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Token tracking
    query_tokens = models.IntegerField(default=0)
    response_tokens = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)
    
    # Context tracking
    used_in_context = models.BooleanField(default=True)  # Whether this turn is in active context
    
    class Meta:
        ordering = ['turn_number']
        unique_together = ['conversation', 'turn_number']
    
    def __str__(self):
        return f"Turn {self.turn_number}: {self.user_query[:50]}..."

class ConversationSummary(models.Model):
    """Stores summarized conversation history"""
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='summaries')
    summary_text = models.TextField()
    covers_turns_start = models.IntegerField()  # First turn number included in summary
    covers_turns_end = models.IntegerField()    # Last turn number included in summary
    summary_tokens = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)  # Whether this summary is currently used
    
    # Recursive summarization tracking
    parent_summary = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True)
    summary_level = models.IntegerField(default=1)  # 1 = first level, 2 = summary of summaries, etc.
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Summary for turns {self.covers_turns_start}-{self.covers_turns_end}: {self.summary_text[:50]}..."

class SystemPrompt(models.Model):
    """Stores system prompts with token counts"""
    name = models.CharField(max_length=100, unique=True)
    prompt_text = models.TextField()
    token_count = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} ({self.token_count} tokens)"

class DocumentFolder(models.Model):
    """Represents a folder for organizing documents with hierarchical support"""
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    color = models.CharField(max_length=7, default="#063970")  # Hex color
    
    # NEW: Parent folder relationship for hierarchical structure
    parent_folder = models.ForeignKey(
        'self', 
        on_delete=models.CASCADE, 
        null=True, 
        blank=True, 
        related_name='subfolders'
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        # Ensure unique names only within the same parent folder
        unique_together = ['name', 'parent_folder']
    
    def __str__(self):
        if self.parent_folder:
            return f"{self.parent_folder.name} / {self.name}"
        return self.name
    
    @property
    def document_count(self):
        """Count only documents directly in this folder"""
        return self.documents.count()
    
    @property
    def total_document_count(self):
        """Count documents in this folder and all subfolders recursively"""
        count = self.documents.count()
        for subfolder in self.subfolders.all():
            count += subfolder.total_document_count
        return count
    
    @property
    def folder_path(self):
        """Get the full path of this folder"""
        path_parts = [self.name]
        current = self.parent_folder
        while current:
            path_parts.insert(0, current.name)
            current = current.parent_folder
        return " / ".join(path_parts)
    
    def get_all_descendant_folders(self):
        """Get all subfolders recursively"""
        descendants = []
        for subfolder in self.subfolders.all():
            descendants.append(subfolder)
            descendants.extend(subfolder.get_all_descendant_folders())
        return descendants
    
    @property
    def level(self):
        """Get the depth level of this folder (0 for root folders)"""
        level = 0
        current = self.parent_folder
        while current:
            level += 1
            current = current.parent_folder
        return level

class DocumentMetadata(models.Model):
    """Enhanced metadata for documents stored in ChromaDB"""
    # Document identification
    document_id = models.CharField(max_length=255, unique=True)  # Matches ChromaDB doc ID
    filename = models.CharField(max_length=255)
    
    # Folder organization
    folder = models.ForeignKey(DocumentFolder, on_delete=models.CASCADE, related_name='documents')
    
    # Categorization metadata
    DOCUMENT_TYPES = [
        ('admission', 'Admission Requirements'),
        ('enrollment', 'Enrollment Process'),
        ('scholarship', 'Scholarships & Financial Aid'),
        ('academic', 'Academic Programs'),
        ('fees', 'Fees & Payments'),
        ('policy', 'Policies & Procedures'),
        ('contact', 'Contact Information'),
        ('other', 'Other'),
    ]
    document_type = models.CharField(max_length=20, choices=DOCUMENT_TYPES, default='other')
    
    PROGRAMS = [
        ('undergraduate', 'Undergraduate Programs'),
        ('graduate', 'Graduate Programs'),
        ('senior_high', 'Senior High School'),
        ('all', 'All Programs'),
    ]
    target_program = models.CharField(max_length=20, choices=PROGRAMS, default='all')
    
    # Keywords for better searchability
    keywords = models.TextField(blank=True, help_text="Comma-separated keywords")
       
    # Administrative tracking
    uploaded_by = models.CharField(max_length=255, blank=True)
    last_modified = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # ChromaDB sync tracking
    synced_to_chroma = models.BooleanField(default=False)
    chroma_metadata_hash = models.CharField(max_length=64, blank=True)  # For change detection
    
    def __str__(self):
        return f"{self.filename} ({self.folder.name})"
    
    def get_keywords_list(self):
        return [k.strip() for k in self.keywords.split(',') if k.strip()]
    
    def get_chroma_metadata(self):
        """Generate metadata dict for ChromaDB storage"""
        return {
            'filename': self.filename,
            'source': 'pdf_scrape',
            'folder_id': str(self.folder.id),
            'folder_name': self.folder.name,
            'document_type': self.document_type,
            'target_program': self.target_program,
            'keywords': self.keywords,
            'document_id': self.document_id,
        }
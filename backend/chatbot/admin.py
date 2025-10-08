from django.contrib import admin
from .models import (
    TestItem, Conversation, ConversationTurn, ConversationSummary, 
    SystemPrompt, DocumentFolder, DocumentMetadata, Topic, TopicKeyword
)

# Register your models here.

@admin.register(TestItem)
class TestItemAdmin(admin.ModelAdmin):
    list_display = ['name', 'created_at']
    search_fields = ['name']

class TopicKeywordInline(admin.TabularInline):
    """Inline admin for managing keywords within a topic"""
    model = TopicKeyword
    extra = 3
    fields = ['keyword', 'is_active', 'created_by']
    
@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    """Admin interface for managing topics and their keywords"""
    list_display = ['topic_id', 'label', 'keyword_count', 'is_active', 'updated_at']
    list_filter = ['is_active', 'retrieval_strategy', 'created_at']
    search_fields = ['topic_id', 'label', 'description']
    readonly_fields = ['created_at', 'updated_at']
    inlines = [TopicKeywordInline]
    
    fieldsets = (
        ('Topic Information', {
            'fields': ('topic_id', 'label', 'description', 'retrieval_strategy', 'is_active')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def keyword_count(self, obj):
        """Display the number of active keywords for this topic"""
        return obj.keywords.filter(is_active=True).count()
    keyword_count.short_description = 'Active Keywords'
    
    def get_readonly_fields(self, request, obj=None):
        """Make topic_id readonly for existing objects"""
        if obj:  # Editing existing object
            return self.readonly_fields + ['topic_id']
        return self.readonly_fields

@admin.register(TopicKeyword)
class TopicKeywordAdmin(admin.ModelAdmin):
    """Admin interface for managing individual keywords"""
    list_display = ['topic', 'keyword', 'is_active', 'created_by', 'created_at']
    list_filter = ['topic', 'is_active', 'created_at']
    search_fields = ['keyword', 'topic__label', 'topic__topic_id']
    list_editable = ['is_active']
    
    fieldsets = (
        (None, {
            'fields': ('topic', 'keyword', 'is_active', 'created_by')
        }),
    )
    
    def save_model(self, request, obj, form, change):
        """Automatically set created_by to current user"""
        if not change:  # Only for new objects
            obj.created_by = request.user.username
        super().save_model(request, obj, form, change)

# Register existing models
@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ['session_id', 'title', 'total_exchanges', 'current_token_count', 'is_active', 'created_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['session_id', 'title']
    readonly_fields = ['created_at', 'updated_at']

@admin.register(ConversationTurn)
class ConversationTurnAdmin(admin.ModelAdmin):
    list_display = ['conversation', 'turn_number', 'query_preview', 'total_tokens', 'timestamp']
    list_filter = ['used_in_context', 'timestamp']
    search_fields = ['conversation__session_id', 'user_query']
    readonly_fields = ['timestamp']
    
    def query_preview(self, obj):
        return obj.user_query[:50] + "..." if len(obj.user_query) > 50 else obj.user_query
    query_preview.short_description = 'Query Preview'

@admin.register(DocumentFolder)
class DocumentFolderAdmin(admin.ModelAdmin):
    list_display = ['name', 'document_count', 'color', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at']

@admin.register(DocumentMetadata)
class DocumentMetadataAdmin(admin.ModelAdmin):
    list_display = ['filename', 'folder', 'document_type', 'target_program', 'synced_to_chroma', 'last_modified']
    list_filter = ['document_type', 'target_program', 'synced_to_chroma', 'folder']
    search_fields = ['filename', 'keywords', 'document_id']
    readonly_fields = ['document_id', 'created_at', 'last_modified', 'chroma_metadata_hash']

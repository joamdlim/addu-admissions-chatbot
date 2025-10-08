#!/usr/bin/env python
"""
Script to populate the database with predefined topics and keywords from topics.py
"""

import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from chatbot.models import Topic, TopicKeyword
from chatbot.topics import TOPICS

def populate_topics_and_keywords():
    """Populate the database with topics and keywords from topics.py"""
    
    print("🚀 Starting topic and keyword population...")
    
    for topic_id, topic_data in TOPICS.items():
        print(f"\n📋 Processing topic: {topic_id}")
        
        # Create or get the topic
        topic, created = Topic.objects.get_or_create(
            topic_id=topic_id,
            defaults={
                'label': topic_data['label'],
                'description': topic_data['description'],
                'retrieval_strategy': topic_data.get('retrieval_strategy', 'generic'),
                'is_active': True
            }
        )
        
        if created:
            print(f"   ✅ Created topic: {topic.label}")
        else:
            print(f"   ℹ️  Topic already exists: {topic.label}")
            # Update existing topic data
            topic.label = topic_data['label']
            topic.description = topic_data['description']
            topic.retrieval_strategy = topic_data.get('retrieval_strategy', 'generic')
            topic.save()
            print(f"   🔄 Updated topic data")
        
        # Add keywords
        keywords = topic_data.get('keywords', [])
        added_count = 0
        skipped_count = 0
        
        for keyword_text in keywords:
            keyword_text = keyword_text.strip()
            if not keyword_text:
                continue
                
            # Create keyword if it doesn't exist
            keyword, created = TopicKeyword.objects.get_or_create(
                topic=topic,
                keyword=keyword_text,
                defaults={
                    'is_active': True,
                    'created_by': 'system_migration'
                }
            )
            
            if created:
                added_count += 1
            else:
                skipped_count += 1
        
        print(f"   📝 Keywords: {added_count} added, {skipped_count} already existed")
        print(f"   🏷️  Total keywords for {topic.label}: {topic.keywords.count()}")
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    print(f"   Topics in database: {Topic.objects.count()}")
    print(f"   Total keywords in database: {TopicKeyword.objects.count()}")
    print(f"   Active keywords: {TopicKeyword.objects.filter(is_active=True).count()}")
    
    # Print topic breakdown
    for topic in Topic.objects.all():
        active_keywords = topic.keywords.filter(is_active=True).count()
        total_keywords = topic.keywords.count()
        print(f"   • {topic.label}: {active_keywords}/{total_keywords} active keywords")
    
    print(f"\n✅ Topic and keyword population completed!")

if __name__ == "__main__":
    populate_topics_and_keywords()


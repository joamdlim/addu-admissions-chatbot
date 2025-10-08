#!/usr/bin/env python
"""
Script to create a simple migration for Topic and TopicKeyword models
without touching existing DocumentMetadata fields.
"""

import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from django.core.management import execute_from_command_line
from django.db import migrations, models
import django.db.models.deletion

# Create a custom migration
migration_content = '''# Generated migration for Topic models

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0003_documentfolder_documentmetadata'),
    ]

    operations = [
        migrations.CreateModel(
            name='Topic',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('topic_id', models.CharField(choices=[('admissions_enrollment', 'General Admissions and Enrollment'), ('programs_courses', 'Programs and Courses'), ('fees', 'Fees')], max_length=50, unique=True)),
                ('label', models.CharField(max_length=255)),
                ('description', models.TextField()),
                ('retrieval_strategy', models.CharField(default='generic', max_length=50)),
                ('is_active', models.BooleanField(default=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name='TopicKeyword',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('keyword', models.CharField(max_length=255)),
                ('is_active', models.BooleanField(default=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('created_by', models.CharField(blank=True, help_text='Admin who added this keyword', max_length=255)),
                ('topic', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='keywords', to='chatbot.topic')),
            ],
            options={
                'ordering': ['keyword'],
            },
        ),
        migrations.AlterUniqueTogether(
            name='topickeyw ord',
            unique_together={('topic', 'keyword')},
        ),
    ]
'''

# Write the migration file
migration_dir = 'chatbot/migrations'
migration_file = os.path.join(migration_dir, '0004_add_topic_models.py')

with open(migration_file, 'w') as f:
    f.write(migration_content)

print(f"✅ Created migration file: {migration_file}")
print("Now run: python manage.py migrate")

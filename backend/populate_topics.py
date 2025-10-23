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

def populate_topics_and_keywords():
    """Populate the database with topics and keywords"""
    
    print("🚀 Starting topic and keyword population...")
    
    # Define topics data directly (since TOPICS is removed from topics.py)
    TOPICS_DATA = {
        'admissions_enrollment': {
            'label': 'General Admissions and Enrollment',
            'description': 'Learn about admissions, enrollment processes, requirements, documents, payment procedures, grading system, QPI appeals, and academic standing for all student types',
            'retrieval_strategy': 'admissions_specialized',
            'keywords': [
                # Basic admission terms
                'admission', 'admissions', 'apply', 'application', 'enroll', 'enrollment', 'register', 'registration',
                'requirements', 'requirement', 'documents', 'document', 'submit', 'submission',
                
                # Student types
                'new student', 'transfer student', 'transferee', 'returning student', 'graduate student', 'undergraduate',
                'freshman', 'sophomore', 'junior', 'senior', 'postgraduate', 'masters', 'phd', 'doctorate',
                
                # Application process
                'application form', 'application process', 'how to apply', 'when to apply', 'deadline', 'deadlines',
                'entrance exam', 'entrance test', 'interview', 'screening', 'evaluation',
                
                # Documents and requirements
                'transcript', 'grades', 'diploma', 'certificate', 'birth certificate', 'id', 'identification',
                'recommendation letter', 'letter of recommendation', 'medical certificate', 'clearance',
                'passport', 'visa', 'nso', 'psa', 'form 137', 'form 138', 'tor', 'transcript of records',
                
                # Payment and fees (expanded)
                'payment', 'pay', 'fee', 'fees', 'tuition', 'cost', 'price', 'amount', 'bill', 'billing',
                'online payment', 'pay online', 'bank payment', 'bank deposit', 'bank transfer', 'otc', 'over the counter',
                'cheque', 'check', 'cash payment', 'gcash', 'paymaya', 'digital payment', 'e-payment',
                'payment method', 'payment procedure', 'payment process', 'how to pay', 'where to pay',
                'payment deadline', 'payment schedule', 'installment', 'partial payment',
                'bdo', 'bpi', 'metrobank', 'landbank', 'pnb', 'unionbank', 'security bank', 'rcbc',
                'chinabank', 'eastwest', 'psbank', 'maybank', 'bank of commerce', 'robinsons bank',
                
                # Administrative actions (expanded)
                'loa', 'leave of absence', 'withdrawal', 'withdraw', 'dropping', 'drop', 'refund', 'reimbursement',
                'administrative fee', 'processing fee', 'late payment', 'penalty', 'surcharge',
                
                # Process and status
                'status', 'application status', 'accepted', 'rejected', 'pending', 'waitlist', 'conditional',
                'confirmation', 'confirm', 'slot', 'reservation', 'reserve',
                
                # Grading system and academic performance
                'grading system', 'grading', 'grade', 'grades', 'gpa', 'qpi', 'quality point index',
                'grade point average', 'cumulative', 'semester grade', 'final grade', 'midterm grade',
                'grade computation', 'grade calculation', 'weighted average', 'grade equivalent',
                'letter grade', 'numerical grade', 'percentage grade', 'passing grade', 'failing grade',
                'incomplete', 'inc', 'dropped', 'drp', 'withdrawn', 'wd', 'no grade', 'ng',
                
                # Academic standing and appeals
                'academic standing', 'good standing', 'probation', 'academic probation', 'dismissal',
                'academic dismissal', 'suspension', 'academic suspension', 'warning', 'academic warning',
                'dean\'s list', 'honor roll', 'honors', 'magna cum laude', 'cum laude', 'summa cum laude',
                'latin honors', 'academic distinction', 'academic excellence',
                
                # Grade appeals and petitions
                'grade appeal', 'appeal', 'petition', 'grade petition', 'grade change', 'grade correction',
                'grade reconsideration', 'recomputation', 'grade review', 'academic appeal',
                'appeal process', 'appeal procedure', 'appeal form', 'appeal deadline',
                'grade dispute', 'grade inquiry', 'grade complaint', 'grade grievance',
                
                # Academic records and transcripts
                'transcript of records', 'tor', 'official transcript', 'unofficial transcript',
                'academic record', 'permanent record', 'grade report', 'report card',
                'class record', 'grade sheet', 'academic history', 'scholastic record',
                
                # Grade-related policies
                'grading policy', 'grade policy', 'academic policy', 'grade retention',
                'grade deadline', 'grade submission', 'grade posting', 'grade release',
                'grade inquiry period', 'grade viewing', 'grade access'
            ]
        },
        'programs_courses': {
            'label': 'Programs and Courses',
            'description': 'Learn about available academic programs, degrees, curriculum, and course offerings',
            'retrieval_strategy': 'programs_specialized',
            'keywords': [
                # General program terms
                'program', 'programs', 'course', 'courses', 'degree', 'degrees', 'major', 'majors',
                'curriculum', 'subjects', 'units', 'credit', 'credits', 'semester', 'trimester',
                
                # Program levels
                'undergraduate', 'graduate', 'postgraduate', 'bachelor', 'bachelors', 'master', 'masters',
                'phd', 'doctorate', 'doctoral', 'certificate', 'diploma',
                
                # Colleges and schools
                'college', 'school', 'department', 'faculty', 'institute',
                'engineering', 'business', 'education', 'arts', 'sciences', 'medicine', 'nursing',
                'law', 'psychology', 'computer science', 'information technology', 'architecture',
                
                # Specific programs
                'accountancy', 'accounting', 'marketing', 'management', 'finance', 'economics',
                'civil engineering', 'mechanical engineering', 'electrical engineering', 'chemical engineering',
                'computer engineering', 'industrial engineering', 'environmental engineering',
                'elementary education', 'secondary education', 'special education', 'physical education',
                'biology', 'chemistry', 'physics', 'mathematics', 'statistics', 'psychology',
                'political science', 'sociology', 'anthropology', 'history', 'philosophy',
                'english', 'literature', 'communication', 'journalism', 'mass communication',
                
                # Course details
                'prerequisite', 'corequisite', 'elective', 'required', 'core', 'specialization',
                'track', 'strand', 'concentration', 'minor', 'double major', 'honors',
                'schedule', 'time', 'section', 'class', 'laboratory', 'lecture', 'seminar',
                
                # Academic terms
                'academic year', 'school year', 'term', 'quarter', 'summer', 'midyear',
                'regular', 'irregular', 'full time', 'part time', 'evening', 'weekend',
                'online', 'distance learning', 'hybrid', 'blended'
            ]
        },
        'fees': {
            'label': 'Fees',
            'description': 'Learn about tuition fees, payment options, payment methods, banks, and financial information',
            'retrieval_strategy': 'fees_specialized',
            'keywords': [
                # Basic fee terms
                'fee', 'fees', 'tuition', 'cost', 'price', 'amount', 'charge', 'rate',
                'bill', 'billing', 'invoice', 'statement', 'balance', 'due', 'owed',
                
                # Payment terms
                'payment', 'pay', 'paid', 'paying', 'installment', 'partial', 'full payment',
                'down payment', 'deposit', 'advance payment', 'balance payment',
                'payment plan', 'payment scheme', 'payment option', 'payment method',
                
                # Payment methods (comprehensive)
                'online payment', 'pay online', 'internet banking', 'mobile banking',
                'bank deposit', 'bank transfer', 'wire transfer', 'fund transfer',
                'over the counter', 'otc', 'cash payment', 'cash',
                'cheque', 'check', 'manager check', 'cashier check',
                'gcash', 'paymaya', 'grabpay', 'digital wallet', 'e-wallet',
                'credit card', 'debit card', 'atm', 'pos', 'point of sale',
                
                # Banks (comprehensive list)
                'bdo', 'banco de oro', 'bpi', 'bank of the philippine islands',
                'metrobank', 'metropolitan bank', 'landbank', 'land bank',
                'pnb', 'philippine national bank', 'unionbank', 'union bank',
                'security bank', 'rcbc', 'rizal commercial banking corporation',
                'chinabank', 'china banking corporation', 'eastwest', 'eastwest bank',
                'psbank', 'philippine savings bank', 'maybank', 'maybank philippines',
                'bank of commerce', 'robinsons bank', 'sterling bank', 'ucpb',
                'asia united bank', 'aub', 'philippine bank of communications',
                'pbcom', 'veterans bank', 'card bank', 'card mri',
                
                # Administrative fees
                'administrative fee', 'processing fee', 'service fee', 'handling fee',
                'late payment', 'penalty', 'surcharge', 'interest', 'fine',
                'loa fee', 'leave of absence fee', 'withdrawal fee', 'dropping fee',
                'refund', 'reimbursement', 'return', 'credit'
            ]
        }
    }
    
    for topic_id, topic_data in TOPICS_DATA.items():
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


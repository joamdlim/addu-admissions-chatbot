#!/usr/bin/env python3
"""
Hybrid Retrieval System Semantic Relevance Evaluation
Tests the existing TF-IDF + Word2Vec hybrid retrieval system in the guided chatbot.
Target: F1-score ≥ 0.70 on semantic relevance tests over 50 varied user queries.
"""

import os
import sys
import json
import time
from typing import List, Dict, Tuple
from sklearn.metrics import precision_recall_fscore_support
import requests

# Add the current directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class HybridRetrievalSemanticEvaluator:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8000/chatbot"
        
        # Define 150 varied user queries with expected relevant topics/documents
        self.test_queries = [
            # Admissions & Enrollment Queries (60 queries)
            {
                "query": "What are the admission requirements for first-year students?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["admission", "requirements", "first-year", "students", "documents"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "How do I apply for scholarships and financial aid?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["scholarships", "financial", "aid", "apply", "assistance"],
                "semantic_category": "financial_assistance"
            },
            {
                "query": "What documents do transfer students need to submit?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["transfer", "students", "documents", "submit", "requirements"],
                "semantic_category": "transfer_requirements"
            },
            {
                "query": "When is the application deadline for international students?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["application", "deadline", "international", "students"],
                "semantic_category": "international_admission"
            },
            {
                "query": "How do I enroll as a returning student?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["enroll", "returning", "student", "process"],
                "semantic_category": "returning_student"
            },
            {
                "query": "What are the eligibility criteria for merit scholarships?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["eligibility", "criteria", "merit", "scholarships"],
                "semantic_category": "scholarship_eligibility"
            },
            {
                "query": "Can I get financial assistance for my studies?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["financial", "assistance", "studies", "aid"],
                "semantic_category": "financial_assistance"
            },
            {
                "query": "What is the process for undergraduate admission to ADDU?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["process", "undergraduate", "admission", "ADDU"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "How do I submit my high school transcripts?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["submit", "high", "school", "transcripts", "documents"],
                "semantic_category": "document_submission"
            },
            {
                "query": "What are the requirements for student visa application?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["requirements", "student", "visa", "application", "international"],
                "semantic_category": "visa_requirements"
            },
            {
                "query": "Is there need-based financial aid available?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["need-based", "financial", "aid", "available"],
                "semantic_category": "financial_assistance"
            },
            {
                "query": "How do I check my application status at ADDU?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["check", "application", "status", "ADDU"],
                "semantic_category": "admission_status"
            },
            {
                "query": "What are the English proficiency requirements?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["English", "proficiency", "requirements", "international"],
                "semantic_category": "language_requirements"
            },
            {
                "query": "Can I defer my admission to the next semester at ADDU?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["defer", "admission", "next", "semester", "ADDU"],
                "semantic_category": "admission_deferral"
            },
            {
                "query": "What is the minimum QPI requirement for admission?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["minimum", "QPI", "requirement", "admission"],
                "semantic_category": "academic_requirements"
            },
            {
                "query": "How do I submit my ACAT scores for admission?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["submit", "ACAT", "scores", "admission", "testing"],
                "semantic_category": "document_submission"
            },
            {
                "query": "What are the requirements for early decision applications?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["early", "decision", "applications", "requirements", "deadline"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "Can I apply for multiple programs simultaneously?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["apply", "multiple", "programs", "simultaneously", "application"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "How much is the application fee for undergraduate admission?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["application", "fee", "undergraduate", "admission", "cost"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "How do I request an application fee waiver?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["request", "application", "fee", "waiver", "financial"],
                "semantic_category": "financial_assistance"
            },
            {
                "query": "What documents are required for undergraduate admission at ADDU?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["documents", "required", "undergraduate", "admission", "ADDU"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "How do I submit my TOEFL scores?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["submit", "TOEFL", "scores", "English", "proficiency"],
                "semantic_category": "language_requirements"
            },
            {
                "query": "What is the deadline for second semester applications?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["deadline", "second", "semester", "applications", "admission"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "Can I defer my enrollment to the next academic year?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["defer", "enrollment", "next", "academic", "year"],
                "semantic_category": "admission_deferral"
            },
            {
                "query": "What are the requirements for conditional admission?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["requirements", "conditional", "admission", "criteria", "conditions"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "How do I apply for readmission after academic misconduct?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["apply", "readmission", "academic", "misconduct", "appeal"],
                "semantic_category": "returning_student"
            },
            {
                "query": "What is the process for appealing an admission decision?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["process", "appealing", "admission", "decision", "review"],
                "semantic_category": "admission_status"
            },
            {
                "query": "Are there special requirements for homeschooled students?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["special", "requirements", "homeschooled", "students", "documentation"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "How do I submit official transcripts from multiple institutions?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["submit", "official", "transcripts", "multiple", "institutions"],
                "semantic_category": "document_submission"
            },
            {
                "query": "What are the requirements for regular student admission?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["requirements", "regular", "student", "admission", "enrollment"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "Can I change my program after being admitted?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["change", "program", "admitted", "transfer", "switch"],
                "semantic_category": "admission_status"
            },
            {
                "query": "What are the admission requirements for first year students at ADDU?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["admission", "requirements", "first", "year", "students", "ADDU"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "How do I appeal academic violations?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["appeal", "academic", "violations", "misconduct"],
                "semantic_category": "returning_student"
            },
            {
                "query": "What is the minimum age requirement for admission?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["minimum", "age", "requirement", "admission", "eligibility"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "What are the admission deadlines for different programs?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["admission", "deadlines", "programs", "application"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "How do I submit letters of recommendation electronically?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["submit", "letters", "recommendation", "electronically", "online"],
                "semantic_category": "document_submission"
            },
            {
                "query": "What are the requirements for mature student admission?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["requirements", "mature", "student", "admission", "adult"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "Can I apply for admission while still completing prerequisites?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["apply", "admission", "completing", "prerequisites", "conditional"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "What is the process for international credential evaluation?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["process", "international", "credential", "evaluation", "transcripts"],
                "semantic_category": "international_admission"
            },
            {
                "query": "How do I apply for academic accommodation during admission?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["apply", "academic", "accommodation", "admission", "disability"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "What are the requirements for concurrent enrollment?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["requirements", "concurrent", "enrollment", "multiple", "institutions"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "Can I submit additional materials after the deadline?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["submit", "additional", "materials", "deadline", "late"],
                "semantic_category": "document_submission"
            },
            {
                "query": "What is the process for credit by examination?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["process", "credit", "examination", "testing", "placement"],
                "semantic_category": "academic_requirements"
            },
            {
                "query": "How do I apply for credit transfer from previous university?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["apply", "credit", "transfer", "previous", "university"],
                "semantic_category": "transfer_requirements"
            },
            {
                "query": "What are the requirements for transferee students?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["requirements", "transferee", "students", "enrollment"],
                "semantic_category": "transfer_requirements"
            },
            {
                "query": "Can I apply for multiple semesters at once?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["apply", "multiple", "semesters", "once", "application"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "What is the process for QPI improvement application?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["process", "QPI", "improvement", "application", "grades"],
                "semantic_category": "returning_student"
            },
            {
                "query": "How do I submit proof of immunization?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["submit", "proof", "immunization", "health", "requirements"],
                "semantic_category": "document_submission"
            },
            {
                "query": "What are the requirements for guest student enrollment?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["requirements", "guest", "student", "enrollment", "temporary"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "Can I apply for admission with pending grades?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["apply", "admission", "pending", "grades", "incomplete"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "What is the process for academic reinstatement?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["process", "academic", "reinstatement", "suspension", "appeal"],
                "semantic_category": "returning_student"
            },
            {
                "query": "How do I apply for course credit for life experience?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["apply", "course", "credit", "life", "experience"],
                "semantic_category": "academic_requirements"
            },
            {
                "query": "What are the requirements for audit student enrollment?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["requirements", "audit", "student", "enrollment", "non-credit"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "What are the admission requirements for international students at ADDU?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["admission", "requirements", "international", "students", "ADDU"],
                "semantic_category": "international_admission"
            },
            {
                "query": "What is the process for academic calendar change requests?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["process", "academic", "calendar", "change", "requests"],
                "semantic_category": "admission_requirements"
            },
            {
                "query": "How do I apply for priority course registration?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["apply", "priority", "course", "registration", "enrollment"],
                "semantic_category": "returning_student"
            },
            {
                "query": "What are the thesis requirements for undergraduate students?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["thesis", "requirements", "undergraduate", "students"],
                "semantic_category": "graduation_requirements"
            },
            {
                "query": "Can I apply for graduation with incomplete coursework?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["apply", "graduation", "incomplete", "coursework", "requirements"],
                "semantic_category": "academic_requirements"
            },
            {
                "query": "What is the process for academic misconduct appeal?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["process", "academic", "misconduct", "appeal", "committee"],
                "semantic_category": "returning_student"
            },
            {
                "query": "What are the requirements for completing a thesis at ADDU?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["requirements", "completing", "thesis", "ADDU"],
                "semantic_category": "graduation_requirements"
            },
            
            # Programs & Courses Queries (45 queries)
            {
                "query": "What is the Computer Science curriculum?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["computer", "science", "curriculum", "program"],
                "semantic_category": "program_curriculum"
            },
            {
                "query": "What Engineering programs are offered at ADDU?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["engineering", "programs", "offered", "ADDU"],
                "semantic_category": "program_information"
            },
            {
                "query": "What subjects are in the first year Nursing curriculum at ADDU?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["subjects", "first", "year", "nursing", "curriculum", "ADDU"],
                "semantic_category": "program_curriculum"
            },
            {
                "query": "How many years is the Business Administration program?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["years", "business", "administration", "program", "duration"],
                "semantic_category": "program_duration"
            },
            {
                "query": "What are the major subjects in the Psychology program at ADDU?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["major", "subjects", "psychology", "program", "ADDU"],
                "semantic_category": "program_curriculum"
            },
            {
                "query": "Is there an Information Technology degree program?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["information", "technology", "degree", "program"],
                "semantic_category": "program_availability"
            },
            {
                "query": "What are the clinical requirements for Nursing students?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["clinical", "requirements", "nursing", "students"],
                "semantic_category": "program_requirements"
            },
            {
                "query": "Which programs are available in the School of Medicine?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["programs", "available", "school", "medicine"],
                "semantic_category": "program_information"
            },
            {
                "query": "What mathematics courses are required for Engineering?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["mathematics", "courses", "required", "engineering"],
                "semantic_category": "program_requirements"
            },
            {
                "query": "How many units is the Computer Science program?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["units", "computer", "science", "program"],
                "semantic_category": "program_details"
            },
            {
                "query": "What are the internship requirements for Business Administration students?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["internship", "requirements", "business", "administration", "students"],
                "semantic_category": "program_requirements"
            },
            {
                "query": "What courses are available in the summer semester at ADDU?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["courses", "available", "summer", "semester", "ADDU"],
                "semantic_category": "course_scheduling"
            },
            {
                "query": "What is the curriculum for second year IT students?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["curriculum", "second", "year", "IT", "students"],
                "semantic_category": "program_curriculum"
            },
            {
                "query": "Which programs have the most laboratory work?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["programs", "laboratory", "work"],
                "semantic_category": "program_characteristics"
            },
            {
                "query": "What are the graduation requirements for undergraduate students?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["graduation", "requirements", "undergraduate", "students"],
                "semantic_category": "graduation_requirements"
            },
            {
                "query": "Are there summer classes for Engineering programs?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["summer", "classes", "engineering", "programs"],
                "semantic_category": "course_scheduling"
            },
            {
                "query": "What programming languages are taught in Computer Science?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["programming", "languages", "taught", "computer", "science"],
                "semantic_category": "program_curriculum"
            },
            {
                "query": "How many semesters is the Nursing program?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["semesters", "nursing", "program", "duration"],
                "semantic_category": "program_duration"
            },
            {
                "query": "What are the prerequisites for advanced courses?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["prerequisites", "advanced", "courses"],
                "semantic_category": "course_requirements"
            },
            {
                "query": "What undergraduate degree programs are available at ADDU?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["undergraduate", "degree", "programs", "available", "ADDU"],
                "semantic_category": "program_information"
            },
            {
                "query": "What is the Information Technology curriculum at ADDU?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["information", "technology", "curriculum", "ADDU"],
                "semantic_category": "program_curriculum"
            },
            {
                "query": "Are there online degree programs available?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["online", "degree", "programs", "distance", "learning"],
                "semantic_category": "course_delivery"
            },
            {
                "query": "How many years is the Business Administration program at ADDU?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["years", "business", "administration", "program", "ADDU"],
                "semantic_category": "program_duration"
            },
            {
                "query": "What is the class schedule for first semester at ADDU?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["class", "schedule", "first", "semester", "ADDU"],
                "semantic_category": "course_scheduling"
            },
            {
                "query": "What research opportunities are available in Psychology?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["research", "opportunities", "psychology", "projects", "faculty"],
                "semantic_category": "program_characteristics"
            },
            {
                "query": "What are the elective courses available at ADDU?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["elective", "courses", "available", "ADDU"],
                "semantic_category": "course_requirements"
            },
            {
                "query": "What are the laboratory requirements for Chemistry?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["laboratory", "requirements", "chemistry", "practical", "work"],
                "semantic_category": "course_requirements"
            },
            {
                "query": "Are there accelerated degree programs?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["accelerated", "degree", "programs", "fast", "track"],
                "semantic_category": "program_options"
            },
            {
                "query": "What is the curriculum for Environmental Science?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["curriculum", "environmental", "science", "courses", "program"],
                "semantic_category": "program_curriculum"
            },
            {
                "query": "Are there study abroad opportunities?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["study", "abroad", "opportunities", "international", "exchange"],
                "semantic_category": "program_characteristics"
            },
            {
                "query": "What are the prerequisites for advanced courses?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["prerequisites", "advanced", "courses", "requirements", "foundation"],
                "semantic_category": "course_requirements"
            },
            {
                "query": "What are the course requirements for undergraduate programs?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["course", "requirements", "undergraduate", "programs"],
                "semantic_category": "course_requirements"
            },
            {
                "query": "What is the typical class size for undergraduate courses?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["class", "size", "undergraduate", "courses"],
                "semantic_category": "program_characteristics"
            },
            {
                "query": "Are there certificate programs available?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["certificate", "programs", "professional", "development", "credentials"],
                "semantic_category": "program_options"
            },
            {
                "query": "What software is used in the Digital Media program?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["software", "digital", "media", "program", "technology"],
                "semantic_category": "program_curriculum"
            },
            {
                "query": "Can I change my program after being admitted to ADDU?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["change", "program", "admitted", "ADDU"],
                "semantic_category": "admission_status"
            },
            {
                "query": "What are the clinical requirements for Nursing?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["clinical", "requirements", "nursing", "practical", "experience"],
                "semantic_category": "course_requirements"
            },
            {
                "query": "Are there weekend classes available?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["weekend", "classes", "schedule", "flexible", "options"],
                "semantic_category": "course_scheduling"
            },
            {
                "query": "What is the curriculum structure for Engineering?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["curriculum", "structure", "engineering", "program", "courses"],
                "semantic_category": "program_curriculum"
            },
            {
                "query": "What are the prerequisites for advanced undergraduate courses?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["prerequisites", "advanced", "undergraduate", "courses"],
                "semantic_category": "course_requirements"
            },
            {
                "query": "What are the fieldwork requirements for Social Work?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["fieldwork", "requirements", "social", "work", "practical"],
                "semantic_category": "course_requirements"
            },
            {
                "query": "Are there honors programs available?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["honors", "programs", "academic", "excellence", "distinction"],
                "semantic_category": "program_options"
            },
            {
                "query": "What are the capstone project requirements for undergraduate programs?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["capstone", "project", "requirements", "undergraduate", "programs"],
                "semantic_category": "graduation_requirements"
            },
            {
                "query": "How many units are required to complete an undergraduate degree?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["units", "required", "complete", "undergraduate", "degree"],
                "semantic_category": "program_details"
            },
            {
                "query": "What are the capstone project requirements?",
                "expected_topics": ["programs_courses"],
                "expected_keywords": ["capstone", "project", "requirements", "final", "assessment"],
                "semantic_category": "graduation_requirements"
            },
            
            # Fees & Payment Queries (45 queries)
            {
                "query": "What are the tuition fees for Engineering programs?",
                "expected_topics": ["fees"],
                "expected_keywords": ["tuition", "fees", "engineering", "programs"],
                "semantic_category": "program_fees"
            },
            {
                "query": "How much does it cost to study Computer Science?",
                "expected_topics": ["fees"],
                "expected_keywords": ["cost", "study", "computer", "science", "fees"],
                "semantic_category": "program_fees"
            },
            {
                "query": "Are there payment plans available for tuition?",
                "expected_topics": ["fees"],
                "expected_keywords": ["payment", "plans", "available", "tuition"],
                "semantic_category": "payment_options"
            },
            {
                "query": "What are the miscellaneous fees for students?",
                "expected_topics": ["fees"],
                "expected_keywords": ["miscellaneous", "fees", "students"],
                "semantic_category": "additional_fees"
            },
            {
                "query": "How much are the laboratory fees for Nursing?",
                "expected_topics": ["fees"],
                "expected_keywords": ["laboratory", "fees", "nursing"],
                "semantic_category": "specific_fees"
            },
            {
                "query": "What payment methods are accepted by the university?",
                "expected_topics": ["fees"],
                "expected_keywords": ["payment", "methods", "accepted", "university"],
                "semantic_category": "payment_methods"
            },
            {
                "query": "Is there a discount for early payment of fees?",
                "expected_topics": ["fees"],
                "expected_keywords": ["discount", "early", "payment", "fees"],
                "semantic_category": "payment_discounts"
            },
            {
                "query": "What is the total cost for a 4-year program?",
                "expected_topics": ["fees"],
                "expected_keywords": ["total", "cost", "4-year", "program"],
                "semantic_category": "total_program_cost"
            },
            {
                "query": "Are there additional fees for international students?",
                "expected_topics": ["fees"],
                "expected_keywords": ["additional", "fees", "international", "students"],
                "semantic_category": "international_fees"
            },
            {
                "query": "What is the enrollment fee at ADDU?",
                "expected_topics": ["fees"],
                "expected_keywords": ["enrollment", "fee", "ADDU"],
                "semantic_category": "enrollment_fees"
            },
            {
                "query": "What are the fees for summer classes?",
                "expected_topics": ["fees"],
                "expected_keywords": ["fees", "summer", "classes"],
                "semantic_category": "summer_fees"
            },
            {
                "query": "Can I pay my tuition in installments?",
                "expected_topics": ["fees"],
                "expected_keywords": ["pay", "tuition", "installments"],
                "semantic_category": "payment_options"
            },
            {
                "query": "What is the refund policy for tuition fees?",
                "expected_topics": ["fees"],
                "expected_keywords": ["refund", "policy", "tuition", "fees"],
                "semantic_category": "refund_policy"
            },
            {
                "query": "Are there any hidden costs in the program fees?",
                "expected_topics": ["fees"],
                "expected_keywords": ["hidden", "costs", "program", "fees"],
                "semantic_category": "fee_transparency"
            },
            {
                "query": "What banks support online payment for fees?",
                "expected_topics": ["fees"],
                "expected_keywords": ["banks", "support", "online", "payment", "fees"],
                "semantic_category": "payment_methods"
            },
            {
                "query": "What is the application fee for undergraduate programs at ADDU?",
                "expected_topics": ["fees"],
                "expected_keywords": ["application", "fee", "undergraduate", "programs", "ADDU"],
                "semantic_category": "application_fees"
            },
            {
                "query": "What are the laboratory fees for science programs at ADDU?",
                "expected_topics": ["fees"],
                "expected_keywords": ["laboratory", "fees", "science", "programs", "ADDU"],
                "semantic_category": "additional_fees"
            },
            {
                "query": "What is the refund policy for tuition fees at ADDU?",
                "expected_topics": ["fees"],
                "expected_keywords": ["refund", "policy", "tuition", "fees", "ADDU"],
                "semantic_category": "refund_policy"
            },
            {
                "query": "What are the dormitory fees per semester?",
                "expected_topics": ["fees"],
                "expected_keywords": ["dormitory", "fees", "semester", "housing", "cost"],
                "semantic_category": "additional_fees"
            },
            {
                "query": "Are there late payment penalties for tuition?",
                "expected_topics": ["fees"],
                "expected_keywords": ["late", "payment", "penalties", "tuition", "charges"],
                "semantic_category": "payment_methods"
            },
            {
                "query": "What are the parking fees on campus?",
                "expected_topics": ["fees"],
                "expected_keywords": ["parking", "fees", "campus", "permit", "cost"],
                "semantic_category": "additional_fees"
            },
            {
                "query": "Can I pay fees using a credit card?",
                "expected_topics": ["fees"],
                "expected_keywords": ["pay", "fees", "credit", "card", "payment"],
                "semantic_category": "payment_methods"
            },
            {
                "query": "What are the graduation ceremony fees?",
                "expected_topics": ["fees"],
                "expected_keywords": ["graduation", "ceremony", "fees", "commencement", "cost"],
                "semantic_category": "additional_fees"
            },
            {
                "query": "What are the miscellaneous fees for students at ADDU?",
                "expected_topics": ["fees"],
                "expected_keywords": ["miscellaneous", "fees", "students", "ADDU"],
                "semantic_category": "additional_fees"
            },
            {
                "query": "What is the cost of transcript requests?",
                "expected_topics": ["fees"],
                "expected_keywords": ["cost", "transcript", "requests", "official", "fees"],
                "semantic_category": "specific_fees"
            },
            {
                "query": "Can I set up automatic payments for tuition?",
                "expected_topics": ["fees"],
                "expected_keywords": ["automatic", "payments", "tuition", "recurring", "setup"],
                "semantic_category": "payment_methods"
            },
            {
                "query": "What are the health insurance fees for students?",
                "expected_topics": ["fees"],
                "expected_keywords": ["health", "insurance", "fees", "students", "mandatory"],
                "semantic_category": "additional_fees"
            },
            {
                "query": "Are there discounts for early payment of fees?",
                "expected_topics": ["fees"],
                "expected_keywords": ["discounts", "early", "payment", "fees", "incentives"],
                "semantic_category": "payment_discounts"
            },
            {
                "query": "What are the library fines and fees?",
                "expected_topics": ["fees"],
                "expected_keywords": ["library", "fines", "fees", "overdue", "books"],
                "semantic_category": "additional_fees"
            },
            {
                "query": "What are the payment methods for tuition fees at ADDU?",
                "expected_topics": ["fees"],
                "expected_keywords": ["payment", "methods", "tuition", "fees", "ADDU"],
                "semantic_category": "payment_methods"
            },
            {
                "query": "What are the student activity fees used for?",
                "expected_topics": ["fees"],
                "expected_keywords": ["student", "activity", "fees", "services", "programs"],
                "semantic_category": "additional_fees"
            },
            {
                "query": "Are there fees for course add/drop after deadline?",
                "expected_topics": ["fees"],
                "expected_keywords": ["fees", "course", "add", "drop", "deadline"],
                "semantic_category": "additional_fees"
            },
            {
                "query": "What is the total cost for a four-year program?",
                "expected_topics": ["fees"],
                "expected_keywords": ["total", "cost", "four-year", "program", "estimate"],
                "semantic_category": "total_program_cost"
            },
            {
                "query": "Can I pay fees through bank transfer?",
                "expected_topics": ["fees"],
                "expected_keywords": ["pay", "fees", "bank", "transfer", "wire"],
                "semantic_category": "payment_methods"
            },
            {
                "query": "What are the recreation center membership fees?",
                "expected_topics": ["fees"],
                "expected_keywords": ["recreation", "center", "membership", "fees", "facilities"],
                "semantic_category": "additional_fees"
            },
            {
                "query": "Are there fees for makeup examinations?",
                "expected_topics": ["fees"],
                "expected_keywords": ["fees", "makeup", "examinations", "testing", "cost"],
                "semantic_category": "additional_fees"
            },
            {
                "query": "What payment plans are available for tuition?",
                "expected_topics": ["fees"],
                "expected_keywords": ["payment", "plans", "tuition", "installment", "options"],
                "semantic_category": "payment_options"
            },
            {
                "query": "Can I pay fees using financial aid?",
                "expected_topics": ["fees"],
                "expected_keywords": ["pay", "fees", "financial", "aid", "scholarship"],
                "semantic_category": "payment_methods"
            },
            {
                "query": "What are the fees for ID card replacement?",
                "expected_topics": ["fees"],
                "expected_keywords": ["fees", "ID", "card", "replacement", "student"],
                "semantic_category": "additional_fees"
            },
            {
                "query": "What are the total fees for first semester at ADDU?",
                "expected_topics": ["fees"],
                "expected_keywords": ["total", "fees", "first", "semester", "ADDU"],
                "semantic_category": "total_program_cost"
            },
            {
                "query": "What are the fees for thesis binding and submission at ADDU?",
                "expected_topics": ["fees"],
                "expected_keywords": ["fees", "thesis", "binding", "submission", "ADDU"],
                "semantic_category": "additional_fees"
            },
            {
                "query": "Can I get a fee waiver for financial hardship?",
                "expected_topics": ["fees"],
                "expected_keywords": ["fee", "waiver", "financial", "hardship", "assistance"],
                "semantic_category": "payment_discounts"
            },
            {
                "query": "What are the conference attendance fees for students?",
                "expected_topics": ["fees"],
                "expected_keywords": ["conference", "attendance", "fees", "students", "academic"],
                "semantic_category": "additional_fees"
            },
            {
                "query": "What are the fee structures for different programs at ADDU?",
                "expected_topics": ["fees"],
                "expected_keywords": ["fee", "structures", "different", "programs", "ADDU"],
                "semantic_category": "program_fees"
            },
            {
                "query": "What is the minimum QPI required for admission to ADDU?",
                "expected_topics": ["admissions_enrollment"],
                "expected_keywords": ["minimum", "QPI", "required", "admission", "ADDU"],
                "semantic_category": "academic_requirements"
            }
        ]
        
        # 100 NEGATIVE test queries - challenging cases that should NOT be relevant
        self.negative_test_queries = [
            # ADDU-Adjacent but Irrelevant Queries (15 queries)
            {
                "query": "What sports teams does ADDU have?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "addu_sports",
                "should_be_relevant": False
            },
            {
                "query": "Where is the ADDU cafeteria located?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "addu_facilities",
                "should_be_relevant": False
            },
            {
                "query": "What time does the ADDU library close?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "addu_facilities",
                "should_be_relevant": False
            },
            {
                "query": "How do I join ADDU student organizations?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "addu_student_life",
                "should_be_relevant": False
            },
            {
                "query": "What are ADDU's research publications?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "addu_research",
                "should_be_relevant": False
            },
            {
                "query": "Who is the current ADDU president?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "addu_administration",
                "should_be_relevant": False
            },
            {
                "query": "What is ADDU's history and founding?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "addu_history",
                "should_be_relevant": False
            },
            {
                "query": "How do I contact ADDU alumni association?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "addu_alumni",
                "should_be_relevant": False
            },
            {
                "query": "What events are happening at ADDU this week?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "addu_events",
                "should_be_relevant": False
            },
            {
                "query": "Where can I park at ADDU campus?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "addu_facilities",
                "should_be_relevant": False
            },
            {
                "query": "What are ADDU's campus safety protocols?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "addu_safety",
                "should_be_relevant": False
            },
            {
                "query": "How do I access ADDU's WiFi network?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "addu_technology",
                "should_be_relevant": False
            },
            {
                "query": "What dining options are available at ADDU?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "addu_dining",
                "should_be_relevant": False
            },
            {
                "query": "How do I report a maintenance issue at ADDU?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "addu_maintenance",
                "should_be_relevant": False
            },
            {
                "query": "What are ADDU's sustainability initiatives?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "addu_sustainability",
                "should_be_relevant": False
            },
            
            # Other Universities' Admissions (15 queries)
            {
                "query": "What are the admission requirements for Ateneo de Manila?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "other_university_admissions",
                "should_be_relevant": False
            },
            {
                "query": "How do I apply to University of the Philippines?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "other_university_admissions",
                "should_be_relevant": False
            },
            {
                "query": "What programs does De La Salle University offer?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "other_university_programs",
                "should_be_relevant": False
            },
            {
                "query": "What are the fees at University of San Carlos?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "other_university_fees",
                "should_be_relevant": False
            },
            {
                "query": "How do I transfer to Silliman University?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "other_university_transfer",
                "should_be_relevant": False
            },
            {
                "query": "What are the scholarship requirements at Xavier University?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "other_university_scholarships",
                "should_be_relevant": False
            },
            {
                "query": "How do I apply for financial aid at Mindanao State University?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "other_university_financial_aid",
                "should_be_relevant": False
            },
            {
                "query": "What is the application deadline for Central Philippine University?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "other_university_deadlines",
                "should_be_relevant": False
            },
            {
                "query": "What documents do I need for Holy Angel University?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "other_university_documents",
                "should_be_relevant": False
            },
            {
                "query": "How much is tuition at Far Eastern University?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "other_university_tuition",
                "should_be_relevant": False
            },
            {
                "query": "What are the entrance exam requirements for Mapua University?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "other_university_exams",
                "should_be_relevant": False
            },
            {
                "query": "Can I apply online to Lyceum of the Philippines?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "other_university_application",
                "should_be_relevant": False
            },
            {
                "query": "What is the GPA requirement for San Beda University?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "other_university_gpa",
                "should_be_relevant": False
            },
            {
                "query": "How do I check my application status at Adamson University?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "other_university_status",
                "should_be_relevant": False
            },
            {
                "query": "What are the English proficiency requirements for Miriam College?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "other_university_english",
                "should_be_relevant": False
            },
            
            # Graduate/Post-Graduate Programs (15 queries)
            {
                "query": "What are the Master's degree requirements at ADDU?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "graduate_programs",
                "should_be_relevant": False
            },
            {
                "query": "How do I apply for PhD programs?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "doctoral_programs",
                "should_be_relevant": False
            },
            {
                "query": "What are the graduate school admission requirements?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "graduate_admissions",
                "should_be_relevant": False
            },
            {
                "query": "Are there MBA programs available?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "mba_programs",
                "should_be_relevant": False
            },
            {
                "query": "What are the doctoral program fees?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "doctoral_fees",
                "should_be_relevant": False
            },
            {
                "query": "How do I apply for a Master's in Education?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "masters_education",
                "should_be_relevant": False
            },
            {
                "query": "What are the thesis requirements for graduate programs?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "graduate_thesis",
                "should_be_relevant": False
            },
            {
                "query": "Can I pursue a PhD in Engineering at ADDU?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "phd_engineering",
                "should_be_relevant": False
            },
            {
                "query": "What are the research requirements for doctoral students?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "doctoral_research",
                "should_be_relevant": False
            },
            {
                "query": "How long does it take to complete a Master's degree?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "masters_duration",
                "should_be_relevant": False
            },
            {
                "query": "What are the prerequisites for graduate school?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "graduate_prerequisites",
                "should_be_relevant": False
            },
            {
                "query": "Can I do part-time graduate studies?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "part_time_graduate",
                "should_be_relevant": False
            },
            {
                "query": "What is the application deadline for Master's programs?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "masters_deadline",
                "should_be_relevant": False
            },
            {
                "query": "Are there graduate scholarships available?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "graduate_scholarships",
                "should_be_relevant": False
            },
            {
                "query": "What are the comprehensive exam requirements?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "comprehensive_exams",
                "should_be_relevant": False
            },
            
            # Employment/Career at ADDU (15 queries)
            {
                "query": "How do I apply for a teaching position at ADDU?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "addu_employment",
                "should_be_relevant": False
            },
            {
                "query": "What are the faculty requirements?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "faculty_requirements",
                "should_be_relevant": False
            },
            {
                "query": "How do I submit my resume to ADDU HR?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "addu_hr",
                "should_be_relevant": False
            },
            {
                "query": "What are the staff benefits at ADDU?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "staff_benefits",
                "should_be_relevant": False
            },
            {
                "query": "Are there job openings at ADDU?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "job_openings",
                "should_be_relevant": False
            },
            {
                "query": "What is the salary range for ADDU professors?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "professor_salary",
                "should_be_relevant": False
            },
            {
                "query": "How do I apply for administrative positions at ADDU?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "admin_positions",
                "should_be_relevant": False
            },
            {
                "query": "What are the working hours for ADDU staff?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "working_hours",
                "should_be_relevant": False
            },
            {
                "query": "Does ADDU offer health insurance for employees?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "employee_insurance",
                "should_be_relevant": False
            },
            {
                "query": "How do I apply for a research position at ADDU?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "research_positions",
                "should_be_relevant": False
            },
            {
                "query": "What are the promotion criteria for ADDU faculty?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "faculty_promotion",
                "should_be_relevant": False
            },
            {
                "query": "Can I work part-time at ADDU?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "part_time_work",
                "should_be_relevant": False
            },
            {
                "query": "What are the retirement benefits for ADDU employees?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "retirement_benefits",
                "should_be_relevant": False
            },
            {
                "query": "How do I request leave as an ADDU employee?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "employee_leave",
                "should_be_relevant": False
            },
            {
                "query": "What training programs does ADDU offer for staff?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "staff_training",
                "should_be_relevant": False
            },
            
            # Borderline Academic Queries (15 queries)
            {
                "query": "What is the difference between semester and trimester systems?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "academic_systems",
                "should_be_relevant": False
            },
            {
                "query": "How do I improve my study habits in college?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "study_habits",
                "should_be_relevant": False
            },
            {
                "query": "What are the benefits of online learning?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "online_learning",
                "should_be_relevant": False
            },
            {
                "query": "How do I choose between different universities?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "university_selection",
                "should_be_relevant": False
            },
            {
                "query": "What should I consider when selecting a major?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "major_selection",
                "should_be_relevant": False
            },
            {
                "query": "What is the difference between public and private universities?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "university_types",
                "should_be_relevant": False
            },
            {
                "query": "How do I prepare for college entrance exams?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "entrance_exam_prep",
                "should_be_relevant": False
            },
            {
                "query": "What are the best study techniques for exams?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "study_techniques",
                "should_be_relevant": False
            },
            {
                "query": "How do I write a college application essay?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "application_essay",
                "should_be_relevant": False
            },
            {
                "query": "What are the best universities in the Philippines?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "university_rankings",
                "should_be_relevant": False
            },
            {
                "query": "How do I balance work and school?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "work_life_balance",
                "should_be_relevant": False
            },
            {
                "query": "What are the benefits of studying abroad?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "study_abroad_benefits",
                "should_be_relevant": False
            },
            {
                "query": "How do I get letters of recommendation?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "recommendation_letters",
                "should_be_relevant": False
            },
            {
                "query": "What is the FAFSA and how do I fill it out?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "financial_aid_general",
                "should_be_relevant": False
            },
            {
                "query": "How do I prepare for graduate school?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "graduate_preparation",
                "should_be_relevant": False
            },
            
            # Keep some obviously off-topic cases for baseline (25 queries)
            {
                "query": "What's the weather like today?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "weather",
                "should_be_relevant": False
            },
            {
                "query": "How do I cook pasta?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "cooking",
                "should_be_relevant": False
            },
            {
                "query": "Who won the football game last night?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "sports",
                "should_be_relevant": False
            },
            {
                "query": "What movies are playing tonight?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "entertainment",
                "should_be_relevant": False
            },
            {
                "query": "Where can I buy shoes?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "shopping",
                "should_be_relevant": False
            },
            {
                "query": "How do I get to Paris?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "travel",
                "should_be_relevant": False
            },
            {
                "query": "How do I fix my computer?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "technology",
                "should_be_relevant": False
            },
            {
                "query": "I have a headache, what should I do?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "health",
                "should_be_relevant": False
            },
            {
                "query": "What's your favorite color?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "personal",
                "should_be_relevant": False
            },
            {
                "query": "What's the capital of France?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "geography",
                "should_be_relevant": False
            },
            {
                "query": "How fast can a cheetah run?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "animals",
                "should_be_relevant": False
            },
            {
                "query": "How do I clean my house?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "household",
                "should_be_relevant": False
            },
            {
                "query": "What should I wear today?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "fashion",
                "should_be_relevant": False
            },
            {
                "query": "How do I invest in stocks?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "finance",
                "should_be_relevant": False
            },
            {
                "query": "Will it rain tomorrow?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "weather",
                "should_be_relevant": False
            },
            {
                "query": "What's the best pizza recipe?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "cooking",
                "should_be_relevant": False
            },
            {
                "query": "What's the basketball score?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "sports",
                "should_be_relevant": False
            },
            {
                "query": "What's on TV right now?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "entertainment",
                "should_be_relevant": False
            },
            {
                "query": "What's the price of this phone?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "shopping",
                "should_be_relevant": False
            },
            {
                "query": "What's the best hotel in Tokyo?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "travel",
                "should_be_relevant": False
            },
            {
                "query": "What's the best video game?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "gaming",
                "should_be_relevant": False
            },
            {
                "query": "What's the best medicine for cold?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "health",
                "should_be_relevant": False
            },
            {
                "query": "Do you have any friends?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "personal",
                "should_be_relevant": False
            },
            {
                "query": "How many planets are in our solar system?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "astronomy",
                "should_be_relevant": False
            },
            {
                "query": "What do elephants eat?",
                "expected_topics": [],
                "expected_keywords": [],
                "semantic_category": "animals",
                "should_be_relevant": False
            }
        ]
    

    def send_guided_chat_request(self, user_input: str, action_type: str = "message", 
                               action_data: str = None, session_id: str = None) -> Dict:
        """Send request to guided chat endpoint"""
        endpoint = f"{self.base_url}/chat/guided/"
        payload = {
            "user_input": user_input,
            "action_type": action_type, 
            "action_data": action_data,
            "session_id": session_id
        }
        
        try:
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"ERROR: Request failed: {e}")
            return None
    
    def evaluate_semantic_relevance(self, query_data: Dict, retrieved_docs: List[Dict]) -> Dict:
        """Evaluate semantic relevance of retrieved documents"""
        expected_topics = query_data["expected_topics"]
        expected_keywords = query_data["expected_keywords"]
        semantic_category = query_data["semantic_category"]
        should_be_relevant = query_data.get("should_be_relevant", True)
        
        # Check topic relevance
        topic_matches = 0
        for doc in retrieved_docs:
            doc_topic = doc.get('current_topic', '')
            if doc_topic in expected_topics:
                topic_matches += 1
        
        # Fix topic_relevance calculation to stay within bounds [0, 1]
        # Normalize by expected topics, but cap at 1.0 to prevent over-scoring
        if expected_topics:
            topic_relevance = min(topic_matches / len(expected_topics), 1.0)
        else:
            # For negative cases with no expected topics, topic relevance should be 0
            topic_relevance = 0
        
        # Check keyword relevance with improved matching
        keyword_matches = 0
        total_content = ""
        for doc in retrieved_docs:
            content = doc.get('content', '').lower()
            total_content += " " + content
        
        # Improved keyword matching - case insensitive and partial matches
        for keyword in expected_keywords:
            keyword_lower = keyword.lower()
            # Check for exact match or partial match (for compound keywords)
            if keyword_lower in total_content or any(part in total_content for part in keyword_lower.split()):
                keyword_matches += 1
        
        keyword_relevance = keyword_matches / len(expected_keywords) if expected_keywords else 0
        
        # Check retrieval strategy (should use hybrid)
        hybrid_used = any(
            'hybrid' in doc.get('retrieval_strategy', '').lower() or
            doc.get('_debug', {}).get('tfidf_word2vec_used', False)
            for doc in retrieved_docs
        )
        
        # Calculate semantic scores from debug info if available
        semantic_scores = []
        tfidf_scores = []
        for doc in retrieved_docs:
            debug_info = doc.get('_debug', {})
            if 'semantic_similarity' in debug_info:
                semantic_scores.append(debug_info['semantic_similarity'])
            if 'specialized_score' in debug_info:
                tfidf_scores.append(debug_info['specialized_score'])
        
        avg_semantic_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0
        avg_tfidf_score = sum(tfidf_scores) / len(tfidf_scores) if tfidf_scores else 0
        
        # Document count penalty/bonus
        doc_count_factor = 1.0
        if should_be_relevant:
            # For positive cases, penalize if no documents retrieved or too many irrelevant docs
            if len(retrieved_docs) == 0:
                doc_count_factor = 0.1  # Heavy penalty for no retrieval
            elif len(retrieved_docs) > 10:
                doc_count_factor = 0.8  # Light penalty for too many docs
        else:
            # For negative cases, penalize if many documents retrieved (indicates potential false positive)
            # But keep factor reasonable to avoid artificially inflating scores
            if len(retrieved_docs) == 0:
                doc_count_factor = 0.1  # Excellent - no irrelevant docs retrieved
            elif len(retrieved_docs) <= 2:
                doc_count_factor = 0.5  # Good - few docs retrieved
            elif len(retrieved_docs) >= 5:
                doc_count_factor = 1.2  # Slight penalty - many docs might indicate false positive
                # Cap at 1.2 to avoid making clearly irrelevant queries appear relevant
        
        # FALSE POSITIVE DETECTION FOR NEGATIVE CASES
        false_positive_penalty = 0.0
        false_positive_docs = 0
        
        if not should_be_relevant:
            # Define what constitutes "relevant" topics that should NOT be retrieved for negative queries
            chatbot_relevant_topics = ["admissions_enrollment", "programs_courses", "fees"]
            
            # Count how many retrieved docs are from chatbot-relevant topics
            for doc in retrieved_docs:
                doc_topic = doc.get('current_topic', '')
                if doc_topic in chatbot_relevant_topics:
                    false_positive_docs += 1
            
            # Calculate false positive penalty
            if retrieved_docs and false_positive_docs > 0:
                # Higher penalty for more false positive docs
                false_positive_ratio = false_positive_docs / len(retrieved_docs)
                
                # Scale penalty based on how many relevant docs were incorrectly retrieved
                if false_positive_ratio >= 0.8:  # 80%+ of docs are from relevant topics
                    false_positive_penalty = 0.6  # Strong penalty - likely false positive
                elif false_positive_ratio >= 0.5:  # 50%+ of docs are from relevant topics
                    false_positive_penalty = 0.4  # Moderate penalty
                elif false_positive_ratio >= 0.3:  # 30%+ of docs are from relevant topics
                    false_positive_penalty = 0.2  # Light penalty
                else:
                    false_positive_penalty = 0.1  # Very light penalty
        
        # Overall relevance score calculation
        if should_be_relevant:
            # For positive cases: use standard weighted combination
            base_relevance = (topic_relevance * 0.4) + (keyword_relevance * 0.4) + (avg_semantic_score * 0.2)
            overall_relevance = base_relevance * doc_count_factor
        else:
            # For negative cases: base relevance + false positive penalty
            base_relevance = (topic_relevance * 0.4) + (keyword_relevance * 0.4) + (avg_semantic_score * 0.2)
            overall_relevance = base_relevance + false_positive_penalty
        
        # Ensure overall_relevance stays within [0, 1] bounds
        overall_relevance = max(0.0, min(1.0, overall_relevance))
        
        # Binary classification: relevant if overall score >= 0.5 (consistent threshold)
        is_relevant = overall_relevance >= 0.5
        
        return {
            "is_relevant": is_relevant,
            "overall_relevance": overall_relevance,
            "base_relevance": base_relevance,
            "doc_count_factor": doc_count_factor,
            "false_positive_penalty": false_positive_penalty,
            "false_positive_docs": false_positive_docs,
            "topic_relevance": topic_relevance,
            "keyword_relevance": keyword_relevance,
            "avg_semantic_score": avg_semantic_score,
            "avg_tfidf_score": avg_tfidf_score,
            "hybrid_used": hybrid_used,
            "topic_matches": topic_matches,
            "keyword_matches": keyword_matches,
            "semantic_category": semantic_category,
            "retrieved_count": len(retrieved_docs),
            "should_be_relevant": should_be_relevant
        }
    
    def run_semantic_evaluation(self) -> Dict:
        """Run semantic relevance evaluation on all test queries"""
        print("Starting Hybrid Retrieval Semantic Relevance Evaluation")
        print("Target: F1-Score >= 0.70 on semantic relevance tests")
        print("Testing: TF-IDF + Word2Vec hybrid retrieval system")
        
        # Combine positive and negative queries
        all_queries = []
        
        # Add positive queries (should be relevant)
        for query_data in self.test_queries:
            query_copy = query_data.copy()
            query_copy["should_be_relevant"] = True
            query_copy["test_type"] = "positive"
            all_queries.append(query_copy)
        
        # Add negative queries (should NOT be relevant)
        for query_data in self.negative_test_queries:
            all_queries.append({**query_data, "test_type": "negative"})
        
        print(f"Total Test Queries: {len(all_queries)} ({len(self.test_queries)} positive, {len(self.negative_test_queries)} negative)")
        print("="*80)
        
        results = []
        y_true = []  # Ground truth: 1 for should be relevant, 0 for should not be relevant
        y_pred = []  # Predictions from retrieval system
        
        category_results = {}
        
        for i, query_data in enumerate(all_queries, 1):
            query = query_data["query"]
            expected_topics = query_data["expected_topics"]
            semantic_category = query_data["semantic_category"]
            should_be_relevant = query_data["should_be_relevant"]
            test_type = query_data["test_type"]
            
            print(f"\nQuery {i}/{len(all_queries)}: [{test_type.upper()}] {query}")
            print(f"Expected Topics: {expected_topics} (Should be relevant: {should_be_relevant})")
            print(f"Category: {semantic_category}")
            
            # Initialize session with appropriate topic
            if should_be_relevant and expected_topics:
                topic_to_test = expected_topics[0]
            else:
                # For negative queries, use a random topic to test robustness
                topics = ["admissions_enrollment", "programs_courses", "fees"]
                topic_to_test = topics[i % len(topics)]
            
            session_id = f"semantic_test_{i}"
            
            # Select topic
            topic_response = self.send_guided_chat_request("", "topic_selection", topic_to_test, session_id)
            if not topic_response:
                print(f"ERROR: Failed to initialize session for query {i}")
                continue
            
            print(f"Topic selected: {topic_response.get('current_topic')}")
            
            # Send query and get retrieval results
            start_time = time.time()
            response_data = self.send_guided_chat_request(query, "message", None, session_id)
            response_time = time.time() - start_time
            
            if not response_data:
                print(f"ERROR: No response for query {i}")
                y_true.append(1 if should_be_relevant else 0)
                y_pred.append(0)  # No response = not relevant
                continue
            
            # Extract retrieval information
            retrieved_docs = response_data.get('sources', [])
            print(f"Retrieved {len(retrieved_docs)} documents ({response_time:.2f}s)")
            
            # Evaluate semantic relevance
            evaluation = self.evaluate_semantic_relevance(query_data, retrieved_docs)
            
            # Store results
            result = {
                "query_id": i,
                "query": query,
                "expected_topics": expected_topics,
                "should_be_relevant": should_be_relevant,
                "test_type": test_type,
                "semantic_category": semantic_category,
                "evaluation": evaluation,
                "retrieved_docs": len(retrieved_docs),
                "response_time": response_time
            }
            results.append(result)
            
            # For F1 calculation with proper binary classification
            y_true.append(1 if should_be_relevant else 0)
            
            # Use consistent threshold (0.5) for both positive and negative cases
            # This ensures that the same relevance criteria apply to all queries
            y_pred.append(1 if evaluation["is_relevant"] else 0)
            
            # Category tracking
            if semantic_category not in category_results:
                category_results[semantic_category] = {"correct": 0, "total": 0}
            category_results[semantic_category]["total"] += 1
            
            # Check if prediction was correct
            prediction_correct = (y_true[-1] == y_pred[-1])
            if prediction_correct:
                category_results[semantic_category]["correct"] += 1
            
            # Print evaluation results
            status = "PASS" if prediction_correct else "FAIL"
            print(f"{status} Relevance: {evaluation['overall_relevance']:.3f}")
            print(f"   Topic Match: {evaluation['topic_relevance']:.3f}")
            print(f"   Keyword Match: {evaluation['keyword_relevance']:.3f}")
            print(f"   Semantic Score: {evaluation['avg_semantic_score']:.3f}")
            print(f"   Hybrid Used: {evaluation['hybrid_used']}")
            
            # Add false positive information for negative cases
            if not should_be_relevant:
                print(f"   False Positive Docs: {evaluation['false_positive_docs']}/{evaluation['retrieved_count']}")
                print(f"   False Positive Penalty: {evaluation['false_positive_penalty']:.3f}")
            
            print(f"   Prediction Correct: {prediction_correct}")
            
            # Small delay between queries
            time.sleep(0.5)
        
        # Calculate overall metrics using proper binary classification
        if len(y_true) > 0:
            # Calculate True Positives, False Positives, True Negatives, False Negatives
            tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
            fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
            tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
            fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
            
            # Calculate precision, recall, and F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / len(y_true)
        # Calculate per-category F1 scores
            category_f1_scores = {}
            for category in set(r["semantic_category"] for r in results):
                cat_results = [r for r in results if r["semantic_category"] == category]
                cat_y_true = [1 if r["should_be_relevant"] else 0 for r in cat_results]
                cat_y_pred = [1 if r["evaluation"]["is_relevant"] else 0 for r in cat_results]
                
                if len(cat_y_true) > 0:
                    cat_tp = sum(1 for true, pred in zip(cat_y_true, cat_y_pred) if true == 1 and pred == 1)
                    cat_fp = sum(1 for true, pred in zip(cat_y_true, cat_y_pred) if true == 0 and pred == 1)
                    cat_fn = sum(1 for true, pred in zip(cat_y_true, cat_y_pred) if true == 1 and pred == 0)
                    
                    cat_precision = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0.0
                    cat_recall = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0.0
                    cat_f1 = 2 * (cat_precision * cat_recall) / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0.0
                    
                    category_f1_scores[category] = {
                        "precision": cat_precision,
                        "recall": cat_recall,
                        "f1_score": cat_f1,
                        "total_queries": len(cat_results),
                        "positive_queries": sum(cat_y_true),
                        "negative_queries": len(cat_y_true) - sum(cat_y_true)
                    }
        else:
            precision = recall = f1 = accuracy = 0.0
            tp = fp = tn = fn = 0
            category_f1_scores = {}
        
        # Analyze misclassified cases for detailed reporting
        misclassified_cases = []
        for i, (result, true_label, pred_label) in enumerate(zip(results, y_true, y_pred)):
            if true_label != pred_label:
                misclassified_cases.append({
                    "query_id": result["query_id"],
                    "query": result["query"],
                    "expected": "relevant" if true_label == 1 else "irrelevant",
                    "predicted": "relevant" if pred_label == 1 else "irrelevant",
                    "overall_relevance": result["evaluation"]["overall_relevance"],
                    "semantic_category": result["semantic_category"],
                    "test_type": result["test_type"]
                })
        
        return {
            "results": results,
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
                "target_f1": 0.70,
                "target_achieved": f1 >= 0.70,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn
            },
            "category_results": category_results,
            "category_f1_scores": category_f1_scores,
            "misclassified_cases": misclassified_cases,
            "total_queries": len(all_queries),
            "positive_queries": len(self.test_queries),
            "negative_queries": len(self.negative_test_queries),
            "y_true": y_true,
            "y_pred": y_pred
        }
    
    def print_results(self, results: Dict):
        """Print comprehensive evaluation results"""
        print(f"\n{'='*80}")
        print("HYBRID RETRIEVAL SEMANTIC RELEVANCE EVALUATION RESULTS")
        print(f"{'='*80}")
        
        metrics = results["metrics"]
        print(f"Target F1-Score: {metrics['target_f1']}")
        print(f"Achieved F1-Score: {metrics['f1_score']:.4f}")
        print(f"Target Achieved: {'YES' if metrics['target_achieved'] else 'NO'}")
        print(f"Total Queries: {results['total_queries']}")
        print(f"Positive Queries: {results['positive_queries']}")
        print(f"Negative Queries: {results['negative_queries']}")
        
        print(f"\nDETAILED METRICS:")
        print(f"   Precision: {metrics['precision']:.4f}")
        if metrics['precision'] == 1.0:
            print(f"      WARNING: Precision = 1.0 means NO false positives (FP=0)")
            print(f"      This indicates the system never incorrectly classifies irrelevant queries as relevant.")
            print(f"      While good, this may indicate:")
            print(f"      - Negative test cases are too obviously off-topic")
            print(f"      - System is genuinely good at filtering irrelevant queries")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        
        print(f"\nCONFUSION MATRIX:")
        print(f"   True Positives:  {metrics['true_positives']}")
        print(f"   False Positives: {metrics['false_positives']}")
        print(f"   True Negatives:  {metrics['true_negatives']}")
        print(f"   False Negatives: {metrics['false_negatives']}")
        
        # Category breakdown
        print(f"\nSEMANTIC CATEGORY PERFORMANCE:")
        category_results = results["category_results"]
        for category, stats in category_results.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            status = "PASS" if accuracy >= 0.7 else "FAIL"
            print(f"   {status} {category}: {accuracy:.3f} ({stats['correct']}/{stats['total']})")
        
        # Hybrid system analysis
        print(f"\nHYBRID SYSTEM ANALYSIS:")
        hybrid_usage = sum(1 for r in results["results"] if r["evaluation"]["hybrid_used"])
        print(f"   Hybrid Retrieval Used: {hybrid_usage}/{results['total_queries']} queries")
        
        avg_semantic_score = sum(r["evaluation"]["avg_semantic_score"] for r in results["results"]) / len(results["results"])
        avg_tfidf_score = sum(r["evaluation"]["avg_tfidf_score"] for r in results["results"]) / len(results["results"])
        
        print(f"   Average Semantic (Word2Vec) Score: {avg_semantic_score:.3f}")
        print(f"   Average TF-IDF Score: {avg_tfidf_score:.3f}")
        
        # Per-category F1 scores
        if "category_f1_scores" in results:
            print(f"\nPER-CATEGORY F1 SCORES:")
            category_f1_scores = results["category_f1_scores"]
            for category, scores in category_f1_scores.items():
                status = "PASS" if scores["f1_score"] >= 0.7 else "FAIL"
                print(f"   {status} {category}:")
                print(f"      F1: {scores['f1_score']:.3f} | P: {scores['precision']:.3f} | R: {scores['recall']:.3f}")
                print(f"      Queries: {scores['total_queries']} (+{scores['positive_queries']} -{scores['negative_queries']})")
        
        # Misclassified cases analysis
        if "misclassified_cases" in results and results["misclassified_cases"]:
            print(f"\nMISCLASSIFIED CASES ({len(results['misclassified_cases'])} total):")
            # Group by error type
            false_positives = [c for c in results["misclassified_cases"] if c["expected"] == "irrelevant" and c["predicted"] == "relevant"]
            false_negatives = [c for c in results["misclassified_cases"] if c["expected"] == "relevant" and c["predicted"] == "irrelevant"]
            
            if false_positives:
                print(f"   False Positives ({len(false_positives)}):")
                for case in false_positives[:5]:  # Show first 5
                    print(f"      - [{case['semantic_category']}] \"{case['query'][:60]}...\" (score: {case['overall_relevance']:.3f})")
                if len(false_positives) > 5:
                    print(f"      ... and {len(false_positives) - 5} more")
            
            if false_negatives:
                print(f"   False Negatives ({len(false_negatives)}):")
                for case in false_negatives[:5]:  # Show first 5
                    print(f"      - [{case['semantic_category']}] \"{case['query'][:60]}...\" (score: {case['overall_relevance']:.3f})")
                if len(false_negatives) > 5:
                    print(f"      ... and {len(false_negatives) - 5} more")
        
        # Performance by topic
        print(f"\nPERFORMANCE BY TOPIC:")
        topic_stats = {}
        for result in results["results"]:
            for topic in result["expected_topics"]:
                if topic not in topic_stats:
                    topic_stats[topic] = {"correct": 0, "total": 0}
                topic_stats[topic]["total"] += 1
                if result["evaluation"]["is_relevant"]:
                    topic_stats[topic]["correct"] += 1
        
        for topic, stats in topic_stats.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            status = "PASS" if accuracy >= 0.7 else "FAIL"
            print(f"   {status} {topic}: {accuracy:.3f} ({stats['correct']}/{stats['total']})")
        
        # Threshold sensitivity analysis
        print(f"\nTHRESHOLD SENSITIVITY ANALYSIS:")
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        print("   Threshold | Precision | Recall | F1-Score")
        print("   ----------|-----------|--------|----------")
        
        for threshold in thresholds:
            # Recalculate metrics with different threshold
            y_true_thresh = [1 if r["should_be_relevant"] else 0 for r in results["results"]]
            y_pred_thresh = [1 if r["evaluation"]["overall_relevance"] >= threshold else 0 for r in results["results"]]
            
            tp_thresh = sum(1 for true, pred in zip(y_true_thresh, y_pred_thresh) if true == 1 and pred == 1)
            fp_thresh = sum(1 for true, pred in zip(y_true_thresh, y_pred_thresh) if true == 0 and pred == 1)
            fn_thresh = sum(1 for true, pred in zip(y_true_thresh, y_pred_thresh) if true == 1 and pred == 0)
            
            precision_thresh = tp_thresh / (tp_thresh + fp_thresh) if (tp_thresh + fp_thresh) > 0 else 0.0
            recall_thresh = tp_thresh / (tp_thresh + fn_thresh) if (tp_thresh + fn_thresh) > 0 else 0.0
            f1_thresh = 2 * (precision_thresh * recall_thresh) / (precision_thresh + recall_thresh) if (precision_thresh + recall_thresh) > 0 else 0.0
            
            marker = " *" if threshold == 0.5 else "  "
            print(f"   {threshold:>9.1f} | {precision_thresh:>9.3f} | {recall_thresh:>6.3f} | {f1_thresh:>8.3f}{marker}")

def main():
    """Main test runner"""
    print("Hybrid Retrieval System Semantic Relevance Evaluation")
    print("WARNING: Make sure Django server is running: python manage.py runserver")
    print("WARNING: This test evaluates the TF-IDF + Word2Vec hybrid retrieval system")
    print("WARNING: Testing semantic relevance with challenging negative cases for realistic F1 scores")
    print("WARNING: Testing across 249 varied user queries (149 positive + 100 negative)")
    
    print("\nStarting evaluation...")
    
    evaluator = HybridRetrievalSemanticEvaluator()
    
    try:
        results = evaluator.run_semantic_evaluation()
        evaluator.print_results(results)
        
        # Save results to file
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_file = os.path.join(script_dir, 'hybrid_retrieval_semantic_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Exit with appropriate code
        sys.exit(0 if results['metrics']['target_achieved'] else 1)
        
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

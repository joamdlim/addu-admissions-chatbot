#!/usr/bin/env python3
"""
Dialogue History BLEU Score Evaluation Test Script
Tests LLaMA's ability to track dialogue history and maintain conversation context.
Target: BLEU score ≥ 0.40 for dialogue history tracking across 15 dialogue sessions.
"""

import os
import sys
import json
import requests
import time
from typing import List, Dict, Tuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DialogueHistoryBLEUEvaluator:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8000/chatbot"
        self.smoothing = SmoothingFunction().method1
        
        # Define 150 dialogue sessions focused on testing dialogue history tracking
        self.dialogue_sessions = [
            {
                "session_id": "history_test_1",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Computer Science curriculum?",
                        "context_keywords": ["computer", "science", "curriculum", "program"]
                    },
                    {
                        "query": "What about first year?",
                        "expected_context_reference": ["computer", "science", "first", "year"],
                        "context_test": "Should understand 'first year' refers to Computer Science first year"
                    },
                    {
                        "query": "How many subjects are there?",
                        "expected_context_reference": ["computer", "science", "first", "year", "subjects"],
                        "context_test": "Should understand 'subjects' refers to CS first year subjects"
                    }
                ]
            },
            {
                "session_id": "history_test_2",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the admission requirements for transfer students?",
                        "context_keywords": ["transfer", "students", "admission", "requirements"]
                    },
                    {
                        "query": "What documents do they need?",
                        "expected_context_reference": ["transfer", "students", "documents", "requirements"],
                        "context_test": "Should understand 'they' refers to transfer students"
                    },
                    {
                        "query": "When should they apply?",
                        "expected_context_reference": ["transfer", "students", "apply", "deadline"],
                        "context_test": "Should maintain transfer student context for application timing"
                    }
                ]
            },
            {
                "session_id": "history_test_3",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the tuition fees for Engineering programs?",
                        "context_keywords": ["engineering", "tuition", "fees", "programs"]
                    },
                    {
                        "query": "Are there payment plans for it?",
                        "expected_context_reference": ["engineering", "payment", "plans", "tuition"],
                        "context_test": "Should understand 'it' refers to Engineering tuition fees"
                    }
                ]
            },
            {
                "session_id": "history_test_4",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Nursing program curriculum",
                        "context_keywords": ["nursing", "program", "curriculum"]
                    },
                    {
                        "query": "What are the clinical requirements?",
                        "expected_context_reference": ["nursing", "clinical", "requirements"],
                        "context_test": "Should understand clinical requirements are for Nursing"
                    },
                    {
                        "query": "How long does the program take?",
                        "expected_context_reference": ["nursing", "program", "duration", "years"],
                        "context_test": "Should understand 'the program' refers to Nursing program"
                    }
                ]
            },
            {
                "session_id": "history_test_5",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I apply for scholarships?",
                        "context_keywords": ["scholarships", "apply", "financial", "aid"]
                    },
                    {
                        "query": "What are the requirements for them?",
                        "expected_context_reference": ["scholarships", "requirements", "eligibility"],
                        "context_test": "Should understand 'them' refers to scholarships"
                    }
                ]
            },
            {
                "session_id": "history_test_6",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What payment methods are available?",
                        "context_keywords": ["payment", "methods", "options"]
                    },
                    {
                        "query": "Which banks support these?",
                        "expected_context_reference": ["payment", "methods", "banks", "support"],
                        "context_test": "Should understand 'these' refers to payment methods"
                    }
                ]
            },
            {
                "session_id": "history_test_7",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programs are available in the School of Engineering?",
                        "context_keywords": ["engineering", "programs", "school", "available"]
                    },
                    {
                        "query": "Which one has the most math courses?",
                        "expected_context_reference": ["engineering", "programs", "math", "courses"],
                        "context_test": "Should understand 'which one' refers to engineering programs"
                    }
                ]
            },
            {
                "session_id": "history_test_8",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What is the application process for international students?",
                        "context_keywords": ["international", "students", "application", "process"]
                    },
                    {
                        "query": "What additional documents do they need?",
                        "expected_context_reference": ["international", "students", "documents", "additional"],
                        "context_test": "Should understand 'they' refers to international students"
                    }
                ]
            },
            {
                "session_id": "history_test_9",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Business Administration curriculum?",
                        "context_keywords": ["business", "administration", "curriculum", "program"]
                    },
                    {
                        "query": "What subjects are in second year?",
                        "expected_context_reference": ["business", "administration", "second", "year"],
                        "context_test": "Should understand 'second year' refers to Business Administration second year"
                    },
                    {
                        "query": "Are there internship requirements?",
                        "expected_context_reference": ["business", "administration", "internship", "requirements"],
                        "context_test": "Should understand internship context for Business Administration"
                    }
                ]
            },
            {
                "session_id": "history_test_10",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the fees for Nursing program?",
                        "context_keywords": ["nursing", "program", "fees", "tuition"]
                    },
                    {
                        "query": "How much per year?",
                        "expected_context_reference": ["nursing", "program", "fees", "year"],
                        "context_test": "Should understand 'per year' refers to Nursing program yearly fees"
                    }
                ]
            },
            {
                "session_id": "history_test_11",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for graduate programs?",
                        "context_keywords": ["graduate", "programs", "requirements", "admission"]
                    },
                    {
                        "query": "What documents are needed for them?",
                        "expected_context_reference": ["graduate", "programs", "documents", "requirements"],
                        "context_test": "Should understand 'them' refers to graduate programs"
                    }
                ]
            },
            {
                "session_id": "history_test_12",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Psychology program",
                        "context_keywords": ["psychology", "program", "curriculum"]
                    },
                    {
                        "query": "What are the major subjects?",
                        "expected_context_reference": ["psychology", "program", "major", "subjects"],
                        "context_test": "Should understand 'major subjects' refers to Psychology program subjects"
                    },
                    {
                        "query": "How many years does it take?",
                        "expected_context_reference": ["psychology", "program", "years", "duration"],
                        "context_test": "Should understand duration question refers to Psychology program"
                    }
                ]
            },
            {
                "session_id": "history_test_13",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the miscellaneous fees?",
                        "context_keywords": ["miscellaneous", "fees", "additional", "costs"]
                    },
                    {
                        "query": "Are they required for all students?",
                        "expected_context_reference": ["miscellaneous", "fees", "required", "students"],
                        "context_test": "Should understand 'they' refers to miscellaneous fees"
                    }
                ]
            },
            {
                "session_id": "history_test_14",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I enroll as a returning student?",
                        "context_keywords": ["enroll", "returning", "student", "process"]
                    },
                    {
                        "query": "What's the deadline for it?",
                        "expected_context_reference": ["enroll", "returning", "student", "deadline"],
                        "context_test": "Should understand 'it' refers to returning student enrollment"
                    }
                ]
            },
            {
                "session_id": "history_test_15",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programs are offered in the School of Medicine?",
                        "context_keywords": ["programs", "school", "medicine", "offered"]
                    },
                    {
                        "query": "What are the admission requirements for these?",
                        "expected_context_reference": ["programs", "school", "medicine", "admission", "requirements"],
                        "context_test": "Should understand 'these' refers to School of Medicine programs"
                    },
                    {
                        "query": "How competitive are they?",
                        "expected_context_reference": ["programs", "school", "medicine", "competitive"],
                        "context_test": "Should understand 'they' refers to School of Medicine programs"
                    }
                ]
            },
            {
                "session_id": "history_test_16",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programming languages are taught in Information Technology?",
                        "context_keywords": ["programming", "languages", "information", "technology"]
                    },
                    {
                        "query": "Which one is taught first?",
                        "expected_context_reference": ["programming", "languages", "information", "technology", "first"],
                        "context_test": "Should understand 'which one' refers to programming languages in IT"
                    }
                ]
            },
            {
                "session_id": "history_test_17",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for graduate school applications?",
                        "context_keywords": ["graduate", "school", "applications", "requirements"]
                    },
                    {
                        "query": "Do I need recommendation letters for them?",
                        "expected_context_reference": ["graduate", "school", "applications", "recommendation", "letters"],
                        "context_test": "Should understand 'them' refers to graduate school applications"
                    }
                ]
            },
            {
                "session_id": "history_test_18",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the laboratory fees for Chemistry programs?",
                        "context_keywords": ["laboratory", "fees", "chemistry", "programs"]
                    },
                    {
                        "query": "Are they paid per semester?",
                        "expected_context_reference": ["laboratory", "fees", "chemistry", "semester"],
                        "context_test": "Should understand 'they' refers to laboratory fees"
                    }
                ]
            },
            {
                "session_id": "history_test_19",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Mathematics curriculum",
                        "context_keywords": ["mathematics", "curriculum", "program"]
                    },
                    {
                        "query": "What are the core subjects?",
                        "expected_context_reference": ["mathematics", "curriculum", "core", "subjects"],
                        "context_test": "Should understand 'core subjects' refers to Mathematics core subjects"
                    },
                    {
                        "query": "How difficult are they?",
                        "expected_context_reference": ["mathematics", "core", "subjects", "difficult"],
                        "context_test": "Should understand 'they' refers to Mathematics core subjects"
                    }
                ]
            },
            {
                "session_id": "history_test_20",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I apply for need-based financial aid?",
                        "context_keywords": ["apply", "need-based", "financial", "aid"]
                    },
                    {
                        "query": "What documents are required for it?",
                        "expected_context_reference": ["need-based", "financial", "aid", "documents"],
                        "context_test": "Should understand 'it' refers to need-based financial aid application"
                    }
                ]
            },
            {
                "session_id": "history_test_21",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the miscellaneous fees for students?",
                        "context_keywords": ["miscellaneous", "fees", "students"]
                    },
                    {
                        "query": "Can I pay them online?",
                        "expected_context_reference": ["miscellaneous", "fees", "pay", "online"],
                        "context_test": "Should understand 'them' refers to miscellaneous fees"
                    }
                ]
            },
            {
                "session_id": "history_test_22",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Physics program curriculum?",
                        "context_keywords": ["physics", "program", "curriculum"]
                    },
                    {
                        "query": "Are there laboratory requirements?",
                        "expected_context_reference": ["physics", "program", "laboratory", "requirements"],
                        "context_test": "Should understand laboratory requirements are for Physics program"
                    }
                ]
            },
            {
                "session_id": "history_test_23",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What is the minimum GPA for admission?",
                        "context_keywords": ["minimum", "GPA", "admission"]
                    },
                    {
                        "query": "Does it vary by program?",
                        "expected_context_reference": ["minimum", "GPA", "admission", "program"],
                        "context_test": "Should understand 'it' refers to minimum GPA requirement"
                    }
                ]
            },
            {
                "session_id": "history_test_24",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What payment methods are accepted for tuition?",
                        "context_keywords": ["payment", "methods", "tuition"]
                    },
                    {
                        "query": "Which banks support these?",
                        "expected_context_reference": ["payment", "methods", "tuition", "banks"],
                        "context_test": "Should understand 'these' refers to payment methods"
                    }
                ]
            },
            {
                "session_id": "history_test_25",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "How many years is the Architecture program?",
                        "context_keywords": ["years", "architecture", "program"]
                    },
                    {
                        "query": "What subjects are covered in the first year?",
                        "expected_context_reference": ["architecture", "program", "first", "year", "subjects"],
                        "context_test": "Should understand first year refers to Architecture program first year"
                    }
                ]
            },
            {
                "session_id": "history_test_26",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I submit my college transcripts?",
                        "context_keywords": ["submit", "college", "transcripts"]
                    },
                    {
                        "query": "Do they need to be certified?",
                        "expected_context_reference": ["college", "transcripts", "certified"],
                        "context_test": "Should understand 'they' refers to college transcripts"
                    }
                ]
            },
            {
                "session_id": "history_test_27",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there installment plans for tuition fees?",
                        "context_keywords": ["installment", "plans", "tuition", "fees"]
                    },
                    {
                        "query": "How do I apply for them?",
                        "expected_context_reference": ["installment", "plans", "tuition", "apply"],
                        "context_test": "Should understand 'them' refers to installment plans"
                    }
                ]
            },
            {
                "session_id": "history_test_28",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Biology program like?",
                        "context_keywords": ["biology", "program"]
                    },
                    {
                        "query": "Are there research opportunities?",
                        "expected_context_reference": ["biology", "program", "research", "opportunities"],
                        "context_test": "Should understand research opportunities are for Biology program"
                    },
                    {
                        "query": "When can students start participating?",
                        "expected_context_reference": ["biology", "research", "opportunities", "students"],
                        "context_test": "Should understand participation refers to Biology research opportunities"
                    }
                ]
            },
            {
                "session_id": "history_test_29",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the English proficiency requirements?",
                        "context_keywords": ["english", "proficiency", "requirements"]
                    },
                    {
                        "query": "Which tests are accepted for them?",
                        "expected_context_reference": ["english", "proficiency", "requirements", "tests"],
                        "context_test": "Should understand 'them' refers to English proficiency requirements"
                    }
                ]
            },
            {
                "session_id": "history_test_30",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What is the total cost for a 4-year program?",
                        "context_keywords": ["total", "cost", "4-year", "program"]
                    },
                    {
                        "query": "Does it include living expenses?",
                        "expected_context_reference": ["total", "cost", "4-year", "living", "expenses"],
                        "context_test": "Should understand 'it' refers to total cost"
                    }
                ]
            },
            {
                "session_id": "history_test_31",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Chemistry program curriculum",
                        "context_keywords": ["chemistry", "program", "curriculum"]
                    },
                    {
                        "query": "What laboratory work is involved?",
                        "expected_context_reference": ["chemistry", "program", "laboratory", "work"],
                        "context_test": "Should understand laboratory work is for Chemistry program"
                    }
                ]
            },
            {
                "session_id": "history_test_32",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I apply as a mature student?",
                        "context_keywords": ["apply", "mature", "student"]
                    },
                    {
                        "query": "Are there special requirements for them?",
                        "expected_context_reference": ["mature", "student", "special", "requirements"],
                        "context_test": "Should understand 'them' refers to mature students"
                    }
                ]
            },
            {
                "session_id": "history_test_33",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the fees for summer courses?",
                        "context_keywords": ["fees", "summer", "courses"]
                    },
                    {
                        "query": "Are they different from regular semester fees?",
                        "expected_context_reference": ["summer", "courses", "fees", "regular", "semester"],
                        "context_test": "Should understand 'they' refers to summer course fees"
                    }
                ]
            },
            {
                "session_id": "history_test_34",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programs are offered in the School of Arts?",
                        "context_keywords": ["programs", "school", "arts"]
                    },
                    {
                        "query": "Which one has the most creative writing courses?",
                        "expected_context_reference": ["programs", "school", "arts", "creative", "writing"],
                        "context_test": "Should understand 'which one' refers to Arts programs"
                    }
                ]
            },
            {
                "session_id": "history_test_35",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What is the application process for part-time students?",
                        "context_keywords": ["application", "process", "part-time", "students"]
                    },
                    {
                        "query": "Is it different from full-time applications?",
                        "expected_context_reference": ["part-time", "students", "application", "full-time"],
                        "context_test": "Should understand 'it' refers to part-time application process"
                    }
                ]
            },
            {
                "session_id": "history_test_36",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there additional fees for international students?",
                        "context_keywords": ["additional", "fees", "international", "students"]
                    },
                    {
                        "query": "How much are they per semester?",
                        "expected_context_reference": ["additional", "fees", "international", "semester"],
                        "context_test": "Should understand 'they' refers to additional fees for international students"
                    }
                ]
            },
            {
                "session_id": "history_test_37",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Social Work program curriculum?",
                        "context_keywords": ["social", "work", "program", "curriculum"]
                    },
                    {
                        "query": "Are there fieldwork requirements?",
                        "expected_context_reference": ["social", "work", "program", "fieldwork", "requirements"],
                        "context_test": "Should understand fieldwork requirements are for Social Work program"
                    }
                ]
            },
            {
                "session_id": "history_test_38",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I defer my admission to next year?",
                        "context_keywords": ["defer", "admission", "next", "year"]
                    },
                    {
                        "query": "What is the deadline for it?",
                        "expected_context_reference": ["defer", "admission", "deadline"],
                        "context_test": "Should understand 'it' refers to admission deferral"
                    }
                ]
            },
            {
                "session_id": "history_test_39",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What is the refund policy for tuition fees?",
                        "context_keywords": ["refund", "policy", "tuition", "fees"]
                    },
                    {
                        "query": "How long does it take to process?",
                        "expected_context_reference": ["refund", "policy", "tuition", "process"],
                        "context_test": "Should understand processing refers to tuition refund"
                    }
                ]
            },
            {
                "session_id": "history_test_40",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Environmental Science program",
                        "context_keywords": ["environmental", "science", "program"]
                    },
                    {
                        "query": "What career opportunities does it offer?",
                        "expected_context_reference": ["environmental", "science", "career", "opportunities"],
                        "context_test": "Should understand 'it' refers to Environmental Science program"
                    }
                ]
            },
            {
                "session_id": "history_test_41",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What documents do homeschooled students need?",
                        "context_keywords": ["documents", "homeschooled", "students"]
                    },
                    {
                        "query": "Do they need additional verification?",
                        "expected_context_reference": ["homeschooled", "students", "additional", "verification"],
                        "context_test": "Should understand 'they' refers to homeschooled students"
                    }
                ]
            },
            {
                "session_id": "history_test_42",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there discounts for early payment?",
                        "context_keywords": ["discounts", "early", "payment"]
                    },
                    {
                        "query": "How much can I save with them?",
                        "expected_context_reference": ["discounts", "early", "payment", "save"],
                        "context_test": "Should understand 'them' refers to early payment discounts"
                    }
                ]
            },
            {
                "session_id": "history_test_43",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Mechanical Engineering curriculum?",
                        "context_keywords": ["mechanical", "engineering", "curriculum"]
                    },
                    {
                        "query": "How many math courses are required?",
                        "expected_context_reference": ["mechanical", "engineering", "math", "courses"],
                        "context_test": "Should understand math courses are for Mechanical Engineering"
                    },
                    {
                        "query": "Are they taken in the first two years?",
                        "expected_context_reference": ["mechanical", "engineering", "math", "courses", "first", "two", "years"],
                        "context_test": "Should understand 'they' refers to Mechanical Engineering math courses"
                    }
                ]
            },
            {
                "session_id": "history_test_44",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I check my application status?",
                        "context_keywords": ["check", "application", "status"]
                    },
                    {
                        "query": "How often is it updated?",
                        "expected_context_reference": ["application", "status", "updated"],
                        "context_test": "Should understand 'it' refers to application status"
                    }
                ]
            },
            {
                "session_id": "history_test_45",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the parking fees on campus?",
                        "context_keywords": ["parking", "fees", "campus"]
                    },
                    {
                        "query": "Are they included in tuition?",
                        "expected_context_reference": ["parking", "fees", "tuition"],
                        "context_test": "Should understand 'they' refers to parking fees"
                    }
                ]
            },
            {
                "session_id": "history_test_46",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programs are available in the School of Law?",
                        "context_keywords": ["programs", "school", "law"]
                    },
                    {
                        "query": "What are the admission requirements for these?",
                        "expected_context_reference": ["programs", "school", "law", "admission", "requirements"],
                        "context_test": "Should understand 'these' refers to Law school programs"
                    }
                ]
            },
            {
                "session_id": "history_test_47",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for student visa renewal?",
                        "context_keywords": ["requirements", "student", "visa", "renewal"]
                    },
                    {
                        "query": "How long before expiry should I apply for it?",
                        "expected_context_reference": ["student", "visa", "renewal", "expiry", "apply"],
                        "context_test": "Should understand 'it' refers to visa renewal"
                    }
                ]
            },
            {
                "session_id": "history_test_48",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the dormitory fees?",
                        "context_keywords": ["dormitory", "fees"]
                    },
                    {
                        "query": "Are they paid monthly or per semester?",
                        "expected_context_reference": ["dormitory", "fees", "monthly", "semester"],
                        "context_test": "Should understand 'they' refers to dormitory fees"
                    }
                ]
            },
            {
                "session_id": "history_test_49",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Electrical Engineering program",
                        "context_keywords": ["electrical", "engineering", "program"]
                    },
                    {
                        "query": "What specializations are available?",
                        "expected_context_reference": ["electrical", "engineering", "specializations"],
                        "context_test": "Should understand specializations are for Electrical Engineering"
                    }
                ]
            },
            {
                "session_id": "history_test_50",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I apply for academic probation appeal?",
                        "context_keywords": ["apply", "academic", "probation", "appeal"]
                    },
                    {
                        "query": "What documents are needed for it?",
                        "expected_context_reference": ["academic", "probation", "appeal", "documents"],
                        "context_test": "Should understand 'it' refers to academic probation appeal"
                    }
                ]
            },
            {
                "session_id": "history_test_51",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the technology fees for students?",
                        "context_keywords": ["technology", "fees", "students"]
                    },
                    {
                        "query": "What services do they cover?",
                        "expected_context_reference": ["technology", "fees", "services"],
                        "context_test": "Should understand 'they' refers to technology fees"
                    }
                ]
            },
            {
                "session_id": "history_test_52",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Journalism program curriculum?",
                        "context_keywords": ["journalism", "program", "curriculum"]
                    },
                    {
                        "query": "Are there internship opportunities?",
                        "expected_context_reference": ["journalism", "program", "internship", "opportunities"],
                        "context_test": "Should understand internship opportunities are for Journalism program"
                    }
                ]
            },
            {
                "session_id": "history_test_53",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for readmission?",
                        "context_keywords": ["requirements", "readmission"]
                    },
                    {
                        "query": "How long is the process for it?",
                        "expected_context_reference": ["readmission", "process"],
                        "context_test": "Should understand 'it' refers to readmission process"
                    }
                ]
            },
            {
                "session_id": "history_test_54",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there student activity fees?",
                        "context_keywords": ["student", "activity", "fees"]
                    },
                    {
                        "query": "What activities do they support?",
                        "expected_context_reference": ["student", "activity", "fees", "activities"],
                        "context_test": "Should understand 'they' refers to student activity fees"
                    }
                ]
            },
            {
                "session_id": "history_test_55",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programs are offered in the School of Education?",
                        "context_keywords": ["programs", "school", "education"]
                    },
                    {
                        "query": "Which one leads to teaching certification?",
                        "expected_context_reference": ["programs", "school", "education", "teaching", "certification"],
                        "context_test": "Should understand 'which one' refers to Education programs"
                    }
                ]
            },
            {
                "session_id": "history_test_56",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I submit letters of recommendation?",
                        "context_keywords": ["submit", "letters", "recommendation"]
                    },
                    {
                        "query": "Can they be submitted electronically?",
                        "expected_context_reference": ["letters", "recommendation", "electronically"],
                        "context_test": "Should understand 'they' refers to letters of recommendation"
                    }
                ]
            },
            {
                "session_id": "history_test_57",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the graduation fees?",
                        "context_keywords": ["graduation", "fees"]
                    },
                    {
                        "query": "When do I need to pay them?",
                        "expected_context_reference": ["graduation", "fees", "pay"],
                        "context_test": "Should understand 'them' refers to graduation fees"
                    }
                ]
            },
            {
                "session_id": "history_test_58",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Pharmacy program",
                        "context_keywords": ["pharmacy", "program"]
                    },
                    {
                        "query": "How long does it take to complete?",
                        "expected_context_reference": ["pharmacy", "program", "complete"],
                        "context_test": "Should understand completion refers to Pharmacy program"
                    },
                    {
                        "query": "Are there clinical rotations required?",
                        "expected_context_reference": ["pharmacy", "program", "clinical", "rotations"],
                        "context_test": "Should understand clinical rotations are for Pharmacy program"
                    }
                ]
            },
            {
                "session_id": "history_test_59",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for course credit transfer?",
                        "context_keywords": ["requirements", "course", "credit", "transfer"]
                    },
                    {
                        "query": "How many credits can be transferred?",
                        "expected_context_reference": ["course", "credit", "transfer", "credits"],
                        "context_test": "Should understand credits refers to transferable course credits"
                    }
                ]
            },
            {
                "session_id": "history_test_60",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there late payment penalties?",
                        "context_keywords": ["late", "payment", "penalties"]
                    },
                    {
                        "query": "How much are they per month?",
                        "expected_context_reference": ["late", "payment", "penalties", "month"],
                        "context_test": "Should understand 'they' refers to late payment penalties"
                    }
                ]
            },
            {
                "session_id": "history_test_61",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Civil Engineering curriculum?",
                        "context_keywords": ["civil", "engineering", "curriculum"]
                    },
                    {
                        "query": "Are there design projects included?",
                        "expected_context_reference": ["civil", "engineering", "design", "projects"],
                        "context_test": "Should understand design projects are for Civil Engineering"
                    }
                ]
            },
            {
                "session_id": "history_test_62",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I apply for academic leave of absence?",
                        "context_keywords": ["apply", "academic", "leave", "absence"]
                    },
                    {
                        "query": "What is the maximum duration for it?",
                        "expected_context_reference": ["academic", "leave", "absence", "duration"],
                        "context_test": "Should understand 'it' refers to academic leave of absence"
                    }
                ]
            },
            {
                "session_id": "history_test_63",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the health insurance fees?",
                        "context_keywords": ["health", "insurance", "fees"]
                    },
                    {
                        "query": "Are they mandatory for all students?",
                        "expected_context_reference": ["health", "insurance", "fees", "mandatory"],
                        "context_test": "Should understand 'they' refers to health insurance fees"
                    }
                ]
            },
            {
                "session_id": "history_test_64",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programs are available in the School of Music?",
                        "context_keywords": ["programs", "school", "music"]
                    },
                    {
                        "query": "Do they require auditions?",
                        "expected_context_reference": ["programs", "school", "music", "auditions"],
                        "context_test": "Should understand 'they' refers to Music programs"
                    }
                ]
            },
            {
                "session_id": "history_test_65",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for dual degree programs?",
                        "context_keywords": ["requirements", "dual", "degree", "programs"]
                    },
                    {
                        "query": "How long do they typically take?",
                        "expected_context_reference": ["dual", "degree", "programs", "take"],
                        "context_test": "Should understand 'they' refers to dual degree programs"
                    }
                ]
            },
            {
                "session_id": "history_test_66",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there application fees for graduate programs?",
                        "context_keywords": ["application", "fees", "graduate", "programs"]
                    },
                    {
                        "query": "Can they be waived for financial hardship?",
                        "expected_context_reference": ["application", "fees", "waived", "financial", "hardship"],
                        "context_test": "Should understand 'they' refers to application fees"
                    }
                ]
            },
            {
                "session_id": "history_test_67",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Graphic Design program",
                        "context_keywords": ["graphic", "design", "program"]
                    },
                    {
                        "query": "What software skills are taught?",
                        "expected_context_reference": ["graphic", "design", "software", "skills"],
                        "context_test": "Should understand software skills are for Graphic Design program"
                    }
                ]
            },
            {
                "session_id": "history_test_68",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I appeal an admission decision?",
                        "context_keywords": ["appeal", "admission", "decision"]
                    },
                    {
                        "query": "What is the deadline for it?",
                        "expected_context_reference": ["appeal", "admission", "decision", "deadline"],
                        "context_test": "Should understand 'it' refers to admission appeal"
                    }
                ]
            },
            {
                "session_id": "history_test_69",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the transcript fees?",
                        "context_keywords": ["transcript", "fees"]
                    },
                    {
                        "query": "How long does it take to process them?",
                        "expected_context_reference": ["transcript", "fees", "process"],
                        "context_test": "Should understand 'them' refers to transcript requests"
                    }
                ]
            },
            {
                "session_id": "history_test_70",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Anthropology program curriculum?",
                        "context_keywords": ["anthropology", "program", "curriculum"]
                    },
                    {
                        "query": "Are there fieldwork requirements?",
                        "expected_context_reference": ["anthropology", "program", "fieldwork", "requirements"],
                        "context_test": "Should understand fieldwork requirements are for Anthropology program"
                    }
                ]
            },
            {
                "session_id": "history_test_71",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What documents do military veterans need for admission?",
                        "context_keywords": ["documents", "military", "veterans", "admission"]
                    },
                    {
                        "query": "Are there special benefits for them?",
                        "expected_context_reference": ["military", "veterans", "special", "benefits"],
                        "context_test": "Should understand 'them' refers to military veterans"
                    }
                ]
            },
            {
                "session_id": "history_test_72",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there fees for course withdrawals?",
                        "context_keywords": ["fees", "course", "withdrawals"]
                    },
                    {
                        "query": "Do they vary by withdrawal date?",
                        "expected_context_reference": ["fees", "course", "withdrawals", "withdrawal", "date"],
                        "context_test": "Should understand 'they' refers to withdrawal fees"
                    }
                ]
            },
            {
                "session_id": "history_test_73",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programs are offered in the School of Dentistry?",
                        "context_keywords": ["programs", "school", "dentistry"]
                    },
                    {
                        "query": "What are the clinical requirements for these?",
                        "expected_context_reference": ["programs", "school", "dentistry", "clinical", "requirements"],
                        "context_test": "Should understand 'these' refers to Dentistry programs"
                    }
                ]
            },
            {
                "session_id": "history_test_74",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I change my major after admission?",
                        "context_keywords": ["change", "major", "admission"]
                    },
                    {
                        "query": "Are there restrictions on it?",
                        "expected_context_reference": ["change", "major", "restrictions"],
                        "context_test": "Should understand 'it' refers to changing major"
                    }
                ]
            },
            {
                "session_id": "history_test_75",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the library fees?",
                        "context_keywords": ["library", "fees"]
                    },
                    {
                        "query": "What services do they include?",
                        "expected_context_reference": ["library", "fees", "services"],
                        "context_test": "Should understand 'they' refers to library fees"
                    }
                ]
            },
            {
                "session_id": "history_test_76",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Film Studies program",
                        "context_keywords": ["film", "studies", "program"]
                    },
                    {
                        "query": "What equipment is available for students?",
                        "expected_context_reference": ["film", "studies", "equipment", "students"],
                        "context_test": "Should understand equipment is for Film Studies students"
                    }
                ]
            },
            {
                "session_id": "history_test_77",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for conditional admission?",
                        "context_keywords": ["requirements", "conditional", "admission"]
                    },
                    {
                        "query": "How do I meet the conditions for it?",
                        "expected_context_reference": ["conditional", "admission", "conditions"],
                        "context_test": "Should understand 'it' refers to conditional admission"
                    }
                ]
            },
            {
                "session_id": "history_test_78",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there orientation fees for new students?",
                        "context_keywords": ["orientation", "fees", "new", "students"]
                    },
                    {
                        "query": "What does the program include?",
                        "expected_context_reference": ["orientation", "fees", "program"],
                        "context_test": "Should understand program refers to orientation program"
                    }
                ]
            },
            {
                "session_id": "history_test_79",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Political Science curriculum?",
                        "context_keywords": ["political", "science", "curriculum"]
                    },
                    {
                        "query": "Are there internship opportunities available?",
                        "expected_context_reference": ["political", "science", "internship", "opportunities"],
                        "context_test": "Should understand internship opportunities are for Political Science"
                    }
                ]
            },
            {
                "session_id": "history_test_80",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I apply for priority registration?",
                        "context_keywords": ["apply", "priority", "registration"]
                    },
                    {
                        "query": "What are the eligibility criteria for it?",
                        "expected_context_reference": ["priority", "registration", "eligibility", "criteria"],
                        "context_test": "Should understand 'it' refers to priority registration"
                    }
                ]
            },
            {
                "session_id": "history_test_81",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the recreation center fees?",
                        "context_keywords": ["recreation", "center", "fees"]
                    },
                    {
                        "query": "What facilities do they provide access to?",
                        "expected_context_reference": ["recreation", "center", "fees", "facilities"],
                        "context_test": "Should understand 'they' refers to recreation center fees"
                    }
                ]
            },
            {
                "session_id": "history_test_82",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programs are available in the School of Public Health?",
                        "context_keywords": ["programs", "school", "public", "health"]
                    },
                    {
                        "query": "Which one focuses on epidemiology?",
                        "expected_context_reference": ["programs", "school", "public", "health", "epidemiology"],
                        "context_test": "Should understand 'which one' refers to Public Health programs"
                    }
                ]
            },
            {
                "session_id": "history_test_83",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for early admission?",
                        "context_keywords": ["requirements", "early", "admission"]
                    },
                    {
                        "query": "What are the benefits of it?",
                        "expected_context_reference": ["early", "admission", "benefits"],
                        "context_test": "Should understand 'it' refers to early admission"
                    }
                ]
            },
            {
                "session_id": "history_test_84",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there fees for course add/drop?",
                        "context_keywords": ["fees", "course", "add", "drop"]
                    },
                    {
                        "query": "When are they charged?",
                        "expected_context_reference": ["fees", "course", "add", "drop", "charged"],
                        "context_test": "Should understand 'they' refers to add/drop fees"
                    }
                ]
            },
            {
                "session_id": "history_test_85",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Theatre Arts program",
                        "context_keywords": ["theatre", "arts", "program"]
                    },
                    {
                        "query": "Are there performance requirements?",
                        "expected_context_reference": ["theatre", "arts", "performance", "requirements"],
                        "context_test": "Should understand performance requirements are for Theatre Arts"
                    }
                ]
            },
            {
                "session_id": "history_test_86",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I submit portfolio for art programs?",
                        "context_keywords": ["submit", "portfolio", "art", "programs"]
                    },
                    {
                        "query": "What format should it be in?",
                        "expected_context_reference": ["portfolio", "art", "programs", "format"],
                        "context_test": "Should understand 'it' refers to art portfolio"
                    }
                ]
            },
            {
                "session_id": "history_test_87",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the student union fees?",
                        "context_keywords": ["student", "union", "fees"]
                    },
                    {
                        "query": "What services do they fund?",
                        "expected_context_reference": ["student", "union", "fees", "services"],
                        "context_test": "Should understand 'they' refers to student union fees"
                    }
                ]
            },
            {
                "session_id": "history_test_88",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Economics program curriculum?",
                        "context_keywords": ["economics", "program", "curriculum"]
                    },
                    {
                        "query": "Are there statistics courses required?",
                        "expected_context_reference": ["economics", "program", "statistics", "courses"],
                        "context_test": "Should understand statistics courses are for Economics program"
                    }
                ]
            },
            {
                "session_id": "history_test_89",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for honors program admission?",
                        "context_keywords": ["requirements", "honors", "program", "admission"]
                    },
                    {
                        "query": "What GPA is needed for it?",
                        "expected_context_reference": ["honors", "program", "admission", "GPA"],
                        "context_test": "Should understand 'it' refers to honors program admission"
                    }
                ]
            },
            {
                "session_id": "history_test_90",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there fees for ID card replacement?",
                        "context_keywords": ["fees", "ID", "card", "replacement"]
                    },
                    {
                        "query": "How much do they cost?",
                        "expected_context_reference": ["fees", "ID", "card", "replacement", "cost"],
                        "context_test": "Should understand 'they' refers to ID card replacement fees"
                    }
                ]
            },
            {
                "session_id": "history_test_91",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programs are offered in the School of Veterinary Medicine?",
                        "context_keywords": ["programs", "school", "veterinary", "medicine"]
                    },
                    {
                        "query": "What are the clinical rotations like?",
                        "expected_context_reference": ["programs", "school", "veterinary", "medicine", "clinical", "rotations"],
                        "context_test": "Should understand clinical rotations are for Veterinary Medicine programs"
                    }
                ]
            },
            {
                "session_id": "history_test_92",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I apply for course overload permission?",
                        "context_keywords": ["apply", "course", "overload", "permission"]
                    },
                    {
                        "query": "What is the maximum credits allowed for it?",
                        "expected_context_reference": ["course", "overload", "permission", "credits"],
                        "context_test": "Should understand 'it' refers to course overload"
                    }
                ]
            },
            {
                "session_id": "history_test_93",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the commencement fees?",
                        "context_keywords": ["commencement", "fees"]
                    },
                    {
                        "query": "Are they required for all graduates?",
                        "expected_context_reference": ["commencement", "fees", "graduates"],
                        "context_test": "Should understand 'they' refers to commencement fees"
                    }
                ]
            },
            {
                "session_id": "history_test_94",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Linguistics program",
                        "context_keywords": ["linguistics", "program"]
                    },
                    {
                        "query": "What research opportunities are available?",
                        "expected_context_reference": ["linguistics", "program", "research", "opportunities"],
                        "context_test": "Should understand research opportunities are for Linguistics program"
                    }
                ]
            },
            {
                "session_id": "history_test_95",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for study abroad programs?",
                        "context_keywords": ["requirements", "study", "abroad", "programs"]
                    },
                    {
                        "query": "How do I apply for them?",
                        "expected_context_reference": ["study", "abroad", "programs", "apply"],
                        "context_test": "Should understand 'them' refers to study abroad programs"
                    }
                ]
            },
            {
                "session_id": "history_test_96",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there fees for online course delivery?",
                        "context_keywords": ["fees", "online", "course", "delivery"]
                    },
                    {
                        "query": "Are they different from in-person courses?",
                        "expected_context_reference": ["fees", "online", "course", "in-person"],
                        "context_test": "Should understand 'they' refers to online course fees"
                    }
                ]
            },
            {
                "session_id": "history_test_97",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Sociology program curriculum?",
                        "context_keywords": ["sociology", "program", "curriculum"]
                    },
                    {
                        "query": "Are there research methods courses included?",
                        "expected_context_reference": ["sociology", "program", "research", "methods", "courses"],
                        "context_test": "Should understand research methods courses are for Sociology program"
                    }
                ]
            },
            {
                "session_id": "history_test_98",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I apply for academic accommodation?",
                        "context_keywords": ["apply", "academic", "accommodation"]
                    },
                    {
                        "query": "What documentation is needed for it?",
                        "expected_context_reference": ["academic", "accommodation", "documentation"],
                        "context_test": "Should understand 'it' refers to academic accommodation"
                    }
                ]
            },
            {
                "session_id": "history_test_99",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the thesis binding fees?",
                        "context_keywords": ["thesis", "binding", "fees"]
                    },
                    {
                        "query": "When do I need to pay them?",
                        "expected_context_reference": ["thesis", "binding", "fees", "pay"],
                        "context_test": "Should understand 'them' refers to thesis binding fees"
                    }
                ]
            },
            {
                "session_id": "history_test_100",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programs are available in the School of Agriculture?",
                        "context_keywords": ["programs", "school", "agriculture"]
                    },
                    {
                        "query": "Do they include hands-on farming experience?",
                        "expected_context_reference": ["programs", "school", "agriculture", "farming", "experience"],
                        "context_test": "Should understand 'they' refers to Agriculture programs"
                    }
                ]
            },
            {
                "session_id": "history_test_101",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for graduate assistantship?",
                        "context_keywords": ["requirements", "graduate", "assistantship"]
                    },
                    {
                        "query": "What are the benefits of it?",
                        "expected_context_reference": ["graduate", "assistantship", "benefits"],
                        "context_test": "Should understand 'it' refers to graduate assistantship"
                    }
                ]
            },
            {
                "session_id": "history_test_102",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there fees for course repeat?",
                        "context_keywords": ["fees", "course", "repeat"]
                    },
                    {
                        "query": "How many times can I repeat without additional charges?",
                        "expected_context_reference": ["fees", "course", "repeat", "additional", "charges"],
                        "context_test": "Should understand repeat refers to course repetition"
                    }
                ]
            },
            {
                "session_id": "history_test_103",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Philosophy program",
                        "context_keywords": ["philosophy", "program"]
                    },
                    {
                        "query": "What areas of specialization are offered?",
                        "expected_context_reference": ["philosophy", "program", "specialization"],
                        "context_test": "Should understand specialization areas are for Philosophy program"
                    }
                ]
            },
            {
                "session_id": "history_test_104",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I apply for independent study courses?",
                        "context_keywords": ["apply", "independent", "study", "courses"]
                    },
                    {
                        "query": "Do I need faculty sponsorship for them?",
                        "expected_context_reference": ["independent", "study", "courses", "faculty", "sponsorship"],
                        "context_test": "Should understand 'them' refers to independent study courses"
                    }
                ]
            },
            {
                "session_id": "history_test_105",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the laboratory breakage fees?",
                        "context_keywords": ["laboratory", "breakage", "fees"]
                    },
                    {
                        "query": "How are they calculated?",
                        "expected_context_reference": ["laboratory", "breakage", "fees", "calculated"],
                        "context_test": "Should understand 'they' refers to laboratory breakage fees"
                    }
                ]
            },
            {
                "session_id": "history_test_106",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the History program curriculum?",
                        "context_keywords": ["history", "program", "curriculum"]
                    },
                    {
                        "query": "Are there thesis requirements for graduation?",
                        "expected_context_reference": ["history", "program", "thesis", "requirements", "graduation"],
                        "context_test": "Should understand thesis requirements are for History program graduation"
                    }
                ]
            },
            {
                "session_id": "history_test_107",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for credit by examination?",
                        "context_keywords": ["requirements", "credit", "examination"]
                    },
                    {
                        "query": "What subjects are eligible for it?",
                        "expected_context_reference": ["credit", "examination", "subjects", "eligible"],
                        "context_test": "Should understand 'it' refers to credit by examination"
                    }
                ]
            },
            {
                "session_id": "history_test_108",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there fees for audit courses?",
                        "context_keywords": ["fees", "audit", "courses"]
                    },
                    {
                        "query": "Are they the same as regular enrollment?",
                        "expected_context_reference": ["fees", "audit", "courses", "regular", "enrollment"],
                        "context_test": "Should understand 'they' refers to audit course fees"
                    }
                ]
            },
            {
                "session_id": "history_test_109",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programs are offered in the School of Social Work?",
                        "context_keywords": ["programs", "school", "social", "work"]
                    },
                    {
                        "query": "What field placement opportunities do they provide?",
                        "expected_context_reference": ["programs", "school", "social", "work", "field", "placement"],
                        "context_test": "Should understand 'they' refers to Social Work programs"
                    }
                ]
            },
            {
                "session_id": "history_test_110",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I apply for thesis extension?",
                        "context_keywords": ["apply", "thesis", "extension"]
                    },
                    {
                        "query": "What is the maximum duration for it?",
                        "expected_context_reference": ["thesis", "extension", "duration"],
                        "context_test": "Should understand 'it' refers to thesis extension"
                    }
                ]
            },
            {
                "session_id": "history_test_111",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the field trip fees?",
                        "context_keywords": ["field", "trip", "fees"]
                    },
                    {
                        "query": "Are they covered by course fees?",
                        "expected_context_reference": ["field", "trip", "fees", "course", "fees"],
                        "context_test": "Should understand 'they' refers to field trip fees"
                    }
                ]
            },
            {
                "session_id": "history_test_112",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Geography program",
                        "context_keywords": ["geography", "program"]
                    },
                    {
                        "query": "Are there GIS courses included?",
                        "expected_context_reference": ["geography", "program", "GIS", "courses"],
                        "context_test": "Should understand GIS courses are for Geography program"
                    }
                ]
            },
            {
                "session_id": "history_test_113",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for course substitution?",
                        "context_keywords": ["requirements", "course", "substitution"]
                    },
                    {
                        "query": "Who approves them?",
                        "expected_context_reference": ["course", "substitution", "approves"],
                        "context_test": "Should understand 'them' refers to course substitutions"
                    }
                ]
            },
            {
                "session_id": "history_test_114",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there fees for makeup examinations?",
                        "context_keywords": ["fees", "makeup", "examinations"]
                    },
                    {
                        "query": "When are they charged?",
                        "expected_context_reference": ["fees", "makeup", "examinations", "charged"],
                        "context_test": "Should understand 'they' refers to makeup examination fees"
                    }
                ]
            },
            {
                "session_id": "history_test_115",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Statistics program curriculum?",
                        "context_keywords": ["statistics", "program", "curriculum"]
                    },
                    {
                        "query": "What software packages are taught?",
                        "expected_context_reference": ["statistics", "program", "software", "packages"],
                        "context_test": "Should understand software packages are for Statistics program"
                    }
                ]
            },
            {
                "session_id": "history_test_116",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I apply for research ethics approval?",
                        "context_keywords": ["apply", "research", "ethics", "approval"]
                    },
                    {
                        "query": "What documentation is required for it?",
                        "expected_context_reference": ["research", "ethics", "approval", "documentation"],
                        "context_test": "Should understand 'it' refers to research ethics approval"
                    }
                ]
            },
            {
                "session_id": "history_test_117",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the conference presentation fees?",
                        "context_keywords": ["conference", "presentation", "fees"]
                    },
                    {
                        "query": "Are they reimbursable by the university?",
                        "expected_context_reference": ["conference", "presentation", "fees", "reimbursable"],
                        "context_test": "Should understand 'they' refers to conference presentation fees"
                    }
                ]
            },
            {
                "session_id": "history_test_118",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programs are available in the School of Pharmacy?",
                        "context_keywords": ["programs", "school", "pharmacy"]
                    },
                    {
                        "query": "What are the licensure requirements after completing them?",
                        "expected_context_reference": ["programs", "school", "pharmacy", "licensure", "requirements"],
                        "context_test": "Should understand 'them' refers to Pharmacy programs"
                    }
                ]
            },
            {
                "session_id": "history_test_119",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for academic probation removal?",
                        "context_keywords": ["requirements", "academic", "probation", "removal"]
                    },
                    {
                        "query": "How long does the process take?",
                        "expected_context_reference": ["academic", "probation", "removal", "process"],
                        "context_test": "Should understand process refers to probation removal"
                    }
                ]
            },
            {
                "session_id": "history_test_120",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there fees for course scheduling changes?",
                        "context_keywords": ["fees", "course", "scheduling", "changes"]
                    },
                    {
                        "query": "How often can I make them without penalty?",
                        "expected_context_reference": ["course", "scheduling", "changes", "penalty"],
                        "context_test": "Should understand 'them' refers to scheduling changes"
                    }
                ]
            },
            {
                "session_id": "history_test_121",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the International Relations program",
                        "context_keywords": ["international", "relations", "program"]
                    },
                    {
                        "query": "Are there study abroad requirements?",
                        "expected_context_reference": ["international", "relations", "study", "abroad", "requirements"],
                        "context_test": "Should understand study abroad requirements are for International Relations"
                    }
                ]
            },
            {
                "session_id": "history_test_122",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I apply for degree audit?",
                        "context_keywords": ["apply", "degree", "audit"]
                    },
                    {
                        "query": "How long does it take to complete?",
                        "expected_context_reference": ["degree", "audit", "complete"],
                        "context_test": "Should understand completion refers to degree audit"
                    }
                ]
            },
            {
                "session_id": "history_test_123",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the equipment rental fees?",
                        "context_keywords": ["equipment", "rental", "fees"]
                    },
                    {
                        "query": "What items are available for rent?",
                        "expected_context_reference": ["equipment", "rental", "fees", "items"],
                        "context_test": "Should understand items refers to rentable equipment"
                    }
                ]
            },
            {
                "session_id": "history_test_124",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Biochemistry program curriculum?",
                        "context_keywords": ["biochemistry", "program", "curriculum"]
                    },
                    {
                        "query": "What laboratory techniques are taught?",
                        "expected_context_reference": ["biochemistry", "program", "laboratory", "techniques"],
                        "context_test": "Should understand laboratory techniques are for Biochemistry program"
                    }
                ]
            },
            {
                "session_id": "history_test_125",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for academic appeal?",
                        "context_keywords": ["requirements", "academic", "appeal"]
                    },
                    {
                        "query": "What grounds are valid for it?",
                        "expected_context_reference": ["academic", "appeal", "grounds", "valid"],
                        "context_test": "Should understand 'it' refers to academic appeal"
                    }
                ]
            },
            {
                "session_id": "history_test_126",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there fees for diploma replacement?",
                        "context_keywords": ["fees", "diploma", "replacement"]
                    },
                    {
                        "query": "How long does it take to process them?",
                        "expected_context_reference": ["diploma", "replacement", "process"],
                        "context_test": "Should understand 'them' refers to diploma replacement requests"
                    }
                ]
            },
            {
                "session_id": "history_test_127",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programs are offered in the School of Architecture?",
                        "context_keywords": ["programs", "school", "architecture"]
                    },
                    {
                        "query": "What design studios are included?",
                        "expected_context_reference": ["programs", "school", "architecture", "design", "studios"],
                        "context_test": "Should understand design studios are for Architecture programs"
                    }
                ]
            },
            {
                "session_id": "history_test_128",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I apply for course waiver?",
                        "context_keywords": ["apply", "course", "waiver"]
                    },
                    {
                        "query": "What criteria are used to evaluate it?",
                        "expected_context_reference": ["course", "waiver", "criteria", "evaluate"],
                        "context_test": "Should understand 'it' refers to course waiver application"
                    }
                ]
            },
            {
                "session_id": "history_test_129",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the student teaching fees?",
                        "context_keywords": ["student", "teaching", "fees"]
                    },
                    {
                        "query": "Are they required for education majors?",
                        "expected_context_reference": ["student", "teaching", "fees", "education", "majors"],
                        "context_test": "Should understand 'they' refers to student teaching fees"
                    }
                ]
            },
            {
                "session_id": "history_test_130",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Criminology program",
                        "context_keywords": ["criminology", "program"]
                    },
                    {
                        "query": "What career paths does it prepare students for?",
                        "expected_context_reference": ["criminology", "program", "career", "paths"],
                        "context_test": "Should understand 'it' refers to Criminology program"
                    }
                ]
            },
            {
                "session_id": "history_test_131",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for academic reinstatement?",
                        "context_keywords": ["requirements", "academic", "reinstatement"]
                    },
                    {
                        "query": "How do I petition for it?",
                        "expected_context_reference": ["academic", "reinstatement", "petition"],
                        "context_test": "Should understand 'it' refers to academic reinstatement"
                    }
                ]
            },
            {
                "session_id": "history_test_132",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there fees for research facility usage?",
                        "context_keywords": ["fees", "research", "facility", "usage"]
                    },
                    {
                        "query": "How are they calculated per hour?",
                        "expected_context_reference": ["fees", "research", "facility", "calculated", "hour"],
                        "context_test": "Should understand 'they' refers to research facility fees"
                    }
                ]
            },
            {
                "session_id": "history_test_133",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Microbiology program curriculum?",
                        "context_keywords": ["microbiology", "program", "curriculum"]
                    },
                    {
                        "query": "What research opportunities are available?",
                        "expected_context_reference": ["microbiology", "program", "research", "opportunities"],
                        "context_test": "Should understand research opportunities are for Microbiology program"
                    }
                ]
            },
            {
                "session_id": "history_test_134",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I apply for graduation with honors?",
                        "context_keywords": ["apply", "graduation", "honors"]
                    },
                    {
                        "query": "What GPA is required for it?",
                        "expected_context_reference": ["graduation", "honors", "GPA", "required"],
                        "context_test": "Should understand 'it' refers to graduation with honors"
                    }
                ]
            },
            {
                "session_id": "history_test_135",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the thesis defense fees?",
                        "context_keywords": ["thesis", "defense", "fees"]
                    },
                    {
                        "query": "When do I need to pay them?",
                        "expected_context_reference": ["thesis", "defense", "fees", "pay"],
                        "context_test": "Should understand 'them' refers to thesis defense fees"
                    }
                ]
            },
            {
                "session_id": "history_test_136",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programs are available in the School of Fine Arts?",
                        "context_keywords": ["programs", "school", "fine", "arts"]
                    },
                    {
                        "query": "Do they require portfolio submissions?",
                        "expected_context_reference": ["programs", "school", "fine", "arts", "portfolio", "submissions"],
                        "context_test": "Should understand 'they' refers to Fine Arts programs"
                    }
                ]
            },
            {
                "session_id": "history_test_137",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for concurrent enrollment?",
                        "context_keywords": ["requirements", "concurrent", "enrollment"]
                    },
                    {
                        "query": "What institutions are eligible for it?",
                        "expected_context_reference": ["concurrent", "enrollment", "institutions", "eligible"],
                        "context_test": "Should understand 'it' refers to concurrent enrollment"
                    }
                ]
            },
            {
                "session_id": "history_test_138",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there fees for course materials?",
                        "context_keywords": ["fees", "course", "materials"]
                    },
                    {
                        "query": "Are they included in tuition?",
                        "expected_context_reference": ["fees", "course", "materials", "tuition"],
                        "context_test": "Should understand 'they' refers to course material fees"
                    }
                ]
            },
            {
                "session_id": "history_test_139",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Neuroscience program",
                        "context_keywords": ["neuroscience", "program"]
                    },
                    {
                        "query": "What laboratory facilities are available?",
                        "expected_context_reference": ["neuroscience", "program", "laboratory", "facilities"],
                        "context_test": "Should understand laboratory facilities are for Neuroscience program"
                    }
                ]
            },
            {
                "session_id": "history_test_140",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I apply for academic calendar change?",
                        "context_keywords": ["apply", "academic", "calendar", "change"]
                    },
                    {
                        "query": "What deadlines apply to it?",
                        "expected_context_reference": ["academic", "calendar", "change", "deadlines"],
                        "context_test": "Should understand 'it' refers to academic calendar change"
                    }
                ]
            },
            {
                "session_id": "history_test_141",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the examination proctoring fees?",
                        "context_keywords": ["examination", "proctoring", "fees"]
                    },
                    {
                        "query": "When are they applicable?",
                        "expected_context_reference": ["examination", "proctoring", "fees", "applicable"],
                        "context_test": "Should understand 'they' refers to proctoring fees"
                    }
                ]
            },
            {
                "session_id": "history_test_142",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What is the Marine Biology program curriculum?",
                        "context_keywords": ["marine", "biology", "program", "curriculum"]
                    },
                    {
                        "query": "Are there field station requirements?",
                        "expected_context_reference": ["marine", "biology", "program", "field", "station", "requirements"],
                        "context_test": "Should understand field station requirements are for Marine Biology"
                    }
                ]
            },
            {
                "session_id": "history_test_143",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for academic forgiveness?",
                        "context_keywords": ["requirements", "academic", "forgiveness"]
                    },
                    {
                        "query": "How many times can I apply for it?",
                        "expected_context_reference": ["academic", "forgiveness", "apply"],
                        "context_test": "Should understand 'it' refers to academic forgiveness"
                    }
                ]
            },
            {
                "session_id": "history_test_144",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "Are there fees for academic record corrections?",
                        "context_keywords": ["fees", "academic", "record", "corrections"]
                    },
                    {
                        "query": "How long does it take to process them?",
                        "expected_context_reference": ["academic", "record", "corrections", "process"],
                        "context_test": "Should understand 'them' refers to record corrections"
                    }
                ]
            },
            {
                "session_id": "history_test_145",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "What programs are offered in the School of Nursing?",
                        "context_keywords": ["programs", "school", "nursing"]
                    },
                    {
                        "query": "What clinical partnerships do they have?",
                        "expected_context_reference": ["programs", "school", "nursing", "clinical", "partnerships"],
                        "context_test": "Should understand 'they' refers to Nursing programs"
                    }
                ]
            },
            {
                "session_id": "history_test_146",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "How do I apply for course credit for life experience?",
                        "context_keywords": ["apply", "course", "credit", "life", "experience"]
                    },
                    {
                        "query": "What documentation is needed for it?",
                        "expected_context_reference": ["course", "credit", "life", "experience", "documentation"],
                        "context_test": "Should understand 'it' refers to life experience credit"
                    }
                ]
            },
            {
                "session_id": "history_test_147",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the distance learning fees?",
                        "context_keywords": ["distance", "learning", "fees"]
                    },
                    {
                        "query": "Are they different from campus-based programs?",
                        "expected_context_reference": ["distance", "learning", "fees", "campus-based", "programs"],
                        "context_test": "Should understand 'they' refers to distance learning fees"
                    }
                ]
            },
            {
                "session_id": "history_test_148",
                "topic": "programs_courses",
                "conversations": [
                    {
                        "query": "Tell me about the Aerospace Engineering program",
                        "context_keywords": ["aerospace", "engineering", "program"]
                    },
                    {
                        "query": "What industry partnerships does it have?",
                        "expected_context_reference": ["aerospace", "engineering", "program", "industry", "partnerships"],
                        "context_test": "Should understand 'it' refers to Aerospace Engineering program"
                    }
                ]
            },
            {
                "session_id": "history_test_149",
                "topic": "admissions_enrollment",
                "conversations": [
                    {
                        "query": "What are the requirements for academic suspension appeal?",
                        "context_keywords": ["requirements", "academic", "suspension", "appeal"]
                    },
                    {
                        "query": "What evidence should I provide for it?",
                        "expected_context_reference": ["academic", "suspension", "appeal", "evidence"],
                        "context_test": "Should understand 'it' refers to suspension appeal"
                    }
                ]
            },
            {
                "session_id": "history_test_150",
                "topic": "fees",
                "conversations": [
                    {
                        "query": "What are the comprehensive examination fees?",
                        "context_keywords": ["comprehensive", "examination", "fees"]
                    },
                    {
                        "query": "Are they required for all graduate programs?",
                        "expected_context_reference": ["comprehensive", "examination", "fees", "graduate", "programs"],
                        "context_test": "Should understand 'they' refers to comprehensive examination fees"
                    }
                ]
            }
        ]
    
    def send_guided_chat_request(self, user_input: str, action_type: str = "message", 
                               action_data: str = None, session_id: str = None) -> Dict:
        """Send request to guided chat endpoint without timeout"""
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
            print(f"❌ Request failed: {e}")
            return None
    
    def calculate_dialogue_history_bleu(self, response: str, expected_context: List[str], 
                                      previous_responses: List[str]) -> Dict:
        """Calculate BLEU score based on dialogue history context maintenance"""
        
        # Tokenize current response
        response_tokens = nltk.word_tokenize(response.lower())
        response_text = response.lower()
        
        # Count expected context references in current response
        context_matches = []
        for keyword in expected_context:
            if keyword.lower() in response_text:
                context_matches.append(keyword.lower())
        
        # Calculate context coverage
        context_coverage = len(context_matches) / len(expected_context) if expected_context else 0
        
        # Check for cross-reference to previous conversation elements
        previous_context_matches = []
        for prev_response in previous_responses:
            prev_tokens = nltk.word_tokenize(prev_response.lower())
            # Look for shared concepts between current and previous responses
            shared_tokens = set(response_tokens) & set(prev_tokens)
            # Filter to meaningful words (length > 2, not common words)
            meaningful_shared = [token for token in shared_tokens 
                               if len(token) > 2 and token not in ['the', 'and', 'for', 'are', 'you', 'can', 'will', 'have']]
            previous_context_matches.extend(meaningful_shared)
        
        # Remove duplicates
        previous_context_matches = list(set(previous_context_matches))
        
        # Calculate dialogue continuity score
        continuity_score = min(1.0, len(previous_context_matches) / 3)  # Normalize to max 1.0
        
        # Traditional BLEU score using expected context as reference
        try:
            reference = [expected_context]
            bleu_score = sentence_bleu(reference, response_tokens, smoothing_function=self.smoothing)
        except:
            bleu_score = 0.0
        
        # Combined dialogue history score
        # Weight: 50% context coverage, 30% continuity, 20% traditional BLEU
        dialogue_history_score = (context_coverage * 0.5) + (continuity_score * 0.3) + (bleu_score * 0.2)
        
        return {
            "context_coverage": context_coverage,
            "continuity_score": continuity_score,
            "bleu_score": bleu_score,
            "dialogue_history_score": dialogue_history_score,
            "context_matches": context_matches,
            "previous_context_matches": previous_context_matches,
            "response_length": len(response_tokens)
        }
    
    def run_dialogue_session(self, session_data: Dict) -> Dict:
        """Run a complete dialogue session and test history tracking"""
        session_id = session_data["session_id"]
        topic = session_data["topic"]
        conversations = session_data["conversations"]
        
        print(f"\n{'='*70}")
        print(f"🗣️ Testing Dialogue History: {session_id}")
        print(f"📋 Topic: {topic}")
        print(f"💬 Conversations: {len(conversations)}")
        print(f"🎯 Testing: Context maintenance and pronoun resolution")
        print(f"{'='*70}")
        
        # Initialize session with topic selection
        topic_response = self.send_guided_chat_request("", "topic_selection", topic, session_id)
        if not topic_response:
            return {"session_id": session_id, "error": "Failed to initialize session"}
        
        print(f"✅ Topic selected: {topic_response.get('current_topic')}")
        
        session_scores = []
        conversation_results = []
        all_responses = []
        
        # Run each conversation in the session
        for i, conv in enumerate(conversations, 1):
            print(f"\n💬 Exchange {i}: {conv['query']}")
            
            # Send query and get response
            start_time = time.time()
            response_data = self.send_guided_chat_request(conv['query'], "message", None, session_id)
            response_time = time.time() - start_time
            
            if not response_data or not response_data.get('response'):
                print(f"❌ No response received")
                conversation_results.append({
                    "exchange": i,
                    "query": conv['query'],
                    "error": "No response received",
                    "dialogue_history_score": 0.0
                })
                continue
            
            bot_response = response_data['response']
            all_responses.append(bot_response)
            print(f"🤖 Response ({response_time:.2f}s): {bot_response[:100]}...")
            
            # For first exchange, just store context keywords
            if i == 1:
                print(f"📝 Establishing context: {conv.get('context_keywords', [])}")
                conversation_results.append({
                    "exchange": i,
                    "query": conv['query'],
                    "response": bot_response,
                    "context_keywords": conv.get('context_keywords', []),
                    "dialogue_history_score": 1.0,  # First exchange always gets full score
                    "analysis": "Initial context establishment"
                })
                session_scores.append(1.0)
                continue
            
            # For follow-up exchanges, test dialogue history
            expected_context = conv.get('expected_context_reference', [])
            context_test = conv.get('context_test', '')
            
            print(f"🔍 Testing: {context_test}")
            print(f"🎯 Expected context: {expected_context}")
            
            # Calculate dialogue history BLEU score
            scores = self.calculate_dialogue_history_bleu(
                bot_response, expected_context, all_responses[:-1]  # Previous responses
            )
            
            session_scores.append(scores['dialogue_history_score'])
            
            conversation_results.append({
                "exchange": i,
                "query": conv['query'],
                "response": bot_response,
                "expected_context": expected_context,
                "context_test": context_test,
                "scores": scores,
                "dialogue_history_score": scores['dialogue_history_score']
            })
            
            print(f"📊 Context Coverage: {scores['context_coverage']:.3f}")
            print(f"📊 Continuity Score: {scores['continuity_score']:.3f}")
            print(f"📊 Dialogue History Score: {scores['dialogue_history_score']:.3f}")
            print(f"✅ Context Matches: {scores['context_matches']}")
            print(f"🔗 Previous Context: {scores['previous_context_matches']}")
            
            # Small delay between exchanges
            time.sleep(1)
        
        # Calculate session average
        avg_score = sum(session_scores) / len(session_scores) if session_scores else 0.0
        
        return {
            "session_id": session_id,
            "topic": topic,
            "conversations": conversation_results,
            "session_scores": session_scores,
            "average_dialogue_history_score": avg_score,
            "exchange_count": len(conversations)
        }
    
    def run_all_sessions(self) -> Dict:
        """Run all dialogue history test sessions"""
        print("🚀 Starting Dialogue History BLEU Evaluation")
        print("🎯 Target: BLEU Score ≥ 0.40 for dialogue history tracking")
        print("🔍 Testing: Context maintenance, pronoun resolution, conversation continuity")
        print("="*80)
        
        all_results = []
        all_scores = []
        
        for session_data in self.dialogue_sessions:
            result = self.run_dialogue_session(session_data)
            all_results.append(result)
            
            if 'average_dialogue_history_score' in result:
                all_scores.append(result['average_dialogue_history_score'])
        
        # Calculate overall statistics
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        
        return {
            "test_results": all_results,
            "overall_dialogue_history_score": overall_score,
            "target_score": 0.40,
            "target_achieved": overall_score >= 0.40,
            "total_sessions": len(all_results),
            "successful_sessions": len(all_scores)
        }
    
    def print_results(self, results: Dict):
        """Print comprehensive dialogue history analysis"""
        print(f"\n{'='*80}")
        print("📊 DIALOGUE HISTORY BLEU EVALUATION RESULTS")
        print(f"{'='*80}")
        
        print(f"🎯 Target Score: {results['target_score']}")
        print(f"📈 Achieved Score: {results['overall_dialogue_history_score']:.4f}")
        print(f"✅ Target Achieved: {'YES' if results['target_achieved'] else 'NO'}")
        print(f"📊 Total Sessions: {results['total_sessions']}")
        print(f"✅ Successful Sessions: {results['successful_sessions']}")
        
        print(f"\n📋 SESSION BREAKDOWN:")
        for result in results['test_results']:
            if 'average_dialogue_history_score' in result:
                status = "✅" if result['average_dialogue_history_score'] >= 0.40 else "❌"
                print(f"   {status} {result['session_id']}: {result['average_dialogue_history_score']:.4f} "
                      f"({result['exchange_count']} exchanges) - {result['topic']}")
            else:
                print(f"   ❌ {result['session_id']}: ERROR")
        
        # Detailed analysis of dialogue history issues
        print(f"\n🔍 DIALOGUE HISTORY ANALYSIS:")
        
        total_context_coverage = 0
        total_continuity = 0
        total_exchanges = 0
        
        context_failures = []
        continuity_failures = []
        
        for result in results['test_results']:
            if 'conversations' in result:
                for conv in result['conversations']:
                    if 'scores' in conv:
                        scores = conv['scores']
                        total_context_coverage += scores['context_coverage']
                        total_continuity += scores['continuity_score']
                        total_exchanges += 1
                        
                        # Track failures
                        if scores['context_coverage'] < 0.5:
                            context_failures.append({
                                'session': result['session_id'],
                                'query': conv['query'],
                                'expected': conv.get('expected_context', []),
                                'matches': scores['context_matches'],
                                'score': scores['context_coverage']
                            })
                        
                        if scores['continuity_score'] < 0.3:
                            continuity_failures.append({
                                'session': result['session_id'],
                                'query': conv['query'],
                                'score': scores['continuity_score'],
                                'previous_matches': scores['previous_context_matches']
                            })
        
        if total_exchanges > 0:
            avg_context_coverage = total_context_coverage / total_exchanges
            avg_continuity = total_continuity / total_exchanges
            
            print(f"   Average Context Coverage: {avg_context_coverage:.3f}")
            print(f"   Average Continuity Score: {avg_continuity:.3f}")
            print(f"   Context Failures: {len(context_failures)}/{total_exchanges}")
            print(f"   Continuity Failures: {len(continuity_failures)}/{total_exchanges}")
        
        # Show specific failure examples
        if context_failures:
            print(f"\n❌ CONTEXT RESOLUTION FAILURES:")
            for failure in context_failures[:3]:  # Show top 3
                print(f"   Session: {failure['session']}")
                print(f"   Query: {failure['query']}")
                print(f"   Expected: {failure['expected']}")
                print(f"   Found: {failure['matches']} (Score: {failure['score']:.3f})")
                print()
        
        if continuity_failures:
            print(f"\n❌ CONVERSATION CONTINUITY FAILURES:")
            for failure in continuity_failures[:3]:  # Show top 3
                print(f"   Session: {failure['session']}")
                print(f"   Query: {failure['query']}")
                print(f"   Previous Context Found: {failure['previous_matches']}")
                print(f"   Continuity Score: {failure['score']:.3f}")
                print()

def main():
    """Main test runner"""
    print("🧪 Dialogue History BLEU Evaluation Test")
    print("⚠️  Make sure Django server is running: python manage.py runserver")
    print("⚠️  This test focuses on LLaMA's dialogue history tracking ability")
    print("⚠️  Testing context maintenance, pronoun resolution, and conversation continuity")
    
    input("\nPress ENTER to start the evaluation...")
    
    evaluator = DialogueHistoryBLEUEvaluator()
    
    try:
        results = evaluator.run_all_sessions()
        evaluator.print_results(results)
        
        # Save results to file
        with open('dialogue_history_bleu_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: dialogue_history_bleu_results.json")
        
        # Exit with appropriate code
        sys.exit(0 if results['target_achieved'] else 1)
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

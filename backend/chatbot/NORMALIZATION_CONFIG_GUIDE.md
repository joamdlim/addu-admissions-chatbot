# Program Normalization Configuration Guide

## Overview

The ADDU Admissions Chatbot uses a configurable, JSON-driven system to normalize program abbreviations in user queries. This system prevents common English words like "is", "it", "me" from being incorrectly interpreted as program abbreviations while still allowing proper normalization when appropriate.

## Configuration File

**Location:** `backend/chatbot/program_normalization_config.json`

## Complete Program List (59 Programs)

### School of Arts & Sciences (31 programs)

#### Humanities & Letters (9 programs)

- **AB ENG** - Bachelor of Arts in English Language
- **AB MC** - Bachelor of Arts in Mass Communication
- **AB IDS-LL** - Bachelor of Arts in Interdisciplinary Studies, minor in Language & Literature
- **AB IDS-MB** - Bachelor of Arts in Interdisciplinary Studies, minor in Media & Business
- **AB IDS-MP** - Bachelor of Arts in Interdisciplinary Studies, minor in Media & Philosophy
- **AB IDS-MT** - Bachelor of Arts in Interdisciplinary Studies, minor in Media & Technology
- **AB IDS-PT** - Bachelor of Arts in Interdisciplinary Studies, minor in Philosophy & Theology
- **AB PHILO** - Bachelor of Arts Major in Philosophy (Pre-Law)

#### Natural Sciences & Mathematics (6 programs)

- **BS BIO** - Bachelor of Science in Biology (General Biology)
- **BS BIO - MEDBIO** - Bachelor of Science in Biology Major in Medical Biology
- **BS CHEM** - Bachelor of Science in Chemistry
- **BS MATH** - Bachelor of Science in Mathematics
- **BS ENVI SCI** - Bachelor of Science in Environmental Science

#### Computer Studies (4 programs)

- **BS IS** - Bachelor of Science in Information Systems
- **BS IT** - Bachelor of Science in Information Technology
- **BS CS** - Bachelor of Science in Computer Science
- **BS DS** - Bachelor of Science in Data Science

#### Social Sciences (12 programs)

- **AB ECON** - Bachelor of Arts Major in Economics
- **AB DS** - Bachelor of Arts in Development Studies
- **AB POLSCI** - Bachelor of Arts Major in Political Studies
- **AB PSYCH** - Bachelor of Arts Major in Psychology
- **AB SOCIO** - Bachelor of Arts Major in Sociology
- **AB IS** - Bachelor of Arts in Islamic Studies
- **AB IS - AMERICAN STUDIES** - Bachelor of Arts in International Studies Major in American Studies
- **AB IS - ASIAN STUDIES** - Bachelor of Arts in International Studies Major in Asian Studies
- **AB ANTHRO - ACADRES** - Bachelor of Arts in Anthropology - Academic Research
- **AB ANTHRO - COMDEV** - Bachelor of Arts in Anthropology - Community Development/Social Enterprise
- **AB ANTHRO - IPED** - Bachelor of Arts in Anthropology - IP Education
- **AB ANTHRO - MEDANTH** - Bachelor of Arts in Anthropology - Medical Anthropology
- **AB ANTHRO - PRELAW** - Bachelor of Arts in Anthropology - Leadership/Pre-Law
- **BS SOCIAL WORK** - Bachelor of Science in Social Work

### School of Business & Governance (10 programs)

- **BS A** - Bachelor of Science in Accountancy
- **BS MA** - Bachelor of Science in Management Accounting
- **BS BM** - Bachelor of Science in Business Management
- **BS ENTREP** - Bachelor of Science in Entrepreneurship
- **BS ENTREP-A** - Bachelor of Science in Entrepreneurship Major in Agri-Business
- **BS FIN** - Bachelor of Science in Finance
- **BS HRDM** - Bachelor of Science in Human Resource Development and Management
- **BS MKTG** - Bachelor of Science in Marketing
- **BPM** - Bachelor of Public Management

### School of Education (7 programs)

- **BECE** - Bachelor of Early Childhood Education
- **BEED** - Bachelor of Elementary Education
- **BSED - English** - Bachelor of Secondary Education Major in English
- **BSED - Math** - Bachelor of Secondary Education Major in Mathematics
- **BSED - Science** - Bachelor of Secondary Education Major in Science
- **BSED - SS** - Bachelor of Secondary Education Major in Social Studies

### School of Engineering & Architecture (10 programs)

- **BS AE** - Bachelor of Science in Aerospace Engineering
- **BS ARCH** - Bachelor of Science in Architecture
- **BS CHE** - Bachelor of Science in Chemical Engineering
- **BS CE** - Bachelor of Science in Civil Engineering
- **BS COMP ENG** - Bachelor of Science in Computer Engineering
- **BS EE** - Bachelor of Science in Electrical Engineering
- **BS ELECTRONICS ENG** - Bachelor of Science in Electronics Engineering
- **BS IE** - Bachelor of Science in Industrial Engineering
- **BS ME** - Bachelor of Science in Mechanical Engineering
- **BS RE** - Bachelor of Science in Robotics Engineering

### School of Nursing (1 program)

- **BS N** - Bachelor of Science in Nursing

## Priority System

The system uses a 3-tier priority system to determine when to normalize abbreviations:

### 1. Safe (Always Normalize)

These are full program codes that are unambiguous and should always be normalized:

- `bscs` → `BS CS`
- `bsit` → `BS IT`
- `abeng` → `AB ENG`
- `bsarch` → `BS ARCH`

### 2. Context-Aware (Normalize with Program Keywords)

These abbreviations are normalized when program-related keywords are detected nearby:

- `cs` → `BS CS` (when "program", "course", "degree", etc. are present)
- `bio` → `BS BIO` (when program context is detected)
- `math` → `BS MATH` (when program context is detected)

### 3. Context-Required (Strong Context Only)

These are common English words that need very strong program context:

- `is` → `BS IS` (only with explicit patterns like "BS IS program")
- `it` → `BS IT` (only with explicit patterns like "study IT")
- `me` → `BS ME` (only with explicit patterns like "ME curriculum")

## Configuration Schema

```json
{
  "program_abbreviations": {
    "School Name": {
      "abbreviation": {
        "full_name": "Official Program Name",
        "priority": "safe|context_aware|context_required",
        "is_common_word": true|false,
        "conflicts_with": ["other_abbreviations"],
        "description": "Human readable description"
      }
    }
  },
  "context_patterns": {
    "problematic": ["regex patterns for problematic contexts"],
    "program_keywords": ["words that indicate program context"],
    "strong_program": ["regex patterns for strong program context"]
  },
  "common_english_words": ["list of common words needing context checks"],
  "conflict_resolution": {
    "abbreviation": {
      "Program1": "Description",
      "Program2": "Description"
    }
  }
}
```

## Adding New Programs

To add a new program:

1. **Edit the JSON file** (`program_normalization_config.json`)
2. **Add to appropriate school section:**

```json
"new_abbrev": {
  "full_name": "BS NEW",
  "priority": "context_aware",
  "description": "Bachelor of Science in New Program"
}
```

3. **Choose appropriate priority:**

   - Use `"safe"` for unambiguous abbreviations
   - Use `"context_aware"` for most abbreviations
   - Use `"context_required"` only for common English words

4. **Add `is_common_word: true`** if the abbreviation is a common English word

## Updating Context Patterns

### Adding Problematic Patterns

Add regex patterns that indicate the word should NOT be normalized:

```json
"problematic": [
  "\\bwhat\\s+is\\s+the\\b",
  "\\bshow\\s+me\\s+the\\b"
]
```

### Adding Program Keywords

Add words that indicate program context:

```json
"program_keywords": [
  "program", "course", "degree", "curriculum", "major"
]
```

### Adding Strong Program Patterns

Add patterns for strong program context (use `{abbrev}` placeholder):

```json
"strong_program": [
  "\\b{abbrev}\\s+(program|course)\\b",
  "\\bstudy\\s+{abbrev}\\b"
]
```

## Examples

### Good Normalizations

- `"Tell me about CS program"` → `"Tell me about BS CS program"`
- `"What are the requirements for bsit"` → `"What are the requirements for BS IT"`
- `"I want to study bio"` → `"I want to study BS BIO"`

### Protected from Normalization

- `"What is the enrollment process"` → `"What is the enrollment process"` (no change)
- `"Show me the form"` → `"Show me the form"` (no change)
- `"How do I apply for it"` → `"How do I apply for it"` (no change)

### Conflict Resolution

- `"Tell me about IS program"` → Checks context for computer vs social sciences
- `"What is Islamic studies"` → `AB IS` (Islamic Studies)
- `"Information systems course"` → `BS IS` (Information Systems)

## Testing Changes

After modifying the configuration:

1. **Restart the chatbot** to reload the config
2. **Test with various queries** to ensure proper normalization
3. **Check the logs** for normalization decisions
4. **Verify no false positives** (common words being incorrectly normalized)

## Troubleshooting

### Common Issues

1. **Abbreviation not normalizing:**

   - Check if it's in the config file
   - Verify the priority level is appropriate
   - Ensure program context is present for `context_aware` items

2. **Common word being incorrectly normalized:**

   - Add to `common_english_words` list
   - Set priority to `context_required`
   - Add problematic patterns if needed

3. **Config not loading:**
   - Check JSON syntax with a validator
   - Verify file path is correct
   - Check console logs for error messages

### Debug Logs

The system logs normalization decisions:

- `[CONFIG] Loaded normalization config from...`
- `[NORM] Applied normalizations: ...`
- `[ERROR] Normalization config file not found...`

## Performance Notes

- Configuration is loaded once and cached in memory
- Use `_reload_normalization_config()` to force reload during development
- Large numbers of patterns may impact performance slightly

## Migration from Hardcoded System

The new system maintains backward compatibility while providing:

- Easy maintenance without code changes
- Consistent treatment of all abbreviations
- Better conflict resolution
- Comprehensive logging and debugging

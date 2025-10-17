import pandas as pd
import io
from typing import Dict, List, Tuple


def process_excel_csv_file(file_bytes: bytes, file_name: str) -> str:
    """
    Process Excel or CSV file and convert to structured text format.
    
    Args:
        file_bytes: Raw file bytes
        file_name: Name of the file
        
    Returns:
        Structured text representation of the data
    """
    try:
        # Determine file type and read accordingly
        if file_name.lower().endswith('.csv'):
            # For CSV files, try to handle multi-section structure
            structured_text = process_multi_section_csv(file_bytes, file_name)
        elif file_name.lower().endswith(('.xlsx', '.xls')):
            # For Excel files, also try multi-section processing
            structured_text = process_multi_section_excel(file_bytes, file_name)
        else:
            raise ValueError(f"Unsupported file type: {file_name}")
        
        return structured_text
        
    except Exception as e:
        raise Exception(f"Error processing {file_name}: {str(e)}")


def convert_dataframe_to_text(df: pd.DataFrame, file_name: str) -> str:
    """
    Convert pandas DataFrame to structured text format optimized for chatbot retrieval.
    
    Args:
        df: Pandas DataFrame
        file_name: Original file name for context
        
    Returns:
        Structured text representation
    """
    if df.empty:
        return f"File: {file_name}\nNo data found in the file."
    
    # Clean column names
    df.columns = df.columns.astype(str).str.strip()
    
    # Start building structured text
    text_parts = []
    text_parts.append(f"FEES INFORMATION FROM: {file_name}")
    text_parts.append("=" * 50)
    
    # Add column information
    text_parts.append(f"Data contains {len(df)} rows and {len(df.columns)} columns")
    text_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
    text_parts.append("")
    
    # Detect if this looks like a fees table
    fees_keywords = ['fee', 'cost', 'price', 'amount', 'tuition', 'payment', 'program', 'course', 'semester', 'year']
    is_fees_table = any(keyword in ' '.join(df.columns).lower() for keyword in fees_keywords)
    
    if is_fees_table:
        text_parts.extend(_format_fees_table(df))
    else:
        text_parts.extend(_format_general_table(df))
    
    # Add summary section
    text_parts.append("")
    text_parts.append("SUMMARY:")
    text_parts.append("-" * 20)
    
    # Try to extract key information (handle mixed data types)
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    # Also check for columns that might contain numeric data as strings
    potential_numeric_cols = []
    for col in df.columns:
        if col not in numeric_columns:
            # Try to convert to numeric
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if not numeric_series.isna().all():
                    potential_numeric_cols.append(col)
            except:
                continue
    
    all_numeric_cols = list(numeric_columns) + potential_numeric_cols
    
    if len(all_numeric_cols) > 0:
        text_parts.append("Numerical Data Summary:")
        for col in all_numeric_cols:
            if col in numeric_columns:
                # Already numeric
                if not df[col].isna().all():
                    min_val = df[col].min()
                    max_val = df[col].max()
                    mean_val = df[col].mean()
                    text_parts.append(f"  {col}: Range {min_val:,.2f} - {max_val:,.2f}, Average: {mean_val:,.2f}")
            else:
                # Convert to numeric for summary
                try:
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_series.isna().all():
                        min_val = numeric_series.min()
                        max_val = numeric_series.max()
                        mean_val = numeric_series.mean()
                        text_parts.append(f"  {col}: Range {min_val:,.2f} - {max_val:,.2f}, Average: {mean_val:,.2f}")
                except:
                    continue
    
    # Add searchable keywords
    text_parts.append("")
    text_parts.append("KEYWORDS: fees, tuition, payment, cost, price, amount, billing, financial, program fees")
    
    return "\n".join(text_parts)


def _format_fees_table(df: pd.DataFrame) -> List[str]:
    """Format DataFrame as a fees table with enhanced readability."""
    text_parts = []
    text_parts.append("FEES TABLE:")
    text_parts.append("-" * 30)
    
    # Try to identify program and fee columns
    program_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['program', 'course', 'degree', 'major'])]
    fee_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['fee', 'cost', 'price', 'amount', 'tuition'])]
    
    # Format each row
    for idx, row in df.iterrows():
        row_text = []
        
        # Add program information first
        for col in program_cols:
            if pd.notna(row[col]):
                row_text.append(f"Program: {row[col]}")
        
        # Add fee information
        for col in fee_cols:
            if pd.notna(row[col]):
                try:
                    # Try to format as currency - handle both numeric and string values
                    if isinstance(row[col], (int, float)):
                        value = float(row[col])
                        row_text.append(f"{col}: ₱{value:,.2f}")
                    else:
                        # Try to convert string to numeric
                        numeric_value = pd.to_numeric(str(row[col]).replace(',', '').replace('₱', '').replace('$', ''), errors='coerce')
                        if not pd.isna(numeric_value):
                            row_text.append(f"{col}: ₱{numeric_value:,.2f}")
                        else:
                            row_text.append(f"{col}: {row[col]}")
                except (ValueError, TypeError):
                    row_text.append(f"{col}: {row[col]}")
        
        # Add other columns
        other_cols = [col for col in df.columns if col not in program_cols and col not in fee_cols]
        for col in other_cols:
            if pd.notna(row[col]):
                row_text.append(f"{col}: {row[col]}")
        
        if row_text:
            text_parts.append(f"{idx + 1}. {' | '.join(row_text)}")
    
    return text_parts


def _format_general_table(df: pd.DataFrame) -> List[str]:
    """Format DataFrame as a general table."""
    text_parts = []
    text_parts.append("DATA TABLE:")
    text_parts.append("-" * 30)
    
    # Format each row
    for idx, row in df.iterrows():
        row_parts = []
        for col in df.columns:
            if pd.notna(row[col]):
                row_parts.append(f"{col}: {row[col]}")
        
        if row_parts:
            text_parts.append(f"{idx + 1}. {' | '.join(row_parts)}")
    
    return text_parts


def validate_fees_file(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate if the uploaded file contains valid fees data.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Tuple of (is_valid, message)
    """
    if df.empty:
        return False, "File is empty or contains no data"
    
    # Check for required columns (flexible matching)
    required_keywords = ['fee', 'cost', 'price', 'amount', 'tuition']
    column_text = ' '.join(df.columns).lower()
    
    has_fee_column = any(keyword in column_text for keyword in required_keywords)
    
    if not has_fee_column:
        return False, "File does not appear to contain fees information. Please ensure it has columns related to fees, costs, or tuition."
    
    # Check for numeric data (more flexible approach)
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    # If no columns are detected as numeric, try to find numeric data in string columns
    if len(numeric_columns) == 0:
        # Check if any columns contain numeric data (even if stored as strings)
        has_numeric_data = False
        for col in df.columns:
            # Try to convert a sample of values to numeric
            sample_values = df[col].dropna().head(10)  # Check first 10 non-null values
            if len(sample_values) > 0:
                try:
                    # Try to convert to numeric
                    pd.to_numeric(sample_values, errors='coerce')
                    # If we can convert at least some values, consider it numeric
                    if not pd.to_numeric(sample_values, errors='coerce').isna().all():
                        has_numeric_data = True
                        break
                except:
                    continue
        
        if not has_numeric_data:
            return False, "File does not contain any numeric data for fees"
    
    return True, "File appears to contain valid fees data"


def process_multi_section_csv(file_bytes: bytes, file_name: str) -> str:
    """
    Process multi-section CSV files with proper section detection and column mapping.
    
    Args:
        file_bytes: Raw file bytes
        file_name: Name of the file
        
    Returns:
        Structured text representation of the data
    """
    try:
        # Read the CSV file as text first to analyze structure
        csv_text = file_bytes.decode('utf-8')
        lines = csv_text.strip().split('\n')
        
        text_parts = []
        text_parts.append(f"FEES INFORMATION FROM: {file_name}")
        text_parts.append("=" * 50)
        
        # Parse the CSV structure more carefully
        sections = parse_csv_sections(lines)
        
        # Debug: Print what sections were found
        print(f"DEBUG: Found {len(sections)} sections")
        for i, (section_name, headers, data) in enumerate(sections):
            print(f"DEBUG: Section {i+1}: '{section_name}', Headers: {len(headers) if headers else 0}, Data rows: {len(data) if data else 0}")
        
        for section_name, headers, data in sections:
            if section_name and headers and data:
                section_text = format_csv_section(section_name, headers, data)
                text_parts.extend(section_text)
            else:
                print(f"DEBUG: Skipping section '{section_name}' - missing headers or data")
        
        # Add summary
        text_parts.append("")
        text_parts.append("SUMMARY:")
        text_parts.append("-" * 20)
        text_parts.append("This file contains comprehensive fee information including:")
        text_parts.append("- School fees (general fees for all students by year level)")
        text_parts.append("- Course fees (program-specific fees)")
        text_parts.append("- Tuition fees (per-unit rates with semester breakdown)")
        text_parts.append("- Program fees (estimated total tuition by program and year)")
        text_parts.append("- Detailed semester information (units and tuition per semester)")
        text_parts.append("- Summer program fees and unit information")
        
        # Extract all program names and acronyms for keywords
        program_keywords = []
        for section_name, headers, data in sections:
            if section_name and section_name.lower() == "program fees" and data:
                for row in data:
                    if len(row) > 0 and row[0]:
                        program_name = str(row[0]).strip()
                        if program_name not in program_keywords:
                            program_keywords.append(program_name)
                        
                        # Add acronym if available
                        if len(headers) >= 9 and len(row) > 1 and row[1]:
                            acronym = str(row[1]).strip()
                            if acronym and acronym not in program_keywords:
                                program_keywords.append(acronym)
        
        text_parts.append("")
        base_keywords = "fees, tuition, payment, cost, price, amount, billing, financial, program fees, registration, graduation, school fees, semester, units, summer"
        if program_keywords:
            all_keywords = base_keywords + ", " + ", ".join(program_keywords)
            text_parts.append(f"KEYWORDS: {all_keywords}")
        else:
            text_parts.append(f"KEYWORDS: {base_keywords}")
        
        return "\n".join(text_parts)
        
    except Exception as e:
        # Fallback to regular CSV processing
        df = pd.read_csv(io.BytesIO(file_bytes))
        return convert_dataframe_to_text(df, file_name)


def parse_csv_sections(lines):
    """
    Parse CSV lines into sections with proper header and data separation.
    
    Returns:
        List of tuples: (section_name, headers, data_rows)
    """
    sections = []
    current_section = None
    current_headers = None
    current_data = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        # Parse CSV line
        fields = parse_csv_line(line)
        
        # Debug output
        print(f"DEBUG: Line {i+1}: {fields[:3]}... (Section: {current_section}, Has headers: {current_headers is not None})")
        
        # Check if this is a section header
        if is_section_header(fields):
            # Clean the section name (remove BOM)
            clean_section_name = fields[0].strip().lstrip('\ufeff')
            print(f"DEBUG: Found section header: {clean_section_name}")
            # Save previous section
            if current_section and current_headers and current_data:
                sections.append((current_section, current_headers, current_data))
                print(f"DEBUG: Saved section '{current_section}' with {len(current_data)} data rows")
            
            # Start new section
            current_section = clean_section_name
            current_headers = None
            current_data = []
            i += 1
            continue
        
        # Check if this is a header row (column names)
        if current_section and is_header_row(fields):
            if not current_headers:
                # First header row in this section
                current_headers = fields
                print(f"DEBUG: Set headers: {fields}")
            else:
                # Repeated header row - skip it but don't reset headers
                print(f"DEBUG: Skipping repeated header row")
            i += 1
            continue
        
        # This is a data row
        if current_headers and is_data_row(fields):
            current_data.append(fields)
            print(f"DEBUG: Added data row: {fields[:2]}...")
        else:
            print(f"DEBUG: Skipping row (not data): {fields[:2]}...")
        
        i += 1
    
    # Save the last section
    if current_section and current_headers and current_data:
        sections.append((current_section, current_headers, current_data))
    
    return sections


def parse_csv_line(line):
    """Parse a CSV line handling quoted fields."""
    fields = []
    current_field = ""
    in_quotes = False
    
    for char in line:
        if char == '"':
            in_quotes = not in_quotes
        elif char == ',' and not in_quotes:
            fields.append(current_field.strip())
            current_field = ""
            continue
        current_field += char
    fields.append(current_field.strip())
    
    return fields


def is_section_header(fields):
    """Check if this row is a section header."""
    if len(fields) == 0 or not fields[0]:
        return False
    
    # Clean the first field (remove BOM and whitespace)
    first_field = fields[0].strip().lstrip('\ufeff')
    
    # Section headers are like "School Fees", "Course Fees", "Tuition Fees", "Program Fees"
    section_names = ["School Fees", "Course Fees", "Tuition Fees", "Program Fees"]
    if first_field in section_names:
        # Check if all other fields are empty or the same as the first field
        return all(field == "" or field == fields[0] for field in fields[1:])
    
    return False


def is_header_row(fields):
    """Check if this row contains column headers."""
    if len(fields) < 2:
        return False
    
    # Look for common header keywords
    header_keywords = ['school', 'course', 'fee', 'year', 'tuition', 'registration', 'level', 'program', 'units', 'semester']
    line_text = ' '.join(fields).lower()
    
    return any(keyword in line_text for keyword in header_keywords)


def is_data_row(fields):
    """Check if this row contains data."""
    if len(fields) < 2:
        return False
    
    # Must have at least 2 non-empty fields
    non_empty_fields = [f for f in fields if f.strip()]
    if len(non_empty_fields) < 2:
        return False
    
    # Don't treat header rows as data rows
    line_text = ' '.join(fields).lower()
    header_indicators = ['program', 'year level', 'units', 'estimated tuition fees', 'semester', 'fee name', 'school']
    if any(indicator in line_text for indicator in header_indicators):
        # Check if this looks like a header (contains multiple header keywords)
        header_count = sum(1 for indicator in header_indicators if indicator in line_text)
        if header_count >= 2:  # If it has 2+ header keywords, it's probably a header row
            return False
    
    return True


def process_multi_section_excel(file_bytes: bytes, file_name: str) -> str:
    """
    Process multi-section Excel files with proper section detection and column mapping.
    
    Args:
        file_bytes: Raw file bytes
        file_name: Name of the file
        
    Returns:
        Structured text representation of the data
    """
    try:
        # Read Excel file and convert to CSV-like structure for processing
        df = pd.read_excel(io.BytesIO(file_bytes), header=None)
        
        # Convert DataFrame to text lines for processing
        lines = []
        for _, row in df.iterrows():
            # Convert row to comma-separated string, handling NaN values
            row_values = []
            for value in row:
                if pd.isna(value):
                    row_values.append("")
                else:
                    row_values.append(str(value))
            lines.append(",".join(row_values))
        
        # Join lines and process as CSV
        csv_text = "\n".join(lines)
        csv_bytes = csv_text.encode('utf-8')
        
        # Use the CSV processing function
        return process_multi_section_csv(csv_bytes, file_name)
        
    except Exception as e:
        # Fallback to regular Excel processing
        df = pd.read_excel(io.BytesIO(file_bytes))
        return convert_dataframe_to_text(df, file_name)


def format_csv_section(section_name: str, headers: List[str], data: List[List[str]]) -> List[str]:
    """
    Format a CSV section with proper column names and structure.
    
    Args:
        section_name: Name of the section (e.g., "School Fees")
        headers: Column headers
        data: Data rows
        
    Returns:
        List of formatted text lines
    """
    text_parts = []
    text_parts.append("")
    text_parts.append(f"{section_name.upper()}:")
    text_parts.append("-" * 40)
    
    if not headers or not data:
        text_parts.append("No data available")
        return text_parts
    
    # Clean up headers - remove empty ones and give meaningful names
    clean_headers = []
    for i, header in enumerate(headers):
        if header and header.strip():
            clean_headers.append(header.strip())
        else:
            # Generate meaningful names for empty headers based on position and section
            if section_name.lower() == "tuition fees":
                # Enhanced tuition fees structure
                tuition_header_map = {
                    0: "School",
                    1: "Year Level", 
                    2: "Tuition Fee Rate Per Unit",
                    3: "Empty Column",
                    4: "Units - 1st Sem",
                    5: "Tuition 1st Sem",
                    6: "Units - 2nd Sem", 
                    7: "Tuition 2nd Sem",
                    8: "Summer Units",
                    9: "Summer Tuition"
                }
                clean_headers.append(tuition_header_map.get(i, f"Column {i+1}"))
            elif section_name.lower() in ["school fees", "course fees"]:
                # School fees and course fees structure
                fee_header_map = {
                    0: "School" if section_name.lower() == "school fees" else "Course",
                    1: "Fee Name",
                    2: "First Year",
                    3: "Second Year", 
                    4: "Third Year",
                    5: "Fourth Year",
                    6: "Fifth Year"
                }
                clean_headers.append(fee_header_map.get(i, f"Column {i+1}"))
            elif section_name.lower() == "program fees":
                # Program fees structure - estimated tuition by program with units
                # Handle different formats based on column count
                if len(headers) >= 10:  # SON format with Acronym and Summer (10 cols)
                    program_fee_header_map = {
                        0: "Program",
                        1: "Acronym",
                        2: "Year Level",
                        3: "Empty Column",
                        4: "Units - 1st Semester",
                        5: "Estimated Tuition Fees - 1st Semester",
                        6: "Units - 2nd Semester",
                        7: "Estimated Tuition Fees - 2nd Semester",
                        8: "Units - Summer",
                        9: "Summer Tuition Fees"
                    }
                elif len(headers) >= 8:  # SOE format with Acronym (8 cols)
                    program_fee_header_map = {
                        0: "Program",
                        1: "Acronym",
                        2: "Year Level",
                        3: "Empty Column",
                        4: "Units - 1st Semester",
                        5: "Estimated Tuition Fees - 1st Semester",
                        6: "Units - 2nd Semester",
                        7: "Estimated Tuition Fees - 2nd Semester"
                    }
                else:  # Old format (6 columns - no acronym)
                    program_fee_header_map = {
                        0: "Program",
                        1: "Year Level",
                        2: "Empty Column",
                        3: "Units - 1st Semester",
                        4: "Estimated Tuition Fees - 1st Semester",
                        5: "Units - 2nd Semester",
                        6: "Estimated Tuition Fees - 2nd Semester"
                    }
                clean_headers.append(program_fee_header_map.get(i, f"Column {i+1}"))
            else:
                clean_headers.append(f"Column {i+1}")
    
    # Filter out empty data rows
    valid_data = []
    for row in data:
        # Skip rows that are completely empty or only have empty strings
        if any(cell and str(cell).strip() and str(cell).strip().upper() != "NULL" for cell in row):
            valid_data.append(row)
    
    if not valid_data:
        text_parts.append("No data available")
        return text_parts
    
    # Special handling for Program Fees to group by program
    if section_name.lower() == "program fees":
        return format_program_fees_grouped(valid_data, clean_headers)
    
    # Format each data row (for other sections)
    for row_idx, row in enumerate(valid_data):
        if len(row) == 0:
            continue
            
        text_parts.append("")
        text_parts.append(f"Entry {row_idx + 1}:")
        
        for i, (header, value) in enumerate(zip(clean_headers, row)):
            if i < len(row) and value and str(value).strip() and str(value).strip().upper() != "NULL":
                # Determine if this should be formatted as currency
                should_format_as_currency = False
                
                # Check if this is a fee amount
                if section_name.lower() == "tuition fees":
                    # Tuition fees - format monetary values
                    if any(keyword in header.lower() for keyword in ["tuition", "fee"]):
                        should_format_as_currency = True
                elif section_name.lower() in ["school fees", "course fees"]:
                    # School/Course fees - format year columns as currency
                    if any(keyword in header.lower() for keyword in ["year", "sem"]):
                        should_format_as_currency = True
                elif section_name.lower() == "program fees":
                    # Program fees - format estimated tuition as currency
                    if any(keyword in header.lower() for keyword in ["tuition", "summer"]):
                        should_format_as_currency = True
                
                # Try to format numeric values
                try:
                    # Handle NULL values
                    if str(value).strip().upper() == "NULL":
                        text_parts.append(f"  {header}: Not Applicable")
                        continue
                        
                    numeric_value = pd.to_numeric(str(value).replace(',', '').replace('₱', '').replace('$', ''), errors='coerce')
                    if not pd.isna(numeric_value):
                        if should_format_as_currency and numeric_value != 0:
                            text_parts.append(f"  {header}: ₱{numeric_value:,.2f}")
                        elif header.lower() in ["units - 1st sem", "units - 2nd sem", "summer units", "year level"]:
                            # Don't format units or year level as currency
                            text_parts.append(f"  {header}: {int(numeric_value) if numeric_value == int(numeric_value) else numeric_value}")
                        else:
                            text_parts.append(f"  {header}: {int(numeric_value) if numeric_value == int(numeric_value) else numeric_value}")
                    else:
                        text_parts.append(f"  {header}: {value}")
                except:
                    text_parts.append(f"  {header}: {value}")
    
    return text_parts


def format_program_fees_grouped(data: List[List[str]], headers: List[str]) -> List[str]:
    """
    Format program fees data grouped by program with comprehensive year-by-year breakdown.
    """
    text_parts = []
    
    # Group data by program
    programs = {}
    for row in data:
        if len(row) > 0 and row[0] and str(row[0]).strip():
            program_name = str(row[0]).strip()
            if program_name not in programs:
                programs[program_name] = []
            programs[program_name].append(row)
    
    # Format each program's data
    for program_name, program_rows in programs.items():
        text_parts.append("")
        
        print(f"DEBUG: Processing program '{program_name}', headers length: {len(headers)}")
        print(f"DEBUG: Headers: {headers}")
        print(f"DEBUG: First row: {program_rows[0] if program_rows else 'No rows'}")
        
        # Get acronym if available (from first row of this program)
        acronym = ""
        if len(program_rows) > 0 and len(program_rows[0]) > 1 and len(headers) >= 8:
            # Acronym is in column 1 for 8+ column format (both SOE and SON have acronym now)
            acronym = str(program_rows[0][1]).strip() if program_rows[0][1] else ""
        
        if acronym:
            text_parts.append(f"PROGRAM: {program_name.upper()} ({acronym})")
        else:
            text_parts.append(f"PROGRAM: {program_name.upper()}")
        text_parts.append("-" * 30)
        
        # Sort rows by year level
        sorted_rows = []
        for row in program_rows:
            try:
                # Get year level from correct column based on format
                if len(headers) >= 8:  # Format with acronym (both SOE 8 cols and SON 10 cols) - year level is in column 2
                    year_col = 2
                else:  # Old format without acronym - year level is in column 1
                    year_col = 1
                    
                year_level = int(str(row[year_col]).strip()) if len(row) > year_col and str(row[year_col]).strip().isdigit() else 999
                sorted_rows.append((year_level, row))
            except:
                sorted_rows.append((999, row))
        
        sorted_rows.sort(key=lambda x: x[0])
        
        # Format each year
        for year_level, row in sorted_rows:
            if len(row) < 2:
                continue
                
            text_parts.append("")
            # Get year level from correct column based on format
            if len(headers) >= 8:  # Format with acronym (both SOE 8 cols and SON 10 cols) - year level is in column 2
                year_value = row[2] if len(row) > 2 else ""
                print(f"DEBUG: 8+ cols format - year_value from row[2]: '{year_value}', row: {row[:4]}")
            else:  # Old format without acronym - year level is in column 1
                year_value = row[1] if len(row) > 1 else ""
                print(f"DEBUG: <8 cols format - year_value from row[1]: '{year_value}', row: {row[:4]}")
            
            year_display = f"Year {year_value}" if str(year_value).strip().isdigit() else str(year_value).strip()
            print(f"DEBUG: Final year_display: '{year_display}'")
            text_parts.append(f"{year_display}:")
            
            for i, (header, value) in enumerate(zip(headers, row)):
                if i < len(row) and value and str(value).strip() and str(value).strip().upper() != "NULL":
                    # Skip program name, acronym, year level, and empty column
                    # Adjust based on format: 8+ cols (0,1,2,3), <8 cols (0,1,2)
                    if len(headers) >= 8:  # Format with acronym (both SOE and SON)
                        if i == 0 or i == 1 or i == 2 or i == 3:  # Skip program, acronym, year level, empty column
                            continue
                    else:  # Old format without acronym
                        if i == 0 or i == 1 or i == 2:  # Skip program name, year level, and empty column
                            continue
                    
                    # Skip empty values
                    if not value or str(value).strip() == "":
                        continue
                        
                    try:
                        numeric_value = pd.to_numeric(str(value).replace(',', '').replace('₱', '').replace('$', ''), errors='coerce')
                        if not pd.isna(numeric_value):
                            # Format tuition as currency, units as numbers
                            if "tuition" in header.lower() or "summer" in header.lower():
                                text_parts.append(f"  {header}: ₱{numeric_value:,.2f}")
                            elif "units" in header.lower():
                                text_parts.append(f"  {header}: {int(numeric_value)} units")
                            else:
                                text_parts.append(f"  {header}: {int(numeric_value) if numeric_value == int(numeric_value) else numeric_value}")
                        else:
                            text_parts.append(f"  {header}: {value}")
                    except:
                        text_parts.append(f"  {header}: {value}")
        
        # Add program summary
        text_parts.append("")
        if acronym:
            text_parts.append(f"COMPLETE {program_name.upper()} ({acronym}) TUITION BREAKDOWN:")
            text_parts.append("All year levels shown above with semester-wise estimated tuition fees.")
            text_parts.append(f"This covers the complete tuition structure for {program_name} ({acronym}) from 1st year to final year.")
            text_parts.append(f"Keywords: {program_name}, {acronym}, tuition, fees, semester, units")
        else:
            text_parts.append(f"COMPLETE {program_name.upper()} TUITION BREAKDOWN:")
            text_parts.append("All year levels shown above with semester-wise estimated tuition fees.")
            text_parts.append(f"This covers the complete tuition structure for {program_name} from 1st year to final year.")
    
    return text_parts


def get_supported_extensions() -> List[str]:
    """Get list of supported file extensions."""
    return ['.csv', '.xlsx', '.xls']


def is_excel_csv_file(file_name: str) -> bool:
    """Check if file is a supported Excel or CSV file."""
    return any(file_name.lower().endswith(ext) for ext in get_supported_extensions())

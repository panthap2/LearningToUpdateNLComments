import re

def extract_method_name(code_block):
    """Extracts method name from a method, represented as a list of code lines. NOTE: Will need to subtokenize."""
    i = 0
    while i < len(code_block):
        line = code_block[i].strip()
        if len(line.strip()) == 0:
            i += 1
            continue
        if line[0] == '@' and ' ' not in line:
            i += 1
            continue
        if '//' in line or '*' in line:
            i += 1
            continue
        else:
            break
    
    try:
        method_components = line.strip().split('(')[0].split(' ')
        method_components = [m for m in method_components if len(m) > 0]
        method_name = method_components[-1].strip()
    except:
        method_name = ''

    return method_name

def extract_return_type(code_block):
    """Extracts return type from a method, represented as a list of code lines. NOTE: Will need to subtokenize."""
    i = 0
    while i < len(code_block):
        line = code_block[i].strip()
        if len(line.strip()) == 0:
            i += 1
            continue
        if line[0] == '@':
            i += 1
            continue
        if '//' in line or '*' in line:
            i += 1
            continue
        else:
            break
    
    before_method_name_tokens = line.split('(')[0].split(' ')[:-1]
    return_type_tokens = []
    for tok in before_method_name_tokens:
        if tok not in ['private', 'protected', 'public', 'final', 'static']:
            return_type_tokens.append(tok)
    return ' '.join(return_type_tokens)

def extract_return_statements(code_block):
    """Extracts return statement from a method, represented as a list of code lines. NOTE: Will need to subtokenize."""
    cleaned_lines = []
    for l in code_block:
        cleaned_l = strip_comment(l)
        if len(cleaned_l) > 0:
            cleaned_lines.append(cleaned_l)
 
    combined_block = ' '.join(cleaned_lines)
    if 'return' not in combined_block:
        return []
    indices = [m.start() for m in re.finditer('return ', combined_block)]
    return_statements = []
    for idx in indices:
        s_idx = idx + len('return ')
        e_idx = s_idx + combined_block[s_idx:].index(';')
        statement = combined_block[s_idx:e_idx].strip()
        if len(statement) > 0:
            return_statements.append(statement)

    return return_statements

def strip_comment(s):
    """Checks whether a single line follows the structure of a comment."""
    new_s = re.sub(r'\"(.+?)\"', '', s)
    matched_obj = re.findall("(?:/\\*(?:[^*]|(?:\\*+[^*/]))*\\*+/)|(?://.*)", new_s)
    url_match = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', new_s)
    file_match = re.findall('^(.*/)?(?:$|(.+?)(?:(\.[^.]*$)|$))', new_s)

    if matched_obj and not url_match:
        for m in matched_obj:
            s = s.replace(m, ' ')
    return s.strip()
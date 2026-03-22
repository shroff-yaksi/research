import os

docs_dir = 'docs'
output_file = 'docs/THE_ULTIMATE_MASTER_GUIDE.md'
exclude_files = ['README.md', 'JOURNEY.md', 'MASTER_COMPREHENSIVE_GUIDE.md', 'THE_ULTIMATE_MASTER_GUIDE.md']

def merge_docs():
    merged_content = "# THE ULTIMATE MASTER GUIDE TO RE-TabSyn\n\n"
    merged_content += "> This document is a massive compilation of all explanations, lectures, audits, and guides created for the RE-TabSyn project, ranging from complete layman analogies to advanced graduate-level mathematical breakdowns.\n\n"
    merged_content += "---\n\n"
    
    files_to_merge = []
    for f in os.listdir(docs_dir):
        if f.endswith('.md') and f not in exclude_files:
            files_to_merge.append(f)
            
    files_to_merge.sort()
    
    for f in files_to_merge:
        file_path = os.path.join(docs_dir, f)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            merged_content += f"<!-- START OF {f} -->\n"
            merged_content += f"# SECTION: {f.replace('.md', '').replace('_', ' ').upper()}\n\n"
            merged_content += content
            merged_content += f"\n\n<!-- END OF {f} -->\n\n"
            merged_content += "---\n\n"
            
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(merged_content)
        
    print(f"Merged files: {files_to_merge}")
    print(f"Created {output_file}")

if __name__ == "__main__":
    merge_docs()
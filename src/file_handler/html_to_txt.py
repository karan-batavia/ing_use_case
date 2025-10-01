import os
import sys
import argparse
import glob
from pathlib import Path
from bs4 import BeautifulSoup

class HTMLToTextConverter:
    def __init__(self, clean_whitespace=True, remove_empty_lines=True, preserve_links=False):
        """
        Initialize the converter with formatting options.
        
        Args:
            clean_whitespace (bool): Remove extra whitespace and normalize spacing
            remove_empty_lines (bool): Remove empty lines from output
            preserve_links (bool): Preserve URLs from links in the text
        """
        self.clean_whitespace = clean_whitespace
        self.remove_empty_lines = remove_empty_lines
        self.preserve_links = preserve_links

    def convert_html_to_txt(self, input_html_file, output_txt_file=None):
        """
        Convert a single HTML file to text.
        
        Args:
            input_html_file (str): Path to the input HTML file
            output_txt_file (str, optional): Path to the output TXT file. 
                                           If None, auto-generates based on input filename.
        
        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            # Auto-generate output filename if not provided
            if output_txt_file is None:
                input_path = Path(input_html_file)
                output_txt_file = input_path.with_suffix('.txt')

            # Open and read the HTML file
            with open(input_html_file, 'r', encoding='utf-8') as html_file:
                html_content = html_file.read()

            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()

            # Extract plain text from the HTML
            if self.preserve_links:
                # Replace links with "text (URL)" format
                for link in soup.find_all('a', href=True):
                    link_text = link.get_text()
                    link_url = link['href']
                    if link_text and link_url:
                        link.string = f"{link_text} ({link_url})"

            plain_text = soup.get_text()

            # Apply formatting options
            if self.clean_whitespace:
                # Normalize whitespace
                lines = plain_text.split('\n')
                cleaned_lines = []
                for line in lines:
                    cleaned_line = ' '.join(line.split())  # Remove extra whitespace
                    cleaned_lines.append(cleaned_line)
                plain_text = '\n'.join(cleaned_lines)

            if self.remove_empty_lines:
                # Remove empty lines
                lines = plain_text.split('\n')
                non_empty_lines = [line for line in lines if line.strip()]
                plain_text = '\n'.join(non_empty_lines)

            # Write the plain text to the output TXT file
            with open(output_txt_file, 'w', encoding='utf-8') as txt_file:
                txt_file.write(plain_text)

            print(f"✓ Conversion successful! Text saved to '{output_txt_file}'.")
            return True

        except FileNotFoundError:
            print(f"✗ Error: The file '{input_html_file}' was not found.")
            return False
        except Exception as e:
            print(f"✗ An error occurred: {e}")
            return False

    def convert_multiple_files(self, input_pattern, output_dir=None):
        """
        Convert multiple HTML files matching a pattern.
        
        Args:
            input_pattern (str): Glob pattern for input HTML files
            output_dir (str, optional): Directory for output files. If None, uses same directory as input.
        
        Returns:
            int: Number of files successfully converted
        """
        html_files = glob.glob(input_pattern)
        if not html_files:
            print(f"No files found matching pattern: {input_pattern}")
            return 0

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        successful_conversions = 0
        total_files = len(html_files)
        
        print(f"Found {total_files} HTML file(s) to convert...")
        
        for i, html_file in enumerate(html_files, 1):
            print(f"[{i}/{total_files}] Processing: {html_file}")
            
            if output_dir:
                output_file = os.path.join(output_dir, Path(html_file).stem + '.txt')
            else:
                output_file = None  # Auto-generate in same directory
            
            if self.convert_html_to_txt(html_file, output_file):
                successful_conversions += 1

        print(f"\nConversion complete! {successful_conversions}/{total_files} files converted successfully.")
        return successful_conversions

def interactive_mode():
    """Run the converter in interactive mode."""
    print("=== HTML to Text Converter - Interactive Mode ===")
    
    while True:
        print("\nOptions:")
        print("1. Convert single file")
        print("2. Convert multiple files (batch)")
        print("3. Exit")
        
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == '1':
            input_file = input("Enter HTML file path: ").strip()
            if not os.path.exists(input_file):
                print(f"File not found: {input_file}")
                continue
            
            output_file = input("Enter output file path (press Enter for auto-generation): ").strip()
            output_file = output_file if output_file else None
            
            # Get formatting options
            clean_ws = input("Clean whitespace? (y/n, default=y): ").strip().lower() in ['', 'y', 'yes']
            remove_empty = input("Remove empty lines? (y/n, default=y): ").strip().lower() in ['', 'y', 'yes']
            preserve_links = input("Preserve links with URLs? (y/n, default=n): ").strip().lower() in ['y', 'yes']
            
            converter = HTMLToTextConverter(clean_ws, remove_empty, preserve_links)
            converter.convert_html_to_txt(input_file, output_file)
            
        elif choice == '2':
            pattern = input("Enter file pattern (e.g., *.html, files/*.html): ").strip()
            output_dir = input("Enter output directory (press Enter for same directory): ").strip()
            output_dir = output_dir if output_dir else None
            
            # Get formatting options
            clean_ws = input("Clean whitespace? (y/n, default=y): ").strip().lower() in ['', 'y', 'yes']
            remove_empty = input("Remove empty lines? (y/n, default=y): ").strip().lower() in ['', 'y', 'yes']
            preserve_links = input("Preserve links with URLs? (y/n, default=n): ").strip().lower() in ['y', 'yes']
            
            converter = HTMLToTextConverter(clean_ws, remove_empty, preserve_links)
            converter.convert_multiple_files(pattern, output_dir)
            
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

def main():
    """Main function with command-line argument support."""
    parser = argparse.ArgumentParser(
        description="Convert HTML files to plain text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python html_to_txt.py file.html                    # Convert single file
  python html_to_txt.py file.html -o output.txt      # Specify output file
  python html_to_txt.py "*.html" --batch             # Convert all HTML files
  python html_to_txt.py "docs/*.html" --batch -d txt # Batch convert to 'txt' directory
  python html_to_txt.py --interactive                # Run in interactive mode
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input HTML file or pattern')
    parser.add_argument('-o', '--output', help='Output text file (for single file conversion)')
    parser.add_argument('-d', '--output-dir', help='Output directory (for batch conversion)')
    parser.add_argument('--batch', action='store_true', help='Batch mode: treat input as file pattern')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--no-clean', action='store_true', help='Don\'t clean whitespace')
    parser.add_argument('--keep-empty-lines', action='store_true', help='Keep empty lines')
    parser.add_argument('--preserve-links', action='store_true', help='Preserve URLs from links')
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        interactive_mode()
        return
    
    # Check if input is provided
    if not args.input:
        print("No input specified. Use --interactive for interactive mode or provide an input file/pattern.")
        parser.print_help()
        return
    
    # Create converter with options
    converter = HTMLToTextConverter(
        clean_whitespace=not args.no_clean,
        remove_empty_lines=not args.keep_empty_lines,
        preserve_links=args.preserve_links
    )
    
    # Batch mode
    if args.batch:
        converter.convert_multiple_files(args.input, args.output_dir)
    else:
        # Single file mode
        if not os.path.exists(args.input):
            print(f"Error: File '{args.input}' not found.")
            sys.exit(1)
        
        success = converter.convert_html_to_txt(args.input, args.output)
        sys.exit(0 if success else 1)

# Example usage and backward compatibility
if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run with default behavior for backward compatibility
        print("Running in legacy mode...")
        converter = HTMLToTextConverter()
        input_html = "example.html"  # Replace with your HTML file path
        output_txt = "output.txt"   # Replace with your desired TXT file path
        converter.convert_html_to_txt(input_html, output_txt)
    else:
        # Arguments provided, use new dynamic interface
        main()
import os
import re
import glob
def find_top_accuracy_lines(
    directory: str = ".",
    top_n: int = 10
):
    pattern = re.compile(
        r'\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]\s+'
        r'\[\w+\]:\s+'
        r'NM@R1:\s*(\d+\.\d+)%\s+'
        r'BG@R1:\s*(\d+\.\d+)%\s+'
        r'CL@R1:\s*(\d+\.\d+)%'
    )

    results = []

    log_files = glob.glob(os.path.join(directory, "**", "*.log"), recursive=True)

    if not log_files:
        print(f"No log files in '{directory}'")
        return []

    print(f"Found {len(log_files)} logs")

    for filepath in log_files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for lineno, line in enumerate(f, start=1):
                    match = pattern.search(line)
                    if match:
                        try:
                            nm = float(match.group(1))
                            bg = float(match.group(2))
                            cl = float(match.group(3))
                            total = nm + bg + cl
                            rel_path = os.path.relpath(filepath, directory)
                            results.append((total, rel_path, lineno, line.strip()))
                        except (ValueError, IndexError) as e:
                            print(f"Error in '{filepath}':{lineno}: {e}")
        except Exception as e:
            print(f"Read '{filepath}' failed: {e}")

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_n], len(results)

def main():
    top_results, total_matches = find_top_accuracy_lines(".", top_n=10)

    print(f"Total: {total_matches}")

    if not top_results:
        return

    print()
    for idx, (total, filename, lineno, content) in enumerate(top_results, 1):
        print(f"{idx:<2}{total:>8.2f}% {filename}:{lineno}")
        print(f"{content}")
        print()

if __name__ == "__main__":
    main()

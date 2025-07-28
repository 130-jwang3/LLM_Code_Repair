import os
import subprocess
from datetime import datetime

def extract_bug_reports(repo_dir, output_dir, max_per_call=900, start_year=2010, end_year=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Move into the GitHub repo directory
    os.chdir(repo_dir)
    repo_name = os.path.basename(os.getcwd())

    if end_year is None:
        end_year = datetime.today().year

    states = ['open', 'closed']
    for state in states:
        for year in range(start_year, end_year + 1):
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            search_query = f"created:>={start_date} created:<={end_date}"

            output_filename = f"{repo_name}_issues_{state}_{year}.json"
            output_path = os.path.join(output_dir, output_filename)

            print(f"📦 Fetching {state} issues for {year}...")

            cmd = [
                "gh", "issue", "list",
                "--state", state,
                "--search", search_query,
                "--json", "author,body,comments,createdAt,number,state,title,url",
                "--limit", str(max_per_call)
            ]

            with open(output_path, "w", encoding="utf-8") as f:
                subprocess.run(cmd, stdout=f)

    print(f"\n✅ Bug report extraction complete! Files saved to {output_dir}")

if __name__ == "__main__":
    extract_bug_reports("data/raw/PyGithub", "data/bug_reports/")

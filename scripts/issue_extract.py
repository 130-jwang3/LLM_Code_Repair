import os
import subprocess
import time
from datetime import datetime

def extract_issue_reports(repo_dir, output_dir, max_per_call=900, start_year=2010, end_year=None, delay_per_request=30):
    # Use absolute path for output_dir
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Change into the repo directory
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

            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    subprocess.run(cmd, stdout=f, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to fetch issues for {year} ({state}): {e}")
            except Exception as ex:
                print(f"❌ Error writing to {output_path}: {ex}")

            print(f"⏳ Waiting {delay_per_request} seconds to avoid rate limits...\n")
            time.sleep(delay_per_request)

    print(f"\n✅ Bug report extraction complete! Files saved to {output_dir}")

"""Check the Bug Investigator on realistic reports."""
from src.agents.bug_investigator import investigate

REPORTS = [
    "My form submits but nothing appears in Form Records",
    "I get a white screen when I open the form editor",
]


def main():
    for report in REPORTS:
        print(f"\n{'=' * 70}\nReport: {report}\n")
        print(investigate(report))


if __name__ == "__main__":
    main()

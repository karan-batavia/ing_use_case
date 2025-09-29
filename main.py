import argparse
from src.prompt_scrubber import PromptScrubber


def main():
    """
    Scrub prompts against classification files
    """

    parser = argparse.ArgumentParser(
        description="Scrub prompts for classified content",
    )

    parser.add_argument(
        "--prompt", "-p", type=str, required=True, help="Add a prompt to sanitize"
    )

    args = parser.parse_args()

    print(f"Prompt: {args.prompt}")

    # Initialize scrubber
    scrubber = PromptScrubber()

    # Check for matches
    matches = scrubber.scrub(args.prompt)

    if matches:
        print("\nFound matches:")
        for filename, found_values in matches.items():
            print(f"  File '{filename}':")
            for value in found_values:
                print(f"    - {value}")
    else:
        print("\nNo matches found in classification files.")

    # Show scrubbed prompt
    scrubbed_prompt = scrubber.scrub_prompt(args.prompt)
    if scrubbed_prompt != args.prompt:
        print(f"\nScrubbed prompt:")
        print(f"  {scrubbed_prompt}")
    else:
        print(f"\nPrompt contains no classified data.")


if __name__ == "__main__":
    main()

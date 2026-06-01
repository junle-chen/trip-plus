"""LLM-backed report conversion runner."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from agent.call_llm import call_llm, load_model_config
from agent.prompts import get_format_convert_prompt

from .entities import _load_exact_entity_names_for_sample
from .json_extract import extract_json_from_response, extract_last_plan_block
from .parser import (
    _has_canonical_plan_shape,
    _is_plan_block,
    _postprocess_converted_plan,
    deterministic_convert_plan,
)


class ConversionParseError(ValueError):
    """Raised when the model responded but the payload is not parseable JSON."""

    pass


def process_single_report(
    report_file: Path,
    output_dir: Path,
    conversion_model: str,
    format_prompt: str,
    print_lock: Lock,
    max_retries: int = 30,
    database_dir: Optional[Path] = None,
    language: str = "en",
    query_file: Optional[Path] = None,
) -> Dict:
    """
    Process a single report file and convert to JSON

    Args:
        report_file: Input report file path
        output_dir: Output directory for converted JSON
        format_prompt: Format conversion prompt
        print_lock: Thread-safe print lock
        max_retries: Maximum number of retries for JSON parsing errors

    Returns:
        Processing result dictionary
    """
    sample_id = None

    # Extract sample_id from report filenames such as id_0.txt or single_0000.txt.
    filename = report_file.name
    match = re.match(r"id_(.+)\.txt", filename)
    if match:
        sample_id = match.group(1)
    else:
        # Try without id_ prefix
        match = re.match(r"(\d+)_final_answer\.txt", filename)
        if match:
            sample_id = match.group(1)
        else:
            sample_id = report_file.stem.replace("_final_answer", "")

    exact_entity_names = _load_exact_entity_names_for_sample(
        sample_id=sample_id,
        database_dir=database_dir,
        language=language,
        query_file=query_file,
    )

    # Retry loop for JSON parsing errors
    for attempt in range(max_retries + 1):
        content = ""
        try:
            if attempt == 0:
                with print_lock:
                    print(f"\n{'=' * 80}")
                    print(f"🚀 [Thread Started] Processing Sample ID: {sample_id}")
                    print(f"   Input File: {report_file.name}")
                    print(f"{'=' * 80}")
            else:
                with print_lock:
                    print(
                        f"🔄 Sample {sample_id} JSON parsing failed, retry attempt {attempt}..."
                    )

            # Read raw text
            raw_text = report_file.read_text(encoding="utf-8")
            raw_text = extract_last_plan_block(raw_text)

            if _is_plan_block(raw_text) and not _has_canonical_plan_shape(raw_text):
                malformed = {
                    "status": "malformed",
                    "malformed_explanation": (
                        "The plan block does not contain the required canonical "
                        "Day/activity-row format, so it is not an executable itinerary."
                    ),
                }
                output_file = output_dir / f"id_{sample_id}_converted.json"
                output_file.write_text(
                    json.dumps(malformed, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                with print_lock:
                    print(f"⚠️ Sample {sample_id} marked malformed during conversion")
                    print(f"   Output File: {output_file.name}\n")
                return {
                    "success": True,
                    "sample_id": sample_id,
                    "input_file": str(report_file),
                    "output_file": str(output_file),
                    "conversion_method": "malformed_guard",
                }

            deterministic = deterministic_convert_plan(
                raw_text, exact_names=exact_entity_names
            )
            if deterministic is not None:
                deterministic = _postprocess_converted_plan(
                    deterministic, raw_text, exact_entity_names
                )
                output_file = output_dir / f"id_{sample_id}_converted.json"
                output_file.write_text(
                    json.dumps(deterministic, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                with print_lock:
                    print(
                        f"✅ Sample {sample_id} conversion completed deterministically"
                    )
                    print(f"   Output File: {output_file.name}\n")
                return {
                    "success": True,
                    "sample_id": sample_id,
                    "input_file": str(report_file),
                    "output_file": str(output_file),
                    "conversion_method": "deterministic",
                }

            # Construct messages
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"{format_prompt}\n\n"
                        "IMPORTANT: Output only final JSON. "
                        "Do not output thinking process or analysis. "
                        "Copy entity names exactly from the input plan. "
                        "Do not add, remove, or move spaces inside hotel, attraction, restaurant, "
                        "station, or airport names."
                    ),
                },
                {"role": "user", "content": raw_text},
            ]

            # Call LLM for conversion
            resp = call_llm(
                config_name=conversion_model,
                messages=messages,
                request_overrides={"max_tokens": 10240},
            )

            content = resp.choices[0].message.content or ""

            # Extract JSON
            json_payload = extract_json_from_response(content)
            if not json_payload:
                # If no tags and can't parse directly, fail this attempt
                try:
                    json.loads(content)
                    json_payload = content
                except Exception as e:
                    raise ConversionParseError(
                        f"LLM did not return content with <JSON> tags, and direct parsing failed: {e}"
                    )

            # Validate JSON (this is the key step for retry)
            parsed = json.loads(json_payload)
            parsed = _postprocess_converted_plan(parsed, raw_text, exact_entity_names)

            # If successful, save and exit loop
            output_file = output_dir / f"id_{sample_id}_converted.json"
            output_file.write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            with print_lock:
                print(
                    f"✅ Sample {sample_id} conversion completed (attempt {attempt + 1})"
                )
                print(f"   Output File: {output_file.name}\n")

            return {
                "success": True,
                "sample_id": sample_id,
                "input_file": str(report_file),
                "output_file": str(output_file),
            }

        except (json.JSONDecodeError, ConversionParseError) as e:
            # Only retry when the model responded but the payload was not valid JSON.
            # If last attempt, record failure and exit
            if attempt >= max_retries:
                with print_lock:
                    print(
                        f"❌ Sample {sample_id} failed after {max_retries + 1} attempts: {e}\n"
                    )

                return {
                    "success": False,
                    "sample_id": sample_id,
                    "input_file": str(report_file),
                    "error": str(e),
                }
            # If not last attempt, continue to next retry
            preview = (content or "").replace("\n", " ")[:300]
            with print_lock:
                print(f"   ↳ Parse error: {e}")
                if preview:
                    print(f"   ↳ Raw response preview: {preview}")
            time.sleep(1)  # Brief delay to avoid rapid requests

        except Exception as e:
            # Transport / exhausted retry errors are already handled inside call_llm.
            preview = (content or "").replace("\n", " ")[:300]
            with print_lock:
                print(f"   ↳ Unexpected error: {e}")
                if preview:
                    print(f"   ↳ Raw response preview: {preview}")
            return {
                "success": False,
                "sample_id": sample_id,
                "input_file": str(report_file),
                "error": str(e),
            }


def convert_reports(
    result_dir: Path,
    language: str = "en",
    conversion_model: str = "gemma-4-31b-vllm",
    workers: int = 10,
    skip_existing: bool = False,
    database_dir: Optional[Path] = None,
    query_file: Optional[Path] = None,
) -> Dict:
    """
    Convert multiple report files to JSON format

    Note: This function uses the specified conversion model from models_config.json

    Args:
        result_dir: Result directory containing 'reports' subdirectory
        language: Prompt language. This release supports English only.
        conversion_model: Model config name used for report conversion
        workers: Number of concurrent workers
        skip_existing: Skip files that already have output
        database_dir: Optional database root used for exact entity-name cleanup
        query_file: Optional query file used to map multi-turn IDs to sample DB IDs

    Returns:
        dict: {'total': int, 'converted': int, 'skipped': int, 'results': list}
    """
    # Set reports_dir and output_dir based on result_dir
    reports_dir = result_dir / "reports"
    output_dir = result_dir / "converted_plans"
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = load_model_config(conversion_model)
    actual_model_name = model_config.get("model_name", conversion_model)

    # Get format prompt
    format_prompt = get_format_convert_prompt(language)

    # Find all report files. Generated benchmarks may use string sample IDs
    # such as single_0000.txt.
    report_files = sorted(reports_dir.glob("*.txt"))

    if not report_files:
        print(f"⚠️  No report files found in {reports_dir}")
        return {
            "total": 0,
            "converted": 0,
            "skipped": 0,
            "success": 0,
            "failed": 0,
            "results": [],
        }

    # Track original count and filtered files
    original_count = len(report_files)
    skipped_count = 0

    # Filter out existing files if skip_existing is True
    if skip_existing:
        filtered_files = []
        for report_file in report_files:
            # Extract sample_id
            match = re.match(r"id_(.+)\.txt", report_file.name)
            if match:
                sample_id = match.group(1)
            else:
                sample_id = report_file.stem.replace("_final_answer", "")

            # Check if output file already exists
            output_file = output_dir / f"id_{sample_id}_converted.json"
            if not output_file.exists():
                filtered_files.append(report_file)

        skipped_count = original_count - len(filtered_files)
        report_files = filtered_files

        if skipped_count > 0:
            print(f"⏭️  Skipped {skipped_count} existing files")

        if not report_files:
            print(f"✅ All files already converted, nothing to process")
            return {
                "total": original_count,
                "converted": 0,
                "skipped": skipped_count,
                "success": 0,
                "failed": 0,
                "results": [],
            }

    print(f"\n{'=' * 80}")
    print(f"📊 Found {len(report_files)} report files to convert")
    print(f"🚀 Using {workers} concurrent workers")
    print(f"📂 Input Directory: {reports_dir}")
    print(f"📂 Output Directory: {output_dir}")
    print(f"🌐 Language: {language}")
    print(f"🤖 Conversion Model: {actual_model_name}")
    print(f"{'=' * 80}\n")

    # Create print lock
    print_lock = Lock()

    # Record start time
    start_time = time.time()

    # Use thread pool for parallel processing
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        future_to_file = {}
        for report_file in report_files:
            future = executor.submit(
                process_single_report,
                report_file,
                output_dir,
                conversion_model,
                format_prompt,
                print_lock,
                30,
                database_dir,
                language,
                query_file,
            )
            future_to_file[future] = report_file

        # Collect results (in completion order)
        for future in as_completed(future_to_file):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                report_file = future_to_file[future]
                with print_lock:
                    print(
                        f"❌ File {report_file.name} encountered uncaught exception: {e}\n"
                    )
                results.append(
                    {"success": False, "sample_id": report_file.name, "error": str(e)}
                )

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Statistics
    success_count = sum(1 for r in results if r["success"])
    failed_count = len(results) - success_count

    print(f"\n{'=' * 80}")
    print(f"✅ All report conversions completed!")
    print(f"{'=' * 80}")
    print(f"📊 Statistics:")
    print(f"   - Total Files: {len(report_files)}")
    print(f"   - Success: {success_count}")
    print(f"   - Failed: {failed_count}")
    print(f"   - Total Time: {elapsed_time:.2f} seconds")
    print(f"   - Average Time: {elapsed_time / len(report_files):.2f} seconds/file")
    print(f"   - Output Directory: {output_dir}")
    print(f"{'=' * 80}\n")

    return {
        "total": original_count,
        "converted": success_count,
        "skipped": skipped_count,
        "success": success_count,
        "failed": failed_count,
        "results": results,
        "elapsed_time": elapsed_time,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert report files to JSON format")
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Result directory containing reports/ subdirectory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-4-31b-vllm",
        help="Model configuration name from models_config.json",
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="Number of concurrent workers"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have output",
    )
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=None,
        help="Optional database root for exact entity-name cleanup",
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        default=None,
        help="Optional query JSON file for multi-turn sample DB ID mapping",
    )

    args = parser.parse_args()
    result = convert_reports(
        result_dir=args.result_dir,
        language="en",
        conversion_model=args.model,
        workers=args.workers,
        skip_existing=args.skip_existing,
        database_dir=args.database_dir,
        query_file=args.query_file,
    )
    print(f"Conversion completed: {result['success']}/{result['total']} succeeded")


if __name__ == "__main__":
    main()

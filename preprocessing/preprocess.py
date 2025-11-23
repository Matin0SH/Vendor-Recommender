"""
Preprocess vendor data from all_results.json for embedding.
Combines relevant text fields into a single searchable string per vendor.
Handles failed extractions with fallback to raw vendor fields.
"""

import json
from pathlib import Path


def load_vendors(json_path: str) -> list[dict]:
    """Load vendor data from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def combine_text_fields(vendor: dict) -> str:
    """
    Combine relevant text fields into a single string for embedding.
    Prioritizes service-related fields for job matching.
    Falls back to raw vendor fields if extraction failed.
    """
    extracted = vendor.get("extracted") or {}
    is_success = vendor.get("status") == "success"
    record_index = vendor.get("index")
    vendor_name = vendor.get("vendor", "")

    parts = []

    if record_index is not None:
        parts.append(f"Record index: {record_index}")

    # Company identification - try extracted first, fall back to raw
    company_name = extracted.get("company_name") or vendor.get("company_name", "")
    trading_name = extracted.get("trading_name", "")
    if company_name:
        parts.append(f"Company: {company_name}")
    if trading_name:
        parts.append(f"Also known as: {trading_name}")
    if vendor_name and vendor_name != company_name:
        parts.append(f"Vendor (source): {vendor_name}")

    # For successful extractions, use rich extracted data
    if is_success:
        # Services - MOST IMPORTANT for matching
        services = extracted.get("services", "")
        if services:
            parts.append(f"Services: {services}")

        # Products
        products = extracted.get("products", "")
        if products:
            parts.append(f"Products: {products}")

        # Industry
        industry = extracted.get("industry", "")
        if industry:
            parts.append(f"Industry: {industry}")

        # About/Description
        about = extracted.get("about", "")
        if about:
            parts.append(f"About: {about}")

        # SIC codes (business classification)
        sic_codes = extracted.get("sic_codes", "")
        if sic_codes:
            parts.append(f"SIC Codes: {sic_codes}")

        # Location
        city = extracted.get("city", "")
        country = extracted.get("country", "")
        if city or country:
            location = ", ".join(filter(None, [city, country]))
            parts.append(f"Location: {location}")

        # Certifications (quality indicators)
        certifications = extracted.get("certifications", "")
        if certifications:
            parts.append(f"Certifications: {certifications}")

    else:
        # Add note that extraction failed
        parts.append("Note: Limited data available (extraction incomplete)")

    # Contact and operational details (included for completeness)
    address = extracted.get("address", "") or vendor.get("known_address", "")
    if address:
        parts.append(f"Address: {address}")

    phone = extracted.get("phone", "")
    if phone:
        parts.append(f"Phone: {phone}")

    email = extracted.get("email", "")
    if email:
        parts.append(f"Email: {email}")

    website = extracted.get("website", "")
    if website:
        parts.append(f"Website: {website}")

    employees = extracted.get("employees")
    if employees not in (None, ""):
        parts.append(f"Employees: {employees}")

    confidence = extracted.get("confidence")
    if confidence not in (None, ""):
        parts.append(f"Extraction confidence: {confidence}")

    status = vendor.get("status")
    if status:
        parts.append(f"Extraction status: {status}")

    return "\n".join(parts)


def preprocess_vendors(json_path: str) -> list[dict]:
    """
    Preprocess all vendors for embedding.
    Includes vendors with failed extractions using fallback data.

    Returns list of dicts with:
        - id: unique identifier (index)
        - text: combined text for embedding
        - metadata: vendor info for display
    """
    vendors = load_vendors(json_path)
    processed = []
    success_count = 0
    fallback_count = 0

    for vendor in vendors:
        extracted = vendor.get("extracted") or {}
        is_success = vendor.get("status") == "success"

        # Generate combined text (handles both success and fallback)
        text = combine_text_fields(vendor)

        # Skip if we have no meaningful text
        if len(text.strip()) < 20:
            print(f"  Skipping vendor {vendor.get('index')}: insufficient data")
            continue

        if is_success:
            success_count += 1
        else:
            fallback_count += 1

        processed.append({
            "id": str(vendor.get("index", len(processed))),
            "text": text,
            "metadata": {
                "index": vendor.get("index"),
                "vendor": vendor.get("vendor"),
                "company_name": extracted.get("company_name") or vendor.get("company_name"),
                "trading_name": extracted.get("trading_name"),
                "services": extracted.get("services"),
                "products": extracted.get("products"),
                "industry": extracted.get("industry"),
                "about": extracted.get("about"),
                "city": extracted.get("city"),
                "country": extracted.get("country"),
                "address": extracted.get("address") or vendor.get("known_address"),
                "phone": extracted.get("phone"),
                "email": extracted.get("email"),
                "website": extracted.get("website"),
                "employees": extracted.get("employees"),
                "certifications": extracted.get("certifications"),
                "confidence": extracted.get("confidence"),
                "extraction_status": vendor.get("status"),
            }
        })

    print(f"  Processed: {success_count} successful, {fallback_count} with fallback data")
    return processed


def save_processed(processed: list[dict], output_path: str):
    """Save processed vendors to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(processed)} processed vendors to {output_path}")

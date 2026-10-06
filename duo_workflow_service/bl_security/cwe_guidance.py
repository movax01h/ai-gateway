"""Fixed, per-CWE text the report writer puts on BL findings: how to fix it, and its OWASP Top 10 categories.

One entry per CWE in ``bl_report.IN_SCOPE_CWES``; a test keeps the two in step. The text is generic per weakness
class, like a SAST rule's, so it is written once and reviewed here rather than generated per finding.
"""

#: How to fix each weakness class: short, plain and actionable, for the report's ``solution``.
SOLUTIONS = {
    "200": (
        "Return only the fields the caller is allowed to see. "
        "Filter the query or the response by the caller's permissions, and leave sensitive fields out by default."
    ),
    "284": (
        "Add a server-side access check to this action that matches who should be able to use it. "
        "Enforce it in the handler or a shared middleware, not only in the UI."
    ),
    "285": (
        "Check the caller's permission for this action on this resource before doing it. "
        "Use the same policy check as the other handlers for this resource, and return 403 when it fails."
    ),
    "287": (
        "Require a valid session or token before this action runs. "
        "Take the caller's identity from the authenticated session, never from a value the client sends."
    ),
    "362": (
        "Make the check and the change one atomic step. "
        "For example, use a lock, a unique constraint or a conditional update, "
        "so two concurrent requests cannot both pass the check."
    ),
    "367": (
        "Do not act on a value checked earlier without making sure it has not changed. "
        "Re-check it inside the same transaction or lock, or use a conditional update that checks and changes together."
    ),
    "459": (
        "When access is removed or a record is deleted, also remove everything that still grants access through it: "
        "sessions, tokens, shares, memberships and cached permissions."
    ),
    "639": (
        "Check that the caller is allowed to access the specific record before reading or changing it, "
        "for example by scoping the lookup to the caller's user, team or tenant, "
        "and return 403 or 404 when it doesn't match."
    ),
    "840": (
        "Enforce the business rule on the server, on every path that changes this state. "
        "Check the current state and the allowed transition before applying the change, "
        "and reject requests that skip a step."
    ),
    "862": (
        "Add an authorization check to this handler before it reads or changes data, "
        "such as the check used by similar protected handlers, if there are any. "
        "Return 403 when the caller is not allowed."
    ),
    "863": (
        "Make the authorization check cover every condition the protected action requires, "
        "such as role, membership state and ownership, "
        "and deny the request when any of them is missing."
    ),
    "915": (
        "Accept only the fields the caller is allowed to set. "
        "Use an allowlist of assignable fields for this action, and ignore or reject sensitive ones "
        "such as owner, role or price."
    ),
}

#: OWASP Top 10 2021 category of each CWE that OWASP maps to one, as ``(id, name, url)``. CWE-362, CWE-367 and
#: CWE-459 are in no Top 10 2021 category, so they get no 2021 identifier.
_OWASP_2021 = {
    "A01:2021": (
        "Broken Access Control",
        "https://owasp.org/Top10/A01_2021-Broken_Access_Control/",
    ),
    "A04:2021": (
        "Insecure Design",
        "https://owasp.org/Top10/A04_2021-Insecure_Design/",
    ),
    "A07:2021": (
        "Identification and Authentication Failures",
        "https://owasp.org/Top10/A07_2021-Identification_and_Authentication_Failures/",
    ),
    "A08:2021": (
        "Software and Data Integrity Failures",
        "https://owasp.org/Top10/A08_2021-Software_and_Data_Integrity_Failures/",
    ),
}

OWASP_2021_OF_CWE = {
    "200": "A01:2021",
    "284": "A01:2021",
    "285": "A01:2021",
    "639": "A01:2021",
    "862": "A01:2021",
    "863": "A01:2021",
    "840": "A04:2021",
    "287": "A07:2021",
    "915": "A08:2021",
}

#: The same for OWASP Top 10 2025. CWE-459 and CWE-840 are in no Top 10 2025 category.
_OWASP_2025 = {
    "A01:2025": (
        "Broken Access Control",
        "https://owasp.org/Top10/2025/A01_2025-Broken_Access_Control/",
    ),
    "A06:2025": (
        "Insecure Design",
        "https://owasp.org/Top10/2025/A06_2025-Insecure_Design/",
    ),
    "A07:2025": (
        "Authentication Failures",
        "https://owasp.org/Top10/2025/A07_2025-Authentication_Failures/",
    ),
    "A08:2025": (
        "Software or Data Integrity Failures",
        "https://owasp.org/Top10/2025/A08_2025-Software_or_Data_Integrity_Failures/",
    ),
}

OWASP_2025_OF_CWE = {
    "200": "A01:2025",
    "284": "A01:2025",
    "285": "A01:2025",
    "639": "A01:2025",
    "862": "A01:2025",
    "863": "A01:2025",
    "362": "A06:2025",
    "367": "A06:2025",
    "287": "A07:2025",
    "915": "A08:2025",
}


def owasp_identifiers(cwe: str) -> list[dict]:
    """The report identifiers for the CWE's OWASP Top 10 2021 and 2025 categories, in that order."""
    identifiers = []
    for of_cwe, categories in (
        (OWASP_2021_OF_CWE, _OWASP_2021),
        (OWASP_2025_OF_CWE, _OWASP_2025),
    ):
        category = of_cwe.get(cwe)
        if category:
            name, url = categories[category]
            identifiers.append(
                {
                    "type": "owasp",
                    "name": f"{category} - {name}",
                    "value": category,
                    "url": url,
                }
            )
    return identifiers

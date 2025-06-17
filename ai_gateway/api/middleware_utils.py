import structlog


def get_valid_namespace_ids(ids: list[str]) -> list[int]:
    """Filter and convert string IDs to integers, removing any invalid values.

    Args:
        ids (list[str]): List of string IDs to filter and convert.

    Returns:
        A list of valid integer namespaces IDs.
    """
    if not ids:
        return []

    logger = structlog.get_logger()

    valid_ids = []
    invalid_ids = []

    for str_id in ids:
        try:
            valid_ids.append(int(str_id))
        except ValueError:
            invalid_ids.append(str_id)

    if invalid_ids:
        logger.info(
            "Filtered out invalid namespace IDs.", extra={"invalid_ids": invalid_ids}
        )

    return valid_ids

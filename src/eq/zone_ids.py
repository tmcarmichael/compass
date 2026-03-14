"""Zone ID -> short name mapping for the reference client.

Used by TravelManager.detect_zone() to resolve the memory-mapped
zone ID to a config-friendly zone short name.
"""

# Zone ID mappings removed from public release. The perception layer
# resolves zone IDs at runtime; these are environment-specific bindings.
ZONE_ID_MAP: dict[int, str] = {}

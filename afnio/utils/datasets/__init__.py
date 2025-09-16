# afnio/utils/datasets/__init__.py

from afnio.utils.datasets.facility_support import FacilitySupport
from afnio.utils.datasets.trec import TREC

__all__ = ["FacilitySupport", "TREC"]

# Please keep this list sorted
assert __all__ == sorted(__all__)

from __future__ import annotations

from collections import Counter
import logging
import re
from threading import RLock

from gpu_fault.policy.models import CatalogProductFamily


LOGGER = logging.getLogger(__name__)


def catalog_product_family(
    product: str | None,
    product_families: list[CatalogProductFamily] | None = None,
) -> str | None:
    if not product:
        return None
    if product_families is None:
        from gpu_fault.policy.catalog import load_xid_policy

        product_families = load_xid_policy().product_families
    normalized = product.strip().upper()
    prefixes = sorted(
        (
            (prefix.strip().upper(), item.family.strip().upper())
            for item in product_families
            for prefix in item.model_prefixes
            if prefix.strip()
        ),
        key=lambda value: len(value[0]),
        reverse=True,
    )
    for prefix, family in prefixes:
        if re.search(
            rf"\b{re.escape(prefix)}\s*[-_]?\s*\d{{2,4}}\b",
            normalized,
            re.IGNORECASE,
        ):
            return family
    return None


def catalog_supports_product(
    product: str | None,
    supported_products: list[str],
    product_families: list[CatalogProductFamily] | None = None,
) -> bool:
    normalized = (product or "").strip().upper()
    supported = {item.strip().upper() for item in supported_products}
    if not normalized or not supported:
        return False
    if normalized in supported:
        return True
    family = catalog_product_family(normalized, product_families)
    return family is not None and family in supported


class ProductFamilyResolver:
    def __init__(
        self,
        families: list[CatalogProductFamily],
    ) -> None:
        self.families = families
        self._unknown: Counter[str] = Counter()
        self._lock = RLock()

    def family(self, product: str | None) -> str | None:
        return catalog_product_family(product, self.families)

    def supports(
        self,
        product: str | None,
        supported_products: list[str],
        *,
        record_unknown: bool = False,
    ) -> bool:
        if record_unknown:
            self.observe(product)
        return catalog_supports_product(
            product,
            supported_products,
            self.families,
        )

    def observe(self, product: str | None) -> str | None:
        family = self.family(product)
        normalized = (product or "").strip().upper()
        if normalized and family is None:
            with self._lock:
                self._unknown[normalized] += 1
                first = self._unknown[normalized] == 1
            if first:
                LOGGER.error(
                    "GPU product %s does not match any catalog product "
                    "family; XID automation will fail closed",
                    normalized,
                )
        return family

    def unknown_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._unknown)


class ProductFamilyPolicyMixin:
    _product_families: ProductFamilyResolver

    def observe_product(self, product: str | None) -> str | None:
        return self._product_families.observe(product)

    def unknown_product_counts(self) -> dict[str, int]:
        return self._product_families.unknown_counts()

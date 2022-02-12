from enum import Enum


class Multi2ConvAIMapping(Enum):
    """Mapping points from the domain name to the supported language group of multilingual models"""

    CORONA = ["de", "en", "it", "fr"]
    LOGISTIK = ["tr", "de", "en", "pl", "hr"]
    QUALITY = ["de", "en", "it", "fr"]
    STUDENT = ["de", "en", "it", "fr"]

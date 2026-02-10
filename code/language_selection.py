"""
Language selection module for multilingual tokenization experiments.
This module defines the languages to be used in the experiments.
"""

# Define a mix of high-resource and low-resource languages
LANGUAGES = {
    # High-resource languages
    "en": {"name": "English", "resource_level": "high"},
    "es": {"name": "Spanish", "resource_level": "high"},
    "fr": {"name": "French", "resource_level": "high"},
    "de": {"name": "German", "resource_level": "high"},
    "zh": {"name": "Chinese", "resource_level": "high"},
    "ru": {"name": "Russian", "resource_level": "high"},
    # Medium-resource languages
    "ar": {"name": "Arabic", "resource_level": "medium"},
    "hi": {"name": "Hindi", "resource_level": "medium"},
    # Low-resource languages
    "swh_Latn": {"name": "Swahili", "resource_level": "low"},
    "yor_Latn": {"name": "Yoruba", "resource_level": "low"},
    "hau_Latn": {"name": "Hausa", "resource_level": "low"},
    "zul_Latn": {"name": "Zulu", "resource_level": "low"},
}

# Define the subset of languages to use in the experiments
# We'll use all languages defined above
SELECTED_LANGUAGES = list(LANGUAGES.keys())


def get_language_info():
    """Return information about the selected languages."""
    high_resource = [
        code
        for code, info in LANGUAGES.items()
        if info["resource_level"] == "high" and code in SELECTED_LANGUAGES
    ]
    medium_resource = [
        code
        for code, info in LANGUAGES.items()
        if info["resource_level"] == "medium" and code in SELECTED_LANGUAGES
    ]
    low_resource = [
        code
        for code, info in LANGUAGES.items()
        if info["resource_level"] == "low" and code in SELECTED_LANGUAGES
    ]

    return {
        "high_resource": high_resource,
        "medium_resource": medium_resource,
        "low_resource": low_resource,
        "all": SELECTED_LANGUAGES,
        "count": len(SELECTED_LANGUAGES),
    }


if __name__ == "__main__":
    info = get_language_info()
    print(f"Selected {info['count']} languages for the experiment:")
    print(f"High-resource languages: {', '.join(info['high_resource'])}")
    print(f"Medium-resource languages: {', '.join(info['medium_resource'])}")
    print(f"Low-resource languages: {', '.join(info['low_resource'])}")

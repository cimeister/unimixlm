from pathlib import Path

from fhir.resources.codesystem import CodeSystem

from oops_fhir.utils import CodeSystemConcept


__all__ = ["NarrativeStatus"]

_resource = CodeSystem.parse_file(Path(__file__).with_suffix(".json"))


class NarrativeStatus:
    """
    NarrativeStatus

    The status of a resource narrative.

    Status: draft - Version: 4.0.1

    Copyright None

    http://hl7.org/fhir/narrative-status
    """

    generated = CodeSystemConcept(
        {
            "code": "generated",
            "definition": "The contents of the narrative are entirely generated from the core elements in the content.",
            "display": "Generated",
        }
    )
    """
    Generated

    The contents of the narrative are entirely generated from the core elements in the content.
    """

    extensions = CodeSystemConcept(
        {
            "code": "extensions",
            "definition": "The contents of the narrative are entirely generated from the core elements in the content and some of the content is generated from extensions. The narrative SHALL reflect the impact of all modifier extensions.",
            "display": "Extensions",
        }
    )
    """
    Extensions

    The contents of the narrative are entirely generated from the core elements in the content and some of the content is generated from extensions. The narrative SHALL reflect the impact of all modifier extensions.
    """

    additional = CodeSystemConcept(
        {
            "code": "additional",
            "definition": "The contents of the narrative may contain additional information not found in the structured data. Note that there is no computable way to determine what the extra information is, other than by human inspection.",
            "display": "Additional",
        }
    )
    """
    Additional

    The contents of the narrative may contain additional information not found in the structured data. Note that there is no computable way to determine what the extra information is, other than by human inspection.
    """

    empty = CodeSystemConcept(
        {
            "code": "empty",
            "definition": 'The contents of the narrative are some equivalent of "No human-readable text provided in this case".',
            "display": "Empty",
        }
    )
    """
    Empty

    The contents of the narrative are some equivalent of "No human-readable text provided in this case".
    """

    class Meta:
        resource = _resource
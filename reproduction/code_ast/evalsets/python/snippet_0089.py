from .base import SampleExtension
from application.src.misc.health import ReceivedTreatment, PriorInfection


class PatientTreatment(SampleExtension):

    submit_table_name = "samples_patient_treatment"

    clean_keys_strings = ["date_of_prior_antiviral_treat",
                          "date_of_prior_sars_cov_2_infection",
                          "date_of_prior_sars_cov_2_vaccination",
                          "antiviral_treatment_agent",
                          "virus_isolate_of_prior_infection",
                          "vaccine_received"]
    clean_keys_select = ["prior_sars_cov_2_vaccination_id"]


    @classmethod
    def extra_clean_submitted(cls, entry: dict) -> dict:
        key = "prior_sars_cov_2_antiviral_treat"
        treat = ReceivedTreatment.get_item_from_value(entry[key])
        entry[key] = treat.dbsave
        key = "prior_sars_cov_2_infection"
        infect = PriorInfection.get_item_from_value(entry[key])
        entry[key] = infect.dbsave
        return entry


    @classmethod
    def clean_entry(cls, entry):
        for key in cls.clean_keys_strings:
            entry = cls.clean_fetched_string(entry, key)

        key = "prior_sars_cov_2_antiviral_treat"
        if not key in entry: return entry
        treat = ReceivedTreatment.get_item_from_dbvalue(entry[key])
        entry[key] = treat.value

        key = "prior_sars_cov_2_infection"
        if not key in entry: return entry
        infect = PriorInfection.get_item_from_dbvalue(entry[key])
        entry[key] = infect.value
        return entry
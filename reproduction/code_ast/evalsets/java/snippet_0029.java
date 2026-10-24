package uk.co.idv.identity.adapter.json.phonenumber;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.JsonNode;
import uk.co.idv.identity.entities.phonenumber.PhoneNumbers;
import uk.co.mruoc.json.jackson.JsonNodeConverter;

import java.util.Optional;

public interface PhoneNumberExtractor {

    static PhoneNumbers toPhoneNumbers(JsonNode node, JsonParser parser) {
        return Optional.ofNullable(node.get("phoneNumbers"))
                .map(phoneNumbers -> JsonNodeConverter.toObject(phoneNumbers, parser, PhoneNumbers.class))
                .orElseGet(PhoneNumbers::new);
    }

}
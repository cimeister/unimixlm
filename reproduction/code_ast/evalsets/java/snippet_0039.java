package net.nicholaspurdy.sdrslicegrabber.model;

import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
public class SliceFileItem {

    private Long CANCELED_BY;
    private Long CORRECTED_BY;
    private Boolean PROCESSED;

    private Long DISSEMINATION_ID;
    private Long ORIGINAL_DISSEMINATION_ID;
    private String ACTION;
    private LocalDateTime EXECUTION_TIMESTAMP;
    private String CLEARED;
    private String INDICATION_OF_COLLATERALIZATION;
    private String INDICATION_OF_END_USER_EXCEPTION;
    private String INDICATION_OF_OTHER_PRICE_AFFECTING_TERM;
    private String BLOCK_TRADES_AND_LARGE_NOTIONAL_OFF_FACILITY_SWAPS; // BLOCK_TRADES_AND_LARGE_NOTIONAL_OFF-FACILITY_SWAPS
    private String EXECUTION_VENUE;
    private LocalDate EFFECTIVE_DATE;
    private LocalDate END_DATE;
    private String DAY_COUNT_CONVENTION;
    private String SETTLEMENT_CURRENCY;
    private String ASSET_CLASS;
    private String SUB_ASSET_CLASS_FOR_OTHER_COMMODITY; // SUB-ASSET_CLASS_FOR_OTHER_COMMODITY
    private String TAXONOMY;
    private String PRICE_FORMING_CONTINUATION_DATA;
    private String UNDERLYING_ASSET_1;
    private String UNDERLYING_ASSET_2;
    private String PRICE_NOTATION_TYPE;
    private String PRICE_NOTATION;
    private String ADDITIONAL_PRICE_NOTATION_TYPE;
    private String ADDITIONAL_PRICE_NOTATION; // contains ','
    private String NOTIONAL_CURRENCY_1;
    private String NOTIONAL_CURRENCY_2;
    private String ROUNDED_NOTIONAL_AMOUNT_1; // has '+' and ',' so it's a String
    private String ROUNDED_NOTIONAL_AMOUNT_2; // same as above
    private String PAYMENT_FREQUENCY_1;
    private String PAYMENT_FREQUENCY_2;
    private String RESET_FREQUENCY_1;
    private String RESET_FREQUENCY_2;
    private String EMBEDED_OPTION;
    private String OPTION_STRIKE_PRICE;
    private String OPTION_TYPE;
    private String OPTION_FAMILY;
    private String OPTION_CURRENCY;
    private String OPTION_PREMIUM;
    private LocalDate OPTION_LOCK_PERIOD;
    private LocalDate OPTION_EXPIRATION_DATE;
    private String PRICE_NOTATION2_TYPE;
    private String PRICE_NOTATION2;
    private String PRICE_NOTATION3_TYPE;
    private String PRICE_NOTATION3;

}
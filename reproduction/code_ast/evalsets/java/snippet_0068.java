package rnk.bb.views.bean.util;

import rnk.bb.domain.util.Address;
import rnk.bb.services.CacheService;
import rnk.bb.services.bean.CountryBean;

import javax.enterprise.context.SessionScoped;
import javax.inject.Inject;
import javax.validation.constraints.NotNull;
import javax.validation.constraints.Size;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@SessionScoped
public class EditAddressBean implements Serializable {
    private Long id=null;

    @NotNull
    private Integer countryId=643;

    @NotNull
    @Size(max=500)
    private String country="Россия";

    @NotNull
    @Size(max=20)
    private String zip="";

    @NotNull
    @Size(max=500)
    private String settlementPart="";

    @NotNull
    @Size(max=500)
    private String streetPart="";

    public EditAddressBean(){        
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Integer getCountryId() {
        return countryId;
    }

    public void setCountryId(Integer countryId) {
        this.countryId = countryId;
    }

    public String getCountry() {
        return country;
    }

    public void setCountry(String country) {
        this.country = country;
    }

    public String getZip() {
        return zip;
    }

    public void setZip(String zip) {
        this.zip = zip;
    }

    public String getSettlementPart() {
        return settlementPart;
    }

    public void setSettlementPart(String settlementPart) {
        this.settlementPart = settlementPart;
    }

    public String getStreetPart() {
        return streetPart;
    }

    public void setStreetPart(String streetPart) {
        this.streetPart = streetPart;
    }

    public String toString(){
        StringBuilder sb=new StringBuilder();
        List<String> list=new ArrayList<>();
        String country=getCountry();


        if ((zip!=null) && (!zip.trim().isEmpty())){
            ((ArrayList) list).add(zip.trim());
        }

        if (!country.trim().isEmpty()){
            list.add(country.trim());
        }

        if (!settlementPart.trim().isEmpty()){
            list.add(settlementPart.trim());
        }

        if (!streetPart.trim().isEmpty()){
            list.add(streetPart.trim());
        }

        return list.stream().collect(Collectors.joining(","));
    }
}